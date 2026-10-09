"""Train and evaluate MultiDx models.

Usage::

    python -m ml.train --all
    python -m ml.train --task heart
    python -m ml.train --all --quick
    python -m ml.train --all --models logreg random_forest

For every task the pipeline:
  1. splits train/test (stratified),
  2. tunes each model with stratified-k-fold CV (RandomizedSearchCV),
  3. picks the best model by CV score,
  4. calibrates probabilities with CalibratedClassifierCV,
  5. evaluates on the held-out test set and writes an artifact + metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from ml import __version__
from ml.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    MODEL_NAMES,
    PARAM_GRIDS,
    SEED,
    TASKS,
    TEST_SIZE,
    TUNING_ITER,
)
from ml.data_loader import DATA_PROCESSED, load_task
from ml.evaluate import evaluate, select_threshold
from ml.models import n_features_after_preprocessing
from ml.preprocessing import build_pipeline, split_feature_types
from ml.schema.features import active_features, build_feature_specs

STANDARD_TARGET = "target"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dataset_hash(task: str) -> str:
    path = DATA_PROCESSED / f"{task}.csv"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _n_splits(y: pd.Series) -> int:
    counts = y.value_counts()
    return max(2, min(CV_FOLDS, int(counts.min())))


def _scoring(kind: str) -> str:
    return "roc_auc" if kind == "binary" else "f1_macro"


def _cap_iter(grid: dict, n_iter: int) -> int:
    total = 1
    for values in grid.values():
        total *= max(1, len(values))
    return max(1, min(n_iter, total))


def _versions() -> dict[str, str]:
    import lightgbm
    import xgboost

    return {
        "multidx": __version__,
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
    }


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def tune_model(
    task: str,
    kind: str,
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    numeric: list[str],
    categorical: list[str],
    n_iter: int,
) -> dict:
    cv = StratifiedKFold(n_splits=_n_splits(y_tr), shuffle=True, random_state=SEED)
    n_pos = int((y_tr == 1).sum()) if kind == "binary" else 0
    n_neg = int((y_tr == 0).sum()) if kind == "binary" else 0

    pipe = build_pipeline(model_name, kind, numeric, categorical, SEED, n_pos, n_neg)
    grid = PARAM_GRIDS[model_name]
    search = RandomizedSearchCV(
        pipe,
        param_distributions=grid,
        n_iter=_cap_iter(grid, n_iter),
        scoring=_scoring(kind),
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    return {
        "model": model_name,
        "cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }


# --------------------------------------------------------------------------- #
# Train one task
# --------------------------------------------------------------------------- #
def train_task(
    task: str,
    models: list[str] | None = None,
    n_iter: int = TUNING_ITER,
) -> dict:
    models = models or MODEL_NAMES
    spec = TASKS[task]
    kind = spec.kind

    df, _ = load_task(task)
    y_raw = df[STANDARD_TARGET]
    label_encoder = None
    if kind == "binary":
        y = pd.to_numeric(y_raw, errors="coerce").astype(int)
    else:
        # XGBoost/LightGBM require integer-encoded multiclass targets.
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=df.index)

    features = [f for f in active_features(df) if f in df.columns]
    X = df[features].copy()

    # schema is built from the same feature list the model consumes
    specs = build_feature_specs(task, df)
    spec_map = {s.name: s.to_dict() for s in specs}
    numeric, categorical = split_feature_types(df, features)

    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
    )

    results: list[dict] = []
    for model_name in models:
        res = tune_model(task, kind, model_name, X_tr, y_tr, numeric, categorical, n_iter)
        results.append(res)
        print(f"    {model_name:14s} cv={res['cv_score']:.4f}  {res['best_params']}", flush=True)

    best = max(results, key=lambda r: r["cv_score"])

    # SHAP: prefer explainable models (trees/linear) when CV scores are close,
    # so the backend can always return top-k explanations.
    EXPLAINABLE = {"logreg", "random_forest", "xgboost", "lightgbm"}
    SHAP_TOLERANCE = 0.005
    shap_available = best["model"] in EXPLAINABLE
    if not shap_available:
        best_alt = max((r for r in results if r["model"] in EXPLAINABLE), key=lambda r: r["cv_score"], default=None)
        if best_alt is not None and (best["cv_score"] - best_alt["cv_score"]) <= SHAP_TOLERANCE:
            best = best_alt
            shap_available = True
    explain_model = best["best_estimator"]

    cal_cv = _n_splits(y_tr)
    calibrated = CalibratedClassifierCV(
        explain_model, method="sigmoid", cv=cal_cv
    )
    calibrated.fit(X_tr, y_tr)

    proba = calibrated.predict_proba(X_te)
    threshold: float | None = None
    if kind == "binary":
        classes = [str(c) for c in calibrated.classes_]
        pos = list(calibrated.classes_).index(1) if 1 in list(calibrated.classes_) else 1
        proba_vec = proba[:, pos]
        # Threshold chosen on OUT-OF-FOLD calibrated train probabilities, so
        # the operating point is set without looking at the test set.
        oof = calibrated.predict_proba(X_tr)[:, pos]
        threshold = select_threshold(y_tr.values, oof, min_recall=0.60)
        eval_out = evaluate("binary", y_te, proba_vec, threshold=threshold)
        y_te_eval = y_te
    else:
        classes = [str(c) for c in label_encoder.classes_]
        y_te_eval = label_encoder.inverse_transform(y_te.astype(int))
        eval_out = evaluate("multiclass", y_te_eval, proba, classes)

    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    metadata = {
        "task": task,
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_hash": _dataset_hash(task),
        "kind": kind,
        "best_model": best["model"],
        "best_params": best["best_params"],
        "cv_score": best["cv_score"],
        "cv_results": [
            {"model": r["model"], "cv_score": r["cv_score"], "best_params": r["best_params"]}
            for r in results
        ],
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(len(features)),
        "n_features_transformed": n_features_after_preprocessing(explain_model),
        "features": features,
        "feature_schema": spec_map,
        "classes": classes,
        "threshold": threshold,
        "shap_available": shap_available,
        "metrics": eval_out["metrics"],
        "confusion_matrix": eval_out["confusion_matrix"],
        "curves": eval_out["curves"],
        "segments": _segments(kind, y_te_eval, proba, classes),
        "library_versions": _versions(),
    }

    out_dir = ARTIFACTS_DIR / task / version
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "task": task,
        "version": version,
        "kind": kind,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "classes": classes,
        "threshold": threshold,
        "model": calibrated,
        "explain_model": explain_model,
        "schema": spec_map,
        "metrics": eval_out["metrics"],
        "best_model": best["model"],
        "shap_available": shap_available,
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    _update_index(task, version)
    print(
        f"  -> {task}: best={best['model']} "
        f"acc={eval_out['metrics']['accuracy']:.3f} "
        f"brier={eval_out['metrics']['brier']:.3f} version={version}",
        flush=True,
    )
    return metadata


def _segments(kind: str, y_te, proba, classes: list[str]) -> dict:
    """Recall by predicted-positive subgroup is left to the backend; here we
    expose class distribution for reference."""
    if kind == "binary":
        return {"n_pos": int((y_te == 1).sum()), "n_neg": int((y_te == 0).sum())}
    values, counts = np.unique(np.asarray(y_te).astype(str), return_counts=True)
    return {"class_counts": {str(v): int(c) for v, c in zip(values, counts, strict=False)}}


def _update_index(task: str, version: str) -> None:
    idx_path = ARTIFACTS_DIR / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    index[task] = version
    idx_path.write_text(json.dumps(index, indent=2))
    # prune superseded versions for this task
    task_dir = ARTIFACTS_DIR / task
    if task_dir.is_dir():
        for ver in task_dir.iterdir():
            if ver.is_dir() and ver.name != version:
                shutil.rmtree(ver, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train MultiDx models")
    parser.add_argument("--all", action="store_true", help="train every task")
    parser.add_argument("--task", action="append", default=[], help="task id (repeatable)")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, help="subset of models")
    parser.add_argument("--n-iter", type=int, default=TUNING_ITER, help="RandomizedSearchCV candidates")
    parser.add_argument("--quick", action="store_true", help="fast smoke run (3 candidates, 2 models)")
    args = parser.parse_args(argv)

    tasks = list(TASKS) if args.all or not args.task else args.task
    models = args.models
    n_iter = args.n_iter
    if args.quick:
        n_iter = min(n_iter, 3)
        models = ["logreg", "random_forest"]

    print(f"Training {len(tasks)} task(s): {tasks}")
    print(f"Models: {models}  candidates/model: {n_iter}")
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task '{task}'")
        print(f"\n[{task}] {TASKS[task].label}", flush=True)
        train_task(task, models=models, n_iter=n_iter)


if __name__ == "__main__":
    main()
