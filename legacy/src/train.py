"""Fast baseline trainer for MultiDx.

This is a lightweight, "run it locally now" companion to the full paper
pipeline described in :mod:`src.config`.  Instead of the full
cross-validated grid search it fits one sensible baseline per task, caches the
fitted pipeline under ``models/<task>.joblib`` and writes metrics to
``reports/metrics/<task>.json``.

Design goals:
  * never blocks the Streamlit app for more than a few seconds per task;
  * reuses the already-processed datasets in ``data/processed``;
  * handles mixed numeric/categorical features generically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import METRICS_DIR, MODELS_DIR, SEED, TASKS, TEST_SIZE
from src.data_loader import load_task

STANDARD_TARGET = "target"


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def _build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Median-impute numerics, most-frequent-impute + one-hot categoricals."""
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _feature_meta(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Describe every input so the UI can render the right widget."""
    meta: dict[str, dict] = {}
    for col in df.columns:
        if col not in numeric:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        vals = s.dropna()
        meta[col] = {
            "kind": "numeric",
            "median": float(vals.median()) if len(vals) else 0.0,
            "min": float(vals.min()) if len(vals) else 0.0,
            "max": float(vals.max()) if len(vals) else 1.0,
        }
    for col in categorical:
        vals = df[col].dropna().astype(str)
        meta[col] = {"kind": "categorical", "options": sorted(vals.unique().tolist())}
    return meta


def _is_binary_flag_frame(df: pd.DataFrame, features: list[str], min_cols: int = 10) -> bool:
    """True when every feature is a 0/1 flag and there are enough of them.

    Used to switch the symptom checker to a single multi-select instead of
    rendering >100 checkboxes.
    """
    if len(features) < min_cols:
        return False
    for col in features:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not vals.isin([0, 1]).all():
            return False
    return True


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_task(task: str, force: bool = False) -> dict:
    """Fit and cache one task. Returns the saved bundle."""
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'")

    df, _ = load_task(task)
    y = df[STANDARD_TARGET]
    kind = TASKS[task].kind
    if kind == "binary":
        y = pd.to_numeric(y, errors="coerce").astype(int)

    X = df.drop(columns=[STANDARD_TARGET])
    features = list(X.columns)

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in features if c not in numeric]

    meta = _feature_meta(X, numeric, categorical)
    multiselect = _is_binary_flag_frame(X, features)

    pipeline = _build_pipeline(numeric, categorical)

    stratify = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
    )
    pipeline.fit(X_tr, y_tr)
    pred = pipeline.predict(X_te)

    metrics: dict = {
        "task": task,
        "kind": kind,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(len(features)),
        "accuracy": float(accuracy_score(y_te, pred)),
    }
    if kind == "binary":
        proba = pipeline.predict_proba(X_te)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
        metrics["f1"] = float(f1_score(y_te, pred))
    else:
        metrics["f1_macro"] = float(f1_score(y_te, pred, average="macro"))
        metrics["n_classes"] = int(y.nunique())

    bundle = {
        "task": task,
        "kind": kind,
        "pipeline": pipeline,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "feature_meta": meta,
        "multiselect": multiselect,
        "classes": [str(c) for c in np.unique(y)],
        "metrics": metrics,
    }

    model_path = MODELS_DIR / f"{task}.joblib"
    joblib.dump(bundle, model_path)
    (METRICS_DIR / f"{task}.json").write_text(json.dumps(metrics, indent=2))
    return bundle


def model_path(task: str) -> Path:
    return MODELS_DIR / f"{task}.joblib"


def load_bundle(task: str) -> dict | None:
    path = model_path(task)
    if path.exists():
        return joblib.load(path)
    return None


def ensure_models(
    tasks: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """Train any missing bundles and return {task: bundle}."""
    tasks = tasks or list(TASKS)
    bundles: dict[str, dict] = {}
    for i, task in enumerate(tasks, start=1):
        if progress:
            progress(task, i, len(tasks))
        bundle = None if force else load_bundle(task)
        if bundle is None:
            bundle = train_task(task, force=force)
        bundles[task] = bundle
    return bundles


def main() -> None:  # pragma: no cover
    for task in TASKS:
        print(f"[train] {task} ...", flush=True)
        bundle = train_task(task, force=True)
        print(f"[train] {task}: {bundle['metrics']}")


if __name__ == "__main__":  # pragma: no cover
    main()
