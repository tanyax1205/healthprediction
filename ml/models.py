"""Model zoo for MultiDx.

Every model is used behind a scikit-learn Pipeline built in
:mod:`ml.preprocessing`, so imputation/scaling are fitted on training folds
only (no leakage).
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.config import MODEL_LABELS, MODEL_NAMES

__all__ = ["MODEL_NAMES", "MODEL_LABELS", "make_model"]


def _scale_pos_weight(kind: str, n_pos: int, n_neg: int) -> float:
    if kind == "binary" and n_pos > 0:
        return float(n_neg) / float(n_pos)
    return 1.0


def make_model(
    name: str,
    kind: str,
    seed: int,
    n_pos: int = 0,
    n_neg: int = 0,
) -> ClassifierMixin:
    """Return an unfitted estimator for *name*.

    Class imbalance is handled by the estimator itself (``class_weight`` or
    ``scale_pos_weight``), which is fit per training fold and never sees the
    test set.
    """
    if name == "logreg":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "svm":
        # probability=False: CalibratedClassifierCV turns decision_function
        # into calibrated probabilities afterwards.
        return SVC(
            class_weight="balanced",
            random_state=seed,
            probability=False,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=_scale_pos_weight(kind, n_pos, n_neg),
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
            verbose=-1,
        )
    raise KeyError(f"unknown model '{name}'")


def n_features_after_preprocessing(pipeline: Any) -> int | None:
    """Best-effort count of transformed features (for metadata)."""
    try:
        pre = pipeline.named_steps["pre"]
        return int(len(pre.get_feature_names_out()))
    except Exception:
        return None
