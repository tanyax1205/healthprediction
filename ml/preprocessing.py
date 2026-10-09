"""Pipeline construction: imputation + optional scaling + estimator."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import NEEDS_SCALING
from ml.models import make_model

__all__ = ["build_pipeline", "split_feature_types"]


def split_feature_types(df, features: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names."""
    numeric = [f for f in features if str(df[f].dtype) != "object" and df[f].dtype.kind in "iufb"]
    categorical = [f for f in features if f not in numeric]
    return numeric, categorical


def build_pipeline(
    model_name: str,
    kind: str,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_pos: int = 0,
    n_neg: int = 0,
) -> Pipeline:
    """Build a leakage-safe pipeline for *model_name*."""
    transformers = []
    if numeric:
        num_steps: list[tuple[str, object]] = [("imp", SimpleImputer(strategy="median"))]
        if NEEDS_SCALING[model_name]:
            num_steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric))
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

    pre: ClassifierMixin = ColumnTransformer(transformers, remainder="drop")
    clf = make_model(model_name, kind, seed, n_pos=n_pos, n_neg=n_neg)
    return Pipeline([("pre", pre), ("clf", clf)])
