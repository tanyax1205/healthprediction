"""SHAP-based explanations for a single prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TREE_TYPES = (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    DecisionTreeClassifier,
    XGBClassifier,
    LGBMClassifier,
)

__all__ = ["explain_row"]


def _dense(x):
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


def _to_vector(sv: Any, class_index: int) -> np.ndarray:
    """Reduce SHAP output (list / 3-D / 2-D) to one row vector."""
    if isinstance(sv, list):
        arr = np.asarray(sv[class_index])
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        arr = sv[..., class_index]
    else:
        arr = np.asarray(sv)
    if arr.ndim == 2:
        arr = arr[0]
    return arr.ravel()


def _clean_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def explain_row(
    explain_model,
    row_df: pd.DataFrame,
    top_k: int = 5,
    class_index: int = 0,
) -> list[dict[str, Any]]:
    """Return the top-*k* signed SHAP contributions for a single row.

    ``explain_model`` is the fitted (uncalibrated) pipeline.
    """
    pre = explain_model.named_steps["pre"]
    clf = explain_model.named_steps["clf"]

    xt = _dense(pre.transform(row_df))
    names = [_clean_name(n) for n in pre.get_feature_names_out()]

    try:
        if isinstance(clf, TREE_TYPES):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(xt)
        elif hasattr(clf, "coef_"):
            beta = np.ravel(clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_)
            sv = beta * xt
        else:
            return []
        contrib = _to_vector(sv, class_index)
    except Exception:
        return []

    if len(contrib) != len(names):
        names = names[: len(contrib)]

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    out: list[dict[str, Any]] = []
    for i in order:
        raw_value = row_df.iloc[0].get(names[i])
        out.append(
            {
                "feature": names[i],
                "value": None if raw_value is None else (
                    float(raw_value) if isinstance(raw_value, (int, float, np.number)) else str(raw_value)
                ),
                "contribution": float(contrib[i]),
                "direction": "increases" if contrib[i] >= 0 else "decreases",
            }
        )
    return out
