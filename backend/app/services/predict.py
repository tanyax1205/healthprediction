"""Prediction service: validation, inference, SHAP, outlier warnings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.config import DISCLAIMER, RISK_CONFIG, TASKS
from ml.explain import explain_row

__all__ = [
    "FeatureValidationError",
    "validate_features",
    "explain_pipeline",
    "predict_bundle",
]


class FeatureValidationError(ValueError):
    """Raised when input features fail schema validation (422)."""

    def __init__(self, details: list[dict[str, str]]) -> None:
        super().__init__(details)
        self.details = details


def _coerce_binary(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float, np.number)):
        if val in (0, 1):
            return float(val)
        return None
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "positive"}:
        return 1.0
    if s in {"0", "false", "no", "n", "negative"}:
        return 0.0
    return None


def validate_features(
    bundle: dict[str, Any], values: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate *values* against the task schema and build one model-ready row.

    Returns ``(row_df, warnings)`` or raises :class:`FeatureValidationError`.
    """
    features = bundle["features"]
    schema = bundle.get("schema") or {}
    specs = {f: schema.get(f, {}) for f in features}
    all_binary = bool(specs) and all(s.get("type") == "binary" for s in specs.values())

    provided = {k for k, v in values.items() if v is not None and v != ""}
    row: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for feat in features:
        spec = specs.get(feat, {})
        ftype = spec.get("type", "numeric")
        present = feat in provided

        if ftype == "numeric":
            if not present:
                row[feat] = np.nan
                continue
            try:
                value = float(values[feat])
            except (TypeError, ValueError):
                errors.append({"feature": feat, "message": f"'{feat}' must be a number"})
                continue
            row[feat] = value
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and hi is not None and not (lo - 1e-9 <= value <= hi + 1e-9):
                warnings.append(
                    {
                        "feature": feat,
                        "value": value,
                        "min": lo,
                        "max": hi,
                        "message": f"'{feat}' = {value} is outside the training range [{lo}, {hi}]",
                    }
                )
        elif ftype == "binary":
            if not present:
                row[feat] = 0.0 if all_binary else np.nan
                continue
            coerced = _coerce_binary(values[feat])
            if coerced is None:
                errors.append({"feature": feat, "message": f"'{feat}' must be true/false or 0/1"})
            else:
                row[feat] = coerced
        else:  # categorical
            if not present:
                row[feat] = np.nan
                continue
            sval = str(values[feat]).strip()
            options = spec.get("options") or []
            if options and sval not in options:
                errors.append(
                    {"feature": feat, "message": f"'{feat}' must be one of {options}"}
                )
            else:
                row[feat] = sval

    # hard requirements from the task registry
    required = [f for f in features if specs.get(f, {}).get("required")]
    for feat in required:
        v = row.get(feat)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append({"feature": feat, "message": f"required field '{feat}' is missing"})

    min_coverage = TASKS[bundle["task"]].min_coverage
    coverage = len(provided) / len(features) if features else 1.0
    if coverage < min_coverage:
        errors.append(
            {
                "feature": "_coverage",
                "message": (
                    f"insufficient data: {coverage:.0%} features provided "
                    f"< required {min_coverage:.0%}"
                ),
            }
        )

    if errors:
        raise FeatureValidationError(errors)

    row_df = pd.DataFrame([row])[features]
    return row_df, warnings


def explain_pipeline(bundle: dict[str, Any], row_df: pd.DataFrame) -> list[dict[str, Any]]:
    if not bundle.get("shap_available"):
        return []
    classes = bundle["classes"]
    if bundle["kind"] == "binary":
        class_index = classes.index("1")
    else:
        proba = bundle["model"].predict_proba(row_df)[0]
        class_index = int(proba.argmax())
    contribs = explain_row(
        bundle["explain_model"], row_df, top_k=5, class_index=class_index
    )
    for c in contribs:
        magnitude = abs(c["contribution"])
        verb = "increases" if c["contribution"] >= 0 else "decreases"
        c["sentence"] = (
            f"{c['feature'].replace('_', ' ')} {verb} the estimated risk "
            f"by {magnitude:.2f} on the model's scale."
        )
    return contribs


def predict_bundle(
    bundle: dict[str, Any], row_df: pd.DataFrame, warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    proba = np.asarray(bundle["model"].predict_proba(row_df)[0], dtype=float)
    classes = bundle["classes"]
    kind = bundle["kind"]

    if kind == "binary":
        pos = classes.index("1")
        p = float(proba[pos])
        threshold = float(bundle.get("threshold") or 0.5)
        predicted = bool(p >= threshold)
        label = "Positive" if predicted else "Negative"
        if p >= RISK_CONFIG["high_threshold"]:
            risk_level = "High"
        elif p >= RISK_CONFIG["moderate_threshold"]:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        top_classes = [
            {"label": "Positive", "probability": p},
            {"label": "Negative", "probability": 1.0 - p},
        ]
        top_classes.sort(key=lambda c: -c["probability"])
    else:
        idx = int(proba.argmax())
        p = float(proba[idx])
        label = classes[idx]
        predicted = True
        risk_level = None
        threshold = None
        order = np.argsort(proba)[::-1][:3]
        top_classes = [
            {"label": classes[i], "probability": float(proba[i])} for i in order
        ]

    shap = explain_pipeline(bundle, row_df)
    return {
        "label": label,
        "predicted": predicted,
        "probability": p,
        "risk_level": risk_level,
        "threshold": threshold,
        "top_classes": top_classes,
        "shap": shap,
        "outlier_warnings": warnings,
        "model_version": bundle["version"],
        "best_model": bundle["best_model"],
        "disclaimer": DISCLAIMER,
    }