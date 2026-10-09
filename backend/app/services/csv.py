"""Batch prediction: CSV in, CSV out."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from backend.app.services.predict import (
    FeatureValidationError,
    predict_bundle,
    validate_features,
)

__all__ = ["BatchRowError", "run_batch_csv"]

OUTPUT_COLUMNS = [
    "predicted_label",
    "probability",
    "risk_level",
    "model_version",
    "best_model",
    "warnings",
]


class BatchRowError(Exception):
    """Fatal problems with the batch file itself (not per-row)."""


def run_batch_csv(
    bundle: dict[str, Any], file_bytes: bytes, max_rows: int, max_bytes: int
) -> dict[str, Any]:
    if len(file_bytes) > max_bytes:
        raise BatchRowError(f"file too large: {len(file_bytes)} bytes (max {max_bytes})")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - pandas errors vary
        raise BatchRowError(f"could not parse CSV: {exc}") from exc

    if df.empty:
        raise BatchRowError("CSV contains no data rows")

    n_rows = len(df)
    if n_rows > max_rows:
        raise BatchRowError(f"too many rows: {n_rows} (max {max_rows})")

    # header sanity: all required features must be present as columns
    schema = bundle.get("schema") or {}
    required_cols = [f for f, s in schema.items() if s.get("required")]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise BatchRowError(f"missing required column(s): {missing}")

    # normalise headers (strip whitespace/case) but keep order
    df.columns = [str(c).strip() for c in df.columns]
    original_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]

    out_rows: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        values = {str(k): v for k, v in raw.items() if v is not None and not pd.isna(v)}
        warnings_text = ""
        result: dict[str, Any] = {}
        try:
            row_df, warnings = validate_features(bundle, values)
            result = predict_bundle(bundle, row_df, warnings)
            warnings_text = "; ".join(w["message"] for w in result.get("outlier_warnings", []))
        except FeatureValidationError as exc:
            warnings_text = "; ".join(d["message"] for d in exc.details)
        out_rows.append(
            {
                **raw,
                "predicted_label": result.get("label", ""),
                "probability": result.get("probability"),
                "risk_level": result.get("risk_level") or "",
                "model_version": result.get("model_version", bundle["version"]),
                "best_model": result.get("best_model", bundle["best_model"]),
                "warnings": warnings_text,
            }
        )

    # ensure float columns survive even when all values are None
    out_df = pd.DataFrame(out_rows)
    for col in ("probability",):
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    return {
        "columns": list(out_df.columns),
        "data": out_df,
        "original_columns": original_cols,
        "n_rows": n_rows,
        "n_ok": int((out_df["predicted_label"] != "").sum()),
    }


def csv_download(data: pd.DataFrame) -> str:
    return data.to_csv(index=False)