"""Metrics response model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CvResult(BaseModel):
    model: str
    cv_score: float
    best_params: dict[str, Any]


class MetricsResponse(BaseModel):
    task_id: str
    version: str
    best_model: str
    dataset_hash: str
    trained_at: str
    kind: str
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    curves: dict[str, Any]
    cv_results: list[CvResult]
    threshold: float | None = None
    shap_available: bool = False