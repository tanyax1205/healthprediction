"""Model metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_registry, require_bundle
from backend.app.schemas.metrics import CvResult, MetricsResponse
from backend.app.services.registry import ModelRegistry

router = APIRouter(tags=["metrics"])


@router.get(
    "/tasks/{task_id}/metrics",
    response_model=MetricsResponse,
    summary="Offline metrics for a trained task",
)
def task_metrics(
    task_id: str,
    registry: ModelRegistry = Depends(get_registry),
) -> MetricsResponse:
    bundle = require_bundle(task_id, registry)
    meta = bundle.get("metadata") or {}
    cv_results = [CvResult(**r) for r in meta.get("cv_results", [])]
    return MetricsResponse(
        task_id=task_id,
        version=bundle["version"],
        best_model=bundle["best_model"],
        dataset_hash=meta.get("dataset_hash", ""),
        trained_at=meta.get("trained_at", ""),
        kind=bundle["kind"],
        metrics=bundle.get("metrics") or {},
        confusion_matrix=meta.get("confusion_matrix", []),
        curves=meta.get("curves", {}),
        cv_results=cv_results,
        threshold=bundle.get("threshold"),
        shap_available=bundle.get("shap_available", False),
    )