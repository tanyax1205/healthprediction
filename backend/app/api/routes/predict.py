"""Prediction endpoints: single + CSV batch."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.api.deps import get_registry, require_bundle
from backend.app.core.config import SETTINGS
from backend.app.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.csv import BatchRowError, csv_download, run_batch_csv
from backend.app.services.predict import (
    FeatureValidationError,
    predict_bundle,
    validate_features,
)
from backend.app.services.registry import ModelRegistry

router = APIRouter(tags=["predict"])


@router.post(
    "/predict/{task_id}",
    response_model=PredictResponse,
    summary="Single prediction for a task",
)
def predict(
    task_id: str,
    body: PredictRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictResponse:
    bundle = require_bundle(task_id, registry)
    try:
        row_df, warnings = validate_features(bundle, body.values)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.details) from None
    result = predict_bundle(bundle, row_df, warnings)
    return PredictResponse(task_id=task_id, **result)


@router.post(
    "/predict/{task_id}/batch",
    summary="Predict from an uploaded CSV (returns CSV)",
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "Uploaded rows plus prediction columns",
        }
    },
)
async def predict_batch(
    task_id: str,
    file: UploadFile = File(..., description="CSV with one header row of feature names"),
    registry: ModelRegistry = Depends(get_registry),
) -> Response:
    bundle = require_bundle(task_id, registry)
    try:
        file_bytes = await file.read()
        outcome = run_batch_csv(
            bundle,
            file_bytes,
            SETTINGS.batch_max_rows,
            SETTINGS.batch_max_bytes,
        )
    except BatchRowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    filename = f"{task_id}_predictions.csv"
    return Response(
        content=csv_download(outcome["data"]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )