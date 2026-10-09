"""Health / version / root."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.api.deps import get_registry
from backend.app.core.config import SETTINGS
from backend.app.schemas.system import HealthResponse, VersionResponse
from backend.app.services.registry import ModelRegistry

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness + load state")
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=SETTINGS.version,
        tasks_loaded=len(registry.task_schemas),
        models_loaded=registry.count(),
    )


@router.get("/version", response_model=VersionResponse, summary="App + model versions")
def version(registry: ModelRegistry = Depends(get_registry)) -> VersionResponse:
    return VersionResponse(
        version=SETTINGS.version,
        disclaimer=SETTINGS.disclaimer,
        models=registry.versions(),
        libraries=None,
    )


@router.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        {"name": "MultiDx API", "docs": "/docs", "health": "/health"}
    )