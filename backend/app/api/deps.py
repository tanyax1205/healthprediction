"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from backend.app.services.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="model registry not loaded yet")
    return registry


def require_bundle(task: str, registry: ModelRegistry) -> dict:
    """Load one trained bundle or raise a 404. Not a FastAPI dependency:
    route handlers pass ``task_id`` explicitly."""
    try:
        return registry.get(task)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"no trained model for task '{task}'"
        ) from None