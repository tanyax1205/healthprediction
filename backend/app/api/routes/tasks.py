"""Task catalog + dynamic feature schemas."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.deps import get_registry
from backend.app.schemas.task import TaskListResponse, TaskSchema
from backend.app.services.registry import ModelRegistry

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get(
    "",
    response_model=TaskListResponse,
    summary="All prediction tasks with their feature schemas",
)
def list_tasks(registry: ModelRegistry = Depends(get_registry)) -> TaskListResponse:
    return TaskListResponse(tasks=registry.task_schemas)


@router.get(
    "/{task_id}",
    response_model=TaskSchema,
    summary="Feature schema for one task",
)
def get_task(task_id: str, registry: ModelRegistry = Depends(get_registry)) -> TaskSchema:
    schema = registry.task_schemas.get(task_id)
    if schema is None:
        raise HTTPException(status_code=404, detail=f"unknown task '{task_id}'")
    return schema