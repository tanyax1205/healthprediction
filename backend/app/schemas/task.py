"""Task + feature schema models (mirror ml.schema.features)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureSchema(BaseModel):
    name: str
    label: str
    type: str = Field(description="numeric | categorical | binary")
    unit: str = ""
    group: str = "Other"
    min: float | None = None
    max: float | None = None
    default: float | str | None = None
    options: list[str] = Field(default_factory=list)
    required: bool = False


class TaskSchema(BaseModel):
    task_id: str
    label: str
    short_label: str = ""
    kind: str = Field(description="binary | multiclass")
    tier: int
    dataset: str
    min_coverage: float = 0.0
    disclaimer: str
    groups: list[str] = Field(default_factory=list)
    features: list[FeatureSchema]


class TaskListResponse(BaseModel):
    tasks: dict[str, TaskSchema] = Field(description="task_id -> full TaskSchema")