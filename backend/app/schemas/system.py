"""System endpoints models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    tasks_loaded: int
    models_loaded: int


class VersionResponse(BaseModel):
    version: str
    disclaimer: str
    models: dict[str, str]
    libraries: dict[str, Any] | None = None