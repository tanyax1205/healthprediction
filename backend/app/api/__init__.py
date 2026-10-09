"""FastAPI route modules."""

from backend.app.api import deps
from backend.app.api.routes import metrics, predict, system, tasks

__all__ = ["deps", "metrics", "predict", "system", "tasks"]