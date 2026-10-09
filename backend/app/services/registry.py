"""Model registry: loads every trained artifact once at startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from ml.config import TASKS
from ml.schema.features import task_schema

logger = logging.getLogger("multidx.registry")


class ModelRegistry:
    """In-memory registry of trained artifacts, keyed by task id."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.index: dict[str, str] = {}
        self._bundles: dict[str, dict] = {}
        self.task_schemas: dict[str, dict] = {}

    def load(self) -> ModelRegistry:
        index_path = self.artifacts_dir / "index.json"
        if not index_path.exists():
            raise RuntimeError(
                f"No artifacts index at {index_path}. Train models first: "
                "python -m ml.train --all"
            )
        self.index = json.loads(index_path.read_text())
        self.task_schemas = {task: task_schema(task) for task in TASKS}
        for task in TASKS:
            version = self.index.get(task)
            if version is None:
                logger.warning("no artifact for task %s", task)
                continue
            bundle_path = self.artifacts_dir / task / version / "model.joblib"
            if not bundle_path.exists():
                raise RuntimeError(f"Artifact missing: {bundle_path}")
            bundle = joblib.load(bundle_path)
            meta_path = bundle_path.parent / "metadata.json"
            if meta_path.exists():
                bundle["metadata"] = json.loads(meta_path.read_text())
            self._bundles[task] = bundle
            logger.info(
                "loaded %s @ %s (%s)", task, version, bundle.get("best_model")
            )
        return self

    @property
    def ready(self) -> bool:
        return len(self._bundles) == len(TASKS)

    def tasks(self) -> list[str]:
        return sorted(self.index)

    def versions(self) -> dict[str, str]:
        return dict(self.index)

    def get(self, task: str) -> dict:
        if task not in self._bundles:
            raise KeyError(task)
        return self._bundles[task]

    def available_versions(self) -> dict[str, str]:
        return {task: self.index[task] for task in self._bundles}

    def count(self) -> int:
        return len(self._bundles)