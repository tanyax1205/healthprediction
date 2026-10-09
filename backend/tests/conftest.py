"""Pytest fixtures: train a tiny heart model into an isolated artifacts dir."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Must be set before any import of ml.config / backend.app.main.
_ARTIFACTS = Path(tempfile.mkdtemp(prefix="multidx-backend-"))
os.environ["MULTIDX_ARTIFACTS"] = str(_ARTIFACTS)
os.environ["API_RATE_LIMIT"] = "0"  # disable limiter in tests

from backend.app.main import create_app  # noqa: E402
from ml.train import train_task  # noqa: E402


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    train_task("heart", models=["logreg"], n_iter=2)
    return _ARTIFACTS


@pytest.fixture(scope="session")
def client(artifacts_dir: Path):
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as c:
        yield c