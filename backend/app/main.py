"""MultiDx FastAPI application.

Run locally::

    .venv/bin/uvicorn backend.app.main:app --reload --port 8000

Artifacts are loaded exactly once at startup (see ``ModelRegistry``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import metrics, predict, system, tasks
from backend.app.core.config import SETTINGS, Settings
from backend.app.core.logging import setup_logging
from backend.app.core.ratelimit import RateLimiter, RateLimitMiddleware
from backend.app.services.predict import FeatureValidationError
from backend.app.services.registry import ModelRegistry

logger = logging.getLogger("multidx.api")


def create_app(settings: Settings = SETTINGS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        registry = ModelRegistry(settings.artifacts_dir).load()
        app.state.registry = registry
        logger.info(
            "MultiDx API ready: %d tasks, %d models loaded",
            len(registry.task_schemas),
            registry.count(),
        )
        yield

    app = FastAPI(
        title="MultiDx API",
        description=(
            "Preliminary model-based health indications. "
            "Not a medical diagnosis. See /docs for the schema."
        ),
        version=settings.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RateLimiter(settings.rate_limit_per_minute),
        trusted_proxy=settings.trusted_proxy,
    )

    app.include_router(tasks.router)
    app.include_router(predict.router)
    app.include_router(metrics.router)
    app.include_router(system.router)

    @app.exception_handler(FeatureValidationError)
    async def on_feature_validation(_request: Request, exc: FeatureValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.details})

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content={"detail": {"error": "internal_error"}}
        )

    return app


app = create_app()