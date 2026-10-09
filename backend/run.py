"""Run the MultiDx API with one command.

Usage::

    .venv/bin/python -m backend.run              # host/port from .env
    .venv/bin/python -m backend.run --reload      # dev hot-reload
    .venv/bin/python -m backend.run --port 9000

Reads ``.env`` at the repo root (if present) before importing app settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the MultiDx API")
    parser.add_argument("--host", default=None, help="bind address (default: .env API_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="port (default: .env API_PORT or 8000)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        load_dotenv = None  # type: ignore[assignment]
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    import uvicorn

    from backend.app.core.config import SETTINGS

    uvicorn.run(
        "backend.app.main:app",
        host=args.host or SETTINGS.api_host,
        port=args.port or SETTINGS.api_port,
        reload=args.reload,
        log_level=SETTINGS.log_level.lower(),
    )


if __name__ == "__main__":
    main()