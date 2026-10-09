"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _cors_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: list[str] = field(
        default_factory=lambda: _cors_origins(
            os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://localhost:4173")
        )
    )
    # "0" or "false" disables the in-process rate limiter (tests / dev).
    rate_limit_per_minute: int = int(os.getenv("API_RATE_LIMIT", "60") or 0)
    log_level: str = os.getenv("API_LOG_LEVEL", "info").upper()
    artifacts_dir: Path = Path(
        os.getenv("MULTIDX_ARTIFACTS", Path(__file__).resolve().parents[3] / "ml" / "artifacts")
    )
    version: str = "2.0.0"
    disclaimer: str = "Preliminary model-based indication. Not a medical diagnosis."
    batch_max_rows: int = int(os.getenv("API_BATCH_MAX_ROWS", "5000"))
    batch_max_bytes: int = int(os.getenv("API_BATCH_MAX_BYTES", str(10 * 1024 * 1024)))

    # Optional: trust a proxy-provided client IP (comma separated).
    trusted_proxy: bool = os.getenv("API_TRUST_PROXY", "false").lower() in {"1", "true", "yes"}


SETTINGS = Settings()