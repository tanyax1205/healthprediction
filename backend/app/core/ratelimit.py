"""In-process sliding-window rate limiter (ASGI middleware).

Per-instance only; for multi-worker deployments pair it with a shared store
(Redis) or an edge gateway. ``API_RATE_LIMIT=0`` disables it.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

__all__ = ["RateLimiter", "RateLimitMiddleware"]


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            retry_after = int(60 - (now - window[0])) + 1
            return False, retry_after
        window.append(now)
        return True, 0

    def reset(self) -> None:
        self._hits.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter, trusted_proxy: bool = False) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._trusted_proxy = trusted_proxy

    async def dispatch(self, request: Request, call_next):
        if self._limiter.limit > 0:
            client = request.client.host if request.client else "unknown"
            if self._trusted_proxy:
                fwd = request.headers.get("X-Forwarded-For")
                if fwd:
                    client = fwd.split(",")[0].strip()
            allowed, retry_after = self._limiter.check(client)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "error": "rate_limit_exceeded",
                            "retry_after_seconds": retry_after,
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)