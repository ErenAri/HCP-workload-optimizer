"""API key authentication middleware for the hpcopt REST API.

Keys are read from the ``HPCOPT_API_KEYS`` environment variable (comma-
separated).  When the variable is **not** set, authentication is disabled
and all requests pass through.
"""

from __future__ import annotations

import json
import os

from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

# Paths that never require authentication.
_EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/ready"})


def _load_api_keys() -> set[str] | None:
    """Return the set of valid API keys, or *None* when auth is disabled."""
    raw = os.environ.get("HPCOPT_API_KEYS", "").strip()
    if not raw:
        return None
    return {k.strip() for k in raw.split(",") if k.strip()}


class APIKeyMiddleware:
    """Starlette / ASGI middleware that validates an ``X-API-Key`` header.

    * If the ``HPCOPT_API_KEYS`` environment variable is **unset or empty**,
      every request is passed through without authentication.
    * The ``/health`` and ``/ready`` endpoints are always exempt.
    * On failure a ``401`` response with a JSON body is returned.

    Usage with Starlette / FastAPI::

        app.add_middleware(APIKeyMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._valid_keys = _load_api_keys()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # Exempt health-check endpoints.
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        # When no keys are configured, auth is disabled.
        if self._valid_keys is None:
            await self.app(scope, receive, send)
            return

        # Extract the X-API-Key header from the raw ASGI headers.
        api_key: str | None = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-api-key":
                api_key = header_value.decode("latin-1")
                break

        if api_key and api_key in self._valid_keys:
            await self.app(scope, receive, send)
            return

        # Reject the request.
        response = Response(
            content=json.dumps({"detail": "Invalid or missing API key"}),
            status_code=401,
            media_type="application/json",
        )
        await response(scope, receive, send)
