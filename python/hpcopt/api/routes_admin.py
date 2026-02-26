"""Admin API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from hpcopt.api.models import LogLevelRequest

logger = logging.getLogger(__name__)


def register_admin_routes(app: Any) -> None:
    """Register admin routes on a FastAPI app instance."""

    @app.post("/v1/admin/log-level", response_model=None)
    def set_log_level(payload: LogLevelRequest, request: Request) -> dict[str, str] | JSONResponse:
        level_name = payload.level.upper()
        numeric = getattr(logging, level_name, None)
        if not isinstance(numeric, int):
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "INVALID_LEVEL", "message": f"Unknown level: {payload.level}"}},
            )
        old_level = logging.getLevelName(logging.getLogger().level)
        logging.getLogger().setLevel(numeric)
        logger.info("Log level changed to %s", level_name)

        try:
            from hpcopt.utils.audit import audit_log

            audit_log(
                "admin.log_level_change",
                actor=request.headers.get("X-API-Key", "unknown")[:8] + "...",
                details={"old_level": old_level, "new_level": level_name},
            )
        except Exception:
            pass

        return {"status": "ok", "level": level_name}
