"""Recommendation retrieval API routes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from hpcopt.api.errors import error_content, request_trace_id, set_telemetry_headers

logger = logging.getLogger(__name__)


def register_recommendation_routes(app: Any) -> None:
    """Register recommendation routes on a FastAPI app instance."""

    @app.get("/v1/recommendations/{run_id}", response_model=None)
    def get_recommendation(run_id: str, request: Request, response: Response) -> dict[str, Any] | JSONResponse:
        forbidden = {"\\", "/", "..", "*", "?", "[", "]"}
        for ch in forbidden:
            if ch in run_id:
                return JSONResponse(
                    status_code=400,
                    content=error_content(
                        code="INVALID_RUN_ID",
                        message=f"run_id contains forbidden character: {ch!r}",
                        trace_id=request_trace_id(request),
                        status=400,
                    ),
                )

        artifacts_dir = Path(os.getenv("HPCOPT_ARTIFACTS_DIR", "outputs"))
        recommendation_path = artifacts_dir / "recommendations" / f"{run_id}.json"

        if not recommendation_path.exists():
            return JSONResponse(
                status_code=404,
                content=error_content(
                    code="NOT_FOUND",
                    message=f"No recommendation found for run_id: {run_id}",
                    trace_id=request_trace_id(request),
                    status=404,
                ),
            )

        try:
            rec_payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load recommendation %s: %s", run_id, exc)
            return JSONResponse(
                status_code=500,
                content=error_content(
                    code="INTERNAL_ERROR",
                    message="Failed to load recommendation",
                    trace_id=request_trace_id(request),
                    status=500,
                ),
            )

        set_telemetry_headers(response, trace_id=request_trace_id(request))
        return cast(dict[str, Any], rec_payload)
