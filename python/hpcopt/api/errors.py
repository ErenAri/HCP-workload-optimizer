"""RFC 7807 error response helpers and exception handlers for the HPC Workload Optimizer API."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def request_trace_id(request: Request) -> str:
    """Extract or generate a trace ID from the request."""
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return request.headers.get("X-Trace-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]


def error_content(
    code: str,
    message: str,
    trace_id: str,
    details: dict[str, object] | Sequence[object] | None = None,
    status: int | None = None,
) -> dict[str, Any]:
    """Build an RFC 7807 Problem Details response body."""
    body: dict[str, Any] = {
        "type": f"urn:hpcopt:error:{code.lower().replace('_', '-')}",
        "title": code,
        "status": status,
        "detail": message,
        "instance": trace_id,
    }
    if details is not None:
        if isinstance(details, Sequence) and not isinstance(details, (str, bytes, bytearray)):
            body["errors"] = list(details)
        else:
            body["errors"] = details
    return body


def set_telemetry_headers(
    response: Response,
    trace_id: str,
    model_version: str | None = None,
    fallback_used: bool | None = None,
) -> None:
    """Attach standard telemetry headers to response."""
    response.headers["X-Correlation-ID"] = trace_id
    response.headers["X-Trace-ID"] = trace_id
    if model_version is not None:
        response.headers["X-Model-Version"] = model_version
    if fallback_used is not None:
        response.headers["X-Fallback-Used"] = "true" if fallback_used else "false"


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors with RFC 7807 format."""
    trace_id = request_trace_id(request)
    response = JSONResponse(
        status_code=422,
        content=error_content(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            trace_id=trace_id,
            details=exc.errors(),
            status=422,
        ),
    )
    set_telemetry_headers(response, trace_id)
    return response


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions with RFC 7807 format."""
    trace_id = request_trace_id(request)
    response = JSONResponse(
        status_code=exc.status_code,
        content=error_content(
            code="HTTP_ERROR",
            message=str(exc.detail),
            trace_id=trace_id,
            status=exc.status_code,
        ),
    )
    set_telemetry_headers(response, trace_id)
    return response


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions with RFC 7807 format and logging."""
    trace_id = request_trace_id(request)
    logger.exception("Unhandled error for path=%s trace_id=%s", request.url.path, trace_id, exc_info=exc)
    response = JSONResponse(
        status_code=500,
        content=error_content(
            code="INTERNAL_ERROR",
            message="Internal server error",
            trace_id=trace_id,
            status=500,
        ),
    )
    set_telemetry_headers(response, trace_id)
    return response
