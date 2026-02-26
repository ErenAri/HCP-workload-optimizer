"""HTTP middleware for the HPC Workload Optimizer API.

Handles body size limits, auth, rate limiting, correlation IDs, deprecation
headers, logging, metrics, request timeout, and shutdown draining.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, cast

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from hpcopt.api.auth import EXEMPT_PATHS, check_admin_auth, check_api_key_auth
from hpcopt.api.deprecation import load_deprecation_config
from hpcopt.api.errors import error_content, set_telemetry_headers
from hpcopt.api.rate_limit import check_rate_limit

logger = logging.getLogger(__name__)

# ---------- Constants ----------

MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


async def body_size_limit_middleware(request: Request, call_next: Any) -> Response:
    """Reject requests with Content-Length exceeding the configured limit."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body exceeds 1 MB limit"}},
                )
        except ValueError:
            pass
    return cast(Response, await call_next(request))


async def request_middleware(request: Request, call_next: Any) -> Response:
    """Combined middleware: correlation ID, auth, rate limiting, logging, metrics."""
    from hpcopt.api.app import app

    start_time = time.time()
    path = request.url.path
    request_timeout = float(__import__("os").getenv("HPCOPT_REQUEST_TIMEOUT_SEC", "30"))

    trace_id = request.headers.get("X-Trace-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]
    request.state.trace_id = trace_id

    # Request draining: reject new non-exempt requests during shutdown
    if getattr(app.state, "shutdown_requested", False) and path not in EXEMPT_PATHS:
        error_response = JSONResponse(
            status_code=503,
            content=error_content(
                code="SERVICE_UNAVAILABLE",
                message="Server is shutting down",
                trace_id=trace_id,
                status=503,
            ),
        )
        set_telemetry_headers(error_response, trace_id)
        return error_response

    # Auth check (exempt health/ready/metrics/system status)
    response: Response
    provided_key = request.headers.get("X-API-Key", "")
    if not check_api_key_auth(path, provided_key):
        logger.warning("Auth failed for path=%s trace_id=%s", path, trace_id)
        try:
            from hpcopt.api.metrics import record_auth_failure

            record_auth_failure()
        except ImportError:
            pass
        error_response = JSONResponse(
            status_code=401,
            content=error_content(
                code="UNAUTHORIZED",
                message="Invalid or missing API key",
                trace_id=trace_id,
                status=401,
            ),
        )
        set_telemetry_headers(error_response, trace_id)
        return error_response

    # Admin RBAC: /v1/admin/* paths require admin-prefixed API key
    if not check_admin_auth(path, provided_key):
        logger.warning("Admin auth failed for path=%s trace_id=%s", path, trace_id)
        try:
            from hpcopt.api.metrics import record_auth_failure

            record_auth_failure()
        except ImportError:
            pass
        error_response = JSONResponse(
            status_code=403,
            content=error_content(
                code="FORBIDDEN",
                message="Admin privileges required",
                trace_id=trace_id,
                status=403,
            ),
        )
        set_telemetry_headers(error_response, trace_id)
        return error_response

    # Rate limiting (per-endpoint, exempt health/ready)
    if path not in EXEMPT_PATHS:
        api_key = request.headers.get("X-API-Key")
        allowed, retry_after = check_rate_limit(api_key, path)
        if not allowed:
            try:
                from hpcopt.api.metrics import record_rate_limit_rejection

                record_rate_limit_rejection()
            except ImportError:
                pass
            error_response = JSONResponse(
                status_code=429,
                content=error_content(
                    code="RATE_LIMITED",
                    message="Rate limit exceeded",
                    trace_id=trace_id,
                    status=429,
                ),
                headers={"Retry-After": str(retry_after)},
            )
            set_telemetry_headers(error_response, trace_id)
            return error_response

    try:
        response = cast(
            Response,
            await asyncio.wait_for(call_next(request), timeout=request_timeout),
        )
    except asyncio.TimeoutError:
        logger.warning("Request timed out for path=%s trace_id=%s", path, trace_id)
        response = JSONResponse(
            status_code=504,
            content=error_content(
                code="GATEWAY_TIMEOUT",
                message="Request timed out",
                trace_id=trace_id,
                status=504,
            ),
        )
    except Exception as exc:
        logger.exception("Request failed for path=%s trace_id=%s", path, trace_id, exc_info=exc)
        response = JSONResponse(
            status_code=500,
            content=error_content(
                code="INTERNAL_ERROR",
                message="Internal server error",
                trace_id=trace_id,
                status=500,
            ),
        )
    duration = time.time() - start_time

    # Add trace headers to all responses.
    set_telemetry_headers(response, trace_id)

    # Add deprecation/sunset headers for deprecated endpoints
    for entry in load_deprecation_config():
        prefix = entry.get("path_prefix", "")
        if prefix and path.startswith(prefix):
            if entry.get("deprecated_at"):
                response.headers["Deprecation"] = entry["deprecated_at"]
            if entry.get("sunset_at"):
                response.headers["Sunset"] = entry["sunset_at"]
            if entry.get("docs_url"):
                response.headers["Link"] = f'<{entry["docs_url"]}>; rel="successor-version"'
            break

    # Log request
    logger.info(
        "request method=%s path=%s status=%d duration=%.3fs trace_id=%s",
        request.method,
        path,
        response.status_code,
        duration,
        trace_id,
    )

    # Record metrics if prometheus available
    try:
        from hpcopt.api.metrics import record_request_metrics

        record_request_metrics(request.method, path, response.status_code, duration)
    except ImportError:
        pass

    return response
