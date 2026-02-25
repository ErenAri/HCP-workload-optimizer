"""HPC Workload Optimizer FastAPI application.

This module assembles the application from focused submodules:
- ``api.models`` -- Pydantic request/response schemas
- ``api.errors`` -- RFC 7807 error helpers and exception handlers
- ``api.middleware`` -- Auth, rate limiting, body size, correlation IDs
- ``api.endpoints`` -- Route handlers (health, predict, admin, etc.)
- ``api.auth`` -- API key authentication
- ``api.rate_limit`` -- Token-bucket rate limiter
- ``api.model_cache`` -- Thread-safe runtime predictor cache
- ``api.deprecation`` -- Sunset/Deprecation header management
- ``api.metrics`` -- Prometheus metrics
- ``api.tracing`` -- OpenTelemetry instrumentation
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from hpcopt import __version__
from hpcopt.api.endpoints import register_routes
from hpcopt.api.errors import (
    http_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from hpcopt.api.middleware import body_size_limit_middleware, request_middleware

logger = logging.getLogger(__name__)


# ---------- Startup validation ----------


def _validate_startup_env() -> None:
    """Validate environment variables at startup; log warnings for invalid values."""
    rate_limit_raw = os.getenv("HPCOPT_RATE_LIMIT")
    if rate_limit_raw is not None:
        try:
            val = int(rate_limit_raw)
            if val <= 0:
                logger.warning("HPCOPT_RATE_LIMIT=%s must be > 0; using default", rate_limit_raw)
        except ValueError:
            logger.warning("HPCOPT_RATE_LIMIT=%s is not a valid integer; using default", rate_limit_raw)

    timeout_raw = os.getenv("HPCOPT_REQUEST_TIMEOUT_SEC")
    if timeout_raw is not None:
        try:
            timeout_val = float(timeout_raw)
            if timeout_val <= 0:
                logger.warning("HPCOPT_REQUEST_TIMEOUT_SEC=%s must be > 0; using default", timeout_raw)
        except ValueError:
            logger.warning("HPCOPT_REQUEST_TIMEOUT_SEC=%s is not a valid number; using default", timeout_raw)

    env = os.getenv("HPCOPT_ENV")
    valid_envs = {"dev", "staging", "prod"}
    if env is not None and env not in valid_envs:
        logger.warning("HPCOPT_ENV=%s is not in %s; defaulting to 'dev'", env, valid_envs)

    # Validate config files if available
    fidelity_config = Path("configs/simulation/fidelity_gate.yaml")
    if fidelity_config.exists():
        try:
            from hpcopt.utils.config_validation import validate_config
            result = validate_config(fidelity_config, "fidelity_gate_config")
            if not result.get("valid", True):
                logger.warning("Fidelity gate config validation errors: %s", result.get("errors"))
        except (ImportError, FileNotFoundError):
            pass


# ---------- Lifespan (graceful shutdown) ----------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager with SIGTERM/SIGINT handling."""
    app.state.shutdown_requested = False
    app.state.started_at_utc = dt.datetime.now(tz=dt.UTC)

    def _signal_handler(signum: int, _frame: object) -> None:
        app.state.shutdown_requested = True
        logger.info("Shutdown signal received (sig=%s). Draining in-flight requests...", signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # Not all platforms support all signals

    # Optional OpenTelemetry instrumentation
    try:
        from hpcopt.api.tracing import init_tracing
        init_tracing(app)
    except Exception:
        logger.debug("OpenTelemetry instrumentation skipped", exc_info=True)

    # Validate critical environment configuration at startup
    _validate_startup_env()

    # Pre-warm runtime predictor cache
    from hpcopt.api.model_cache import warm_cache
    if warm_cache():
        logger.info("Runtime predictor pre-warmed successfully")
    else:
        logger.info("No runtime model found; will use fallback heuristic")

    logger.info("HPC Workload Optimizer API starting (version=%s)", __version__)
    yield
    logger.info("Shutting down HPC Workload Optimizer API. Draining in-flight requests...")
    app.state.shutdown_requested = True
    await asyncio.sleep(2)  # Grace period for in-flight request draining
    logger.info("Shutdown complete. Flushing metrics.")


# ---------- App assembly ----------

app = FastAPI(
    title="HPC Workload Optimizer API",
    version=__version__,
    description="Systems-first API for runtime/resource-fit predictions and HPC advisory.",
    lifespan=lifespan,
)

# Register middleware
app.middleware("http")(body_size_limit_middleware)
app.middleware("http")(request_middleware)

# Register exception handlers
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Register route handlers
register_routes(app)
