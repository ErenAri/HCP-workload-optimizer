from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import shutil
import signal
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from hpcopt import __version__
from hpcopt.api.auth import EXEMPT_PATHS, check_admin_auth, check_api_key_auth
from hpcopt.api.deprecation import load_deprecation_config
from hpcopt.api.model_cache import get_runtime_predictor
from hpcopt.api.rate_limit import check_rate_limit
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir
from hpcopt.utils.resilience import CircuitBreaker

# Circuit breaker for model prediction I/O
_prediction_circuit = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %.2f", name, raw, default)
        return default

# ---------- Constants ----------

_HEALTH_MIN_DISK_FREE_GB = _float_env("HPCOPT_HEALTH_MIN_DISK_FREE_GB", 0.1)
_REQUEST_TIMEOUT_SEC = _float_env("HPCOPT_REQUEST_TIMEOUT_SEC", 30.0)

# Fallback heuristic constants (used when no trained model is available)
_FALLBACK_BASE_RUNTIME_SEC = 1800  # default base when requested_runtime_sec is absent
_FALLBACK_QUEUE_DEPTH_CAP = 2000  # jobs beyond this have diminishing queue delay impact
_FALLBACK_QUEUE_DEPTH_SCALE = 10_000.0  # scaling factor for queue depth sensitivity
_FALLBACK_P50_FACTOR = 0.72  # multiplier for p50 estimate
_FALLBACK_P90_FACTOR = 1.08  # multiplier for p90 estimate
_FALLBACK_MIN_RUNTIME_SEC = 60  # floor for fallback predictions

# Resource-fit fragmentation risk thresholds (waste_ratio boundaries)
_WASTE_RATIO_LOW_RISK = 0.15
_WASTE_RATIO_MEDIUM_RISK = 0.35


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
            val = float(timeout_raw)
            if val <= 0:
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


app = FastAPI(
    title="HPC Workload Optimizer API",
    version=__version__,
    description="Systems-first API for runtime/resource-fit predictions and HPC advisory.",
    lifespan=lifespan,
)

# ---------- Request body size limit ----------

_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next: Any) -> Response:
    """Reject requests with Content-Length exceeding the configured limit."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body exceeds 1 MB limit"}},
                )
        except ValueError:
            pass
    return await call_next(request)


# ---------- Middleware ----------

def _request_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(trace_id, str) and trace_id:
        return trace_id
    return request.headers.get("X-Trace-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]


def _error_content(
    code: str,
    message: str,
    trace_id: str,
    details: dict[str, object] | list[object] | None = None,
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
        body["errors"] = details
    return body


def _set_telemetry_headers(
    response: Response,
    trace_id: str,
    model_version: str | None = None,
    fallback_used: bool | None = None,
) -> None:
    response.headers["X-Correlation-ID"] = trace_id
    response.headers["X-Trace-ID"] = trace_id
    if model_version is not None:
        response.headers["X-Model-Version"] = model_version
    if fallback_used is not None:
        response.headers["X-Fallback-Used"] = "true" if fallback_used else "false"


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    trace_id = _request_trace_id(request)
    response = JSONResponse(
        status_code=422,
        content=_error_content(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            trace_id=trace_id,
            details=exc.errors(),
            status=422,
        ),
    )
    _set_telemetry_headers(response, trace_id)
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    trace_id = _request_trace_id(request)
    response = JSONResponse(
        status_code=exc.status_code,
        content=_error_content(
            code="HTTP_ERROR",
            message=str(exc.detail),
            trace_id=trace_id,
            status=exc.status_code,
        ),
    )
    _set_telemetry_headers(response, trace_id)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    trace_id = _request_trace_id(request)
    logger.exception("Unhandled error for path=%s trace_id=%s", request.url.path, trace_id, exc_info=exc)
    response = JSONResponse(
        status_code=500,
        content=_error_content(
            code="INTERNAL_ERROR",
            message="Internal server error",
            trace_id=trace_id,
            status=500,
        ),
    )
    _set_telemetry_headers(response, trace_id)
    return response


@app.middleware("http")
async def request_middleware(request: Request, call_next: Any) -> Response:
    """Combined middleware: correlation ID, auth, rate limiting, logging, metrics."""
    start_time = time.time()
    path = request.url.path

    trace_id = request.headers.get("X-Trace-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]
    request.state.trace_id = trace_id

    # Request draining: reject new non-exempt requests during shutdown
    if getattr(app.state, "shutdown_requested", False) and path not in EXEMPT_PATHS:
        error_response = JSONResponse(
            status_code=503,
            content=_error_content(
                code="SERVICE_UNAVAILABLE",
                message="Server is shutting down",
                trace_id=trace_id,
                status=503,
            ),
        )
        _set_telemetry_headers(error_response, trace_id)
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
            content=_error_content(
                code="UNAUTHORIZED",
                message="Invalid or missing API key",
                trace_id=trace_id,
                status=401,
            ),
        )
        _set_telemetry_headers(error_response, trace_id)
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
            content=_error_content(
                code="FORBIDDEN",
                message="Admin privileges required",
                trace_id=trace_id,
                status=403,
            ),
        )
        _set_telemetry_headers(error_response, trace_id)
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
                content=_error_content(
                    code="RATE_LIMITED",
                    message="Rate limit exceeded",
                    trace_id=trace_id,
                    status=429,
                ),
                headers={"Retry-After": str(retry_after)},
            )
            _set_telemetry_headers(error_response, trace_id)
            return error_response

    try:
        response = cast(
            Response,
            await asyncio.wait_for(call_next(request), timeout=_REQUEST_TIMEOUT_SEC),
        )
    except asyncio.TimeoutError:
        logger.warning("Request timed out for path=%s trace_id=%s", path, trace_id)
        response = JSONResponse(
            status_code=504,
            content=_error_content(
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
            content=_error_content(
                code="INTERNAL_ERROR",
                message="Internal server error",
                trace_id=trace_id,
                status=500,
            ),
        )
    duration = time.time() - start_time

    # Add trace headers to all responses.
    _set_telemetry_headers(response, trace_id)

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


# ---------- Request/Response Models ----------

class RuntimePredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int | None = Field(default=None, description="User id when available")
    requested_runtime_sec: int | None = Field(default=None, ge=1, le=31_536_000)
    requested_cpus: int = Field(..., ge=1, le=100_000)
    requested_mem: int | None = Field(default=None, ge=1)
    queue_id: int | None = Field(default=None)
    partition_id: int | None = Field(default=None)
    group_id: int | None = Field(default=None)
    queue_depth_jobs: int | None = Field(default=None, ge=0, le=1_000_000)
    runtime_guard_k: float = Field(default=0.5, ge=0.0, le=2.0)


class RuntimePredictResponse(BaseModel):
    predictor_version: str
    runtime_p50_sec: int
    runtime_p90_sec: int
    runtime_guard_sec: int
    fallback_used: bool
    notes: list[str]


class ResourceFitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_cpus: int = Field(..., ge=1, le=100_000)
    candidate_node_cpus: list[int] = Field(..., min_length=1, max_length=1000)
    queue_depth_jobs: int | None = Field(default=None, ge=0, le=1_000_000)


class ResourceFitResponse(BaseModel):
    recommendation: dict[str, int | float]
    fragmentation_risk: Literal["low", "medium", "high"]
    notes: list[str]



# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict[str, object]:
    """Health check with model staleness, disk, memory, config validation."""
    checks: list[str] = []
    status = "ok"

    # Model check
    model_dir = resolve_runtime_model_dir()
    model_loaded = bool(model_dir is not None and model_dir.exists())
    if not model_loaded:
        checks.append("runtime_model_not_loaded")

    model_staleness_sec = None
    if model_dir is not None:
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                trained_at = meta.get("trained_at_utc")
                if trained_at:
                    trained_dt = dt.datetime.fromisoformat(trained_at)
                    if trained_dt.tzinfo is None:
                        trained_dt = trained_dt.replace(tzinfo=dt.UTC)
                    model_staleness_sec = (dt.datetime.now(tz=dt.UTC) - trained_dt).total_seconds()
                    if model_staleness_sec > 86400 * 30:
                        checks.append("model_stale_over_30d")
            except (json.JSONDecodeError, ValueError, OSError):
                pass

    # Disk check
    try:
        disk = shutil.disk_usage(".")
        disk_free_gb = disk.free / (1024**3)
        if disk_free_gb < _HEALTH_MIN_DISK_FREE_GB:
            checks.append("disk_low")
            status = "degraded"
    except OSError:
        disk_free_gb = None

    # Config check
    config_ok = Path("configs/simulation/fidelity_gate.yaml").exists()
    if not config_ok:
        checks.append("fidelity_gate_config_missing")
        status = "degraded"

    return {
        "status": status,
        "service": "hpcopt-api",
        "version": __version__,
        "model_loaded": model_loaded,
        "model_staleness_seconds": model_staleness_sec,
        "disk_free_gb": round(disk_free_gb, 2) if disk_free_gb is not None else None,
        "checks": checks,
    }


@app.get("/ready", response_model=None)
def ready() -> dict[str, str] | JSONResponse:
    """Kubernetes readiness probe.

    Returns 503 when the service is degraded (low disk, shutting down) while
    still accepting requests when running in fallback mode (no model).
    """
    # Refuse if shutdown is in progress
    if getattr(app.state, "shutdown_requested", False):
        return JSONResponse(status_code=503, content={"status": "shutting_down"})

    # Check disk health
    try:
        disk = shutil.disk_usage(".")
        disk_free_gb = disk.free / (1024**3)
        if disk_free_gb < _HEALTH_MIN_DISK_FREE_GB:
            return JSONResponse(status_code=503, content={"status": "degraded", "reason": "disk_low"})
    except OSError:
        pass

    return {"status": "ok"}


@app.get("/v1/system/status")
def system_status(request: Request) -> dict[str, object]:
    now = dt.datetime.now(tz=dt.UTC)
    started_at = getattr(app.state, "started_at_utc", now)
    if not isinstance(started_at, dt.datetime):
        started_at = now
    shutdown_requested = bool(getattr(app.state, "shutdown_requested", False))
    model_dir = resolve_runtime_model_dir()
    model_loaded = bool(model_dir is not None and model_dir.exists())
    metrics_available = False
    try:
        from hpcopt.api.metrics import is_available

        metrics_available = is_available()
    except ImportError:
        metrics_available = False
    return {
        "status": "ok",
        "service": "hpcopt-api",
        "version": __version__,
        "time_utc": now.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "uptime_seconds": max(0, int((now - started_at).total_seconds())),
        "shutdown_requested": shutdown_requested,
        "model_loaded": model_loaded,
        "runtime_model_dir": str(model_dir) if model_dir is not None else None,
        "metrics_available": metrics_available,
        "trace_id": _request_trace_id(request),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    try:
        from hpcopt.api.metrics import get_metrics_response
        return PlainTextResponse(content=get_metrics_response(), media_type="text/plain")
    except ImportError:
        return PlainTextResponse(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )


@app.post("/v1/runtime/predict", response_model=RuntimePredictResponse)
def predict_runtime(
    payload: RuntimePredictRequest,
    request: Request,
    response: Response,
) -> RuntimePredictResponse:
    # Use circuit breaker to fail fast if model I/O is repeatedly failing
    try:
        predictor, model_dir = get_runtime_predictor()
    except _prediction_circuit.CircuitOpenError:
        logger.warning("Prediction circuit open; using fallback. trace_id=%s", _request_trace_id(request))
        predictor, model_dir = None, None

    if predictor is not None:
        features = {
            "requested_cpus": payload.requested_cpus,
            "runtime_requested_sec": payload.requested_runtime_sec,
            "requested_mem": payload.requested_mem,
            "queue_id": payload.queue_id,
            "partition_id": payload.partition_id,
            "user_id": payload.user_id,
            "group_id": payload.group_id,
        }
        try:
            quantiles = predictor.predict_one(features)
            _prediction_circuit.record_success()
        except Exception:
            _prediction_circuit.record_failure()
            raise
        runtime_p50 = int(max(1, round(quantiles["p50"])))
        runtime_p90 = int(max(runtime_p50, round(quantiles["p90"])))
        runtime_guard = int(runtime_p50 + payload.runtime_guard_k * (runtime_p90 - runtime_p50))
        result = RuntimePredictResponse(
            predictor_version=f"runtime-quantile:{model_dir.name if model_dir else 'unknown'}",
            runtime_p50_sec=runtime_p50,
            runtime_p90_sec=runtime_p90,
            runtime_guard_sec=runtime_guard,
            fallback_used=False,
            notes=[
                "Prediction from persisted quantile model artifacts.",
                "Guard follows policy contract runtime_guard = p50 + k*(p90-p50).",
            ],
        )
        _set_telemetry_headers(
            response,
            trace_id=_request_trace_id(request),
            model_version=result.predictor_version,
            fallback_used=result.fallback_used,
        )
        return result

    # Record fallback usage
    try:
        from hpcopt.api.metrics import record_fallback
        record_fallback()
    except ImportError:
        pass

    fallback_used = True
    base = payload.requested_runtime_sec or _FALLBACK_BASE_RUNTIME_SEC
    queue_depth = payload.queue_depth_jobs or 0
    queue_factor = 1.0 + min(queue_depth, _FALLBACK_QUEUE_DEPTH_CAP) / _FALLBACK_QUEUE_DEPTH_SCALE
    runtime_p50 = max(_FALLBACK_MIN_RUNTIME_SEC, int(base * _FALLBACK_P50_FACTOR * queue_factor))
    runtime_p90 = max(runtime_p50, int(base * _FALLBACK_P90_FACTOR * queue_factor))
    runtime_guard = int(runtime_p50 + payload.runtime_guard_k * (runtime_p90 - runtime_p50))
    result = RuntimePredictResponse(
        predictor_version="runtime-heuristic-fallback",
        runtime_p50_sec=runtime_p50,
        runtime_p90_sec=runtime_p90,
        runtime_guard_sec=runtime_guard,
        fallback_used=fallback_used,
        notes=[
            "No trained model found; set HPCOPT_RUNTIME_MODEL_DIR or run hpcopt train runtime.",
            "Guard follows policy contract runtime_guard = p50 + k*(p90-p50).",
        ],
    )
    _set_telemetry_headers(
        response,
        trace_id=_request_trace_id(request),
        model_version=result.predictor_version,
        fallback_used=result.fallback_used,
    )
    return result


class LogLevelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(..., description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")


@app.post("/v1/admin/log-level", response_model=None)
def set_log_level(payload: LogLevelRequest, request: Request) -> dict[str, str] | JSONResponse:
    """Dynamically change the root log level at runtime."""
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


@app.post("/v1/resource-fit/predict", response_model=ResourceFitResponse)
def predict_resource_fit(
    payload: ResourceFitRequest,
    request: Request,
    response: Response,
) -> ResourceFitResponse:
    candidate_node_cpus = sorted(payload.candidate_node_cpus)

    fit_cpu = candidate_node_cpus[-1]
    for capacity in candidate_node_cpus:
        if capacity >= payload.requested_cpus:
            fit_cpu = capacity
            break

    waste = max(fit_cpu - payload.requested_cpus, 0)
    waste_ratio = waste / fit_cpu if fit_cpu else 0.0
    risk: Literal["low", "medium", "high"]
    if waste_ratio <= _WASTE_RATIO_LOW_RISK:
        risk = "low"
    elif waste_ratio <= _WASTE_RATIO_MEDIUM_RISK:
        risk = "medium"
    else:
        risk = "high"

    result = ResourceFitResponse(
        recommendation={
            "recommended_node_cpus": fit_cpu,
            "requested_cpus": payload.requested_cpus,
            "waste_cpus": waste,
            "waste_ratio": round(waste_ratio, 4),
        },
        fragmentation_risk=risk,
        notes=[
            "Node fit baseline is capacity-first and deterministic.",
            "Replace with learned fit model once topology features are implemented.",
        ],
    )
    _set_telemetry_headers(
        response,
        trace_id=_request_trace_id(request),
        model_version="resource-fit-baseline-v1",
        fallback_used=False,
    )
    return result


@app.get("/v1/recommendations/{run_id}", response_model=None)
def get_recommendation(run_id: str, request: Request, response: Response) -> dict[str, Any] | JSONResponse:
    """Retrieve stored recommendation results for a given run ID."""
    # Validate run_id to prevent path traversal
    forbidden = {"\\", "/", "..", "*", "?", "[", "]"}
    for ch in forbidden:
        if ch in run_id:
            return JSONResponse(
                status_code=400,
                content=_error_content(
                    code="INVALID_RUN_ID",
                    message=f"run_id contains forbidden character: {ch!r}",
                    trace_id=_request_trace_id(request),
                    status=400,
                ),
            )

    artifacts_dir = Path(os.getenv("HPCOPT_ARTIFACTS_DIR", "outputs"))
    recommendation_path = artifacts_dir / "recommendations" / f"{run_id}.json"

    if not recommendation_path.exists():
        return JSONResponse(
            status_code=404,
            content=_error_content(
                code="NOT_FOUND",
                message=f"No recommendation found for run_id: {run_id}",
                trace_id=_request_trace_id(request),
                status=404,
            ),
        )

    try:
        payload = json.loads(recommendation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load recommendation %s: %s", run_id, exc)
        return JSONResponse(
            status_code=500,
            content=_error_content(
                code="INTERNAL_ERROR",
                message="Failed to load recommendation",
                trace_id=_request_trace_id(request),
                status=500,
            ),
        )

    _set_telemetry_headers(response, trace_id=_request_trace_id(request))
    return cast(dict[str, Any], payload)
