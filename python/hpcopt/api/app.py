from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import signal
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from hpcopt import __version__
from hpcopt.models.runtime_quantile import (
    RuntimeQuantilePredictor,
    resolve_runtime_model_dir,
)

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

# ---------- Rate limiter (token bucket per API key) ----------

_RATE_LIMIT = int(os.getenv("HPCOPT_RATE_LIMIT", "60"))  # requests per minute
_RATE_MAX_BUCKETS = 10_000  # cap to prevent unbounded memory growth
_RATE_BUCKETS: dict[str, list[float]] = {}
_RATE_LOCK = threading.Lock()
_HEALTH_MIN_DISK_FREE_GB = _float_env("HPCOPT_HEALTH_MIN_DISK_FREE_GB", 0.1)
_EXEMPT_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/v1/system/status",
}


def _check_rate_limit(api_key: str | None) -> tuple[bool, int]:
    """Token-bucket rate limiter. Returns (allowed, retry_after_sec)."""
    if _RATE_LIMIT <= 0:
        return True, 0
    bucket_key = api_key or "__anonymous__"
    now = time.time()
    window_start = now - 60.0

    with _RATE_LOCK:
        # Evict stale buckets when the dict exceeds the cap.
        if len(_RATE_BUCKETS) > _RATE_MAX_BUCKETS:
            stale_keys = [
                k for k, v in _RATE_BUCKETS.items()
                if not v or v[-1] <= window_start
            ]
            for k in stale_keys:
                del _RATE_BUCKETS[k]

        bucket = _RATE_BUCKETS.get(bucket_key, [])
        bucket = [ts for ts in bucket if ts > window_start]

        if len(bucket) >= _RATE_LIMIT:
            oldest = bucket[0]
            retry_after = max(1, int(oldest + 60.0 - now))
            _RATE_BUCKETS[bucket_key] = bucket
            return False, retry_after
        bucket.append(now)
        _RATE_BUCKETS[bucket_key] = bucket
    return True, 0


# ---------- Lifespan (graceful shutdown) ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    logger.info("HPC Workload Optimizer API starting (version=%s)", __version__)
    yield
    logger.info("Shutting down HPC Workload Optimizer API. Flushing metrics.")


app = FastAPI(
    title="HPC Workload Optimizer API",
    version=__version__,
    description="Systems-first API for runtime/resource-fit predictions and HPC advisory.",
    lifespan=lifespan,
)


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
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "trace_id": trace_id}
    if details is not None:
        error["details"] = details
    return {"error": error}


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
            details={"errors": exc.errors()},
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
        ),
    )
    _set_telemetry_headers(response, trace_id)
    return response


@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Combined middleware: correlation ID, auth, rate limiting, logging, metrics."""
    start_time = time.time()
    path = request.url.path

    trace_id = request.headers.get("X-Trace-ID") or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:12]
    request.state.trace_id = trace_id

    # Auth check (exempt health/ready/metrics/system status)
    from hpcopt.utils.secrets import load_api_keys
    api_keys = load_api_keys()
    response: Response

    if api_keys and path not in _EXEMPT_PATHS:
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key not in api_keys:
            logger.warning("Auth failed for path=%s trace_id=%s", path, trace_id)
            error_response = JSONResponse(
                status_code=401,
                content=_error_content(
                    code="UNAUTHORIZED",
                    message="Invalid or missing API key",
                    trace_id=trace_id,
                ),
            )
            _set_telemetry_headers(error_response, trace_id)
            return error_response

    # Rate limiting (exempt health/ready)
    if path not in _EXEMPT_PATHS:
        api_key = request.headers.get("X-API-Key")
        allowed, retry_after = _check_rate_limit(api_key)
        if not allowed:
            error_response = JSONResponse(
                status_code=429,
                content=_error_content(
                    code="RATE_LIMITED",
                    message="Rate limit exceeded",
                    trace_id=trace_id,
                ),
                headers={"Retry-After": str(retry_after)},
            )
            _set_telemetry_headers(error_response, trace_id)
            return error_response

    try:
        response = cast(Response, await call_next(request))
    except Exception as exc:
        logger.exception("Request failed for path=%s trace_id=%s", path, trace_id, exc_info=exc)
        response = JSONResponse(
            status_code=500,
            content=_error_content(
                code="INTERNAL_ERROR",
                message="Internal server error",
                trace_id=trace_id,
            ),
        )
    duration = time.time() - start_time

    # Add trace headers to all responses.
    _set_telemetry_headers(response, trace_id)

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
    user_id: int | None = Field(default=None, description="User id when available")
    requested_runtime_sec: int | None = Field(default=None, ge=1)
    requested_cpus: int = Field(..., ge=1)
    requested_mem: int | None = Field(default=None, ge=1)
    queue_id: int | None = Field(default=None)
    partition_id: int | None = Field(default=None)
    group_id: int | None = Field(default=None)
    queue_depth_jobs: int | None = Field(default=None, ge=0)
    runtime_guard_k: float = Field(default=0.5, ge=0.0, le=2.0)


class RuntimePredictResponse(BaseModel):
    predictor_version: str
    runtime_p50_sec: int
    runtime_p90_sec: int
    runtime_guard_sec: int
    fallback_used: bool
    notes: list[str]


class ResourceFitRequest(BaseModel):
    requested_cpus: int = Field(..., ge=1)
    candidate_node_cpus: list[int] = Field(..., min_length=1)
    queue_depth_jobs: int | None = Field(default=None, ge=0)


class ResourceFitResponse(BaseModel):
    recommendation: dict[str, int | float]
    fragmentation_risk: Literal["low", "medium", "high"]
    notes: list[str]


# ---------- Model cache (thread-safe) ----------

_RUNTIME_PREDICTOR_CACHE: dict[str, RuntimeQuantilePredictor | Path | None] = {
    "model_dir": None,
    "predictor": None,
}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_runtime_predictor() -> tuple[RuntimeQuantilePredictor | None, Path | None]:
    """Resolve model directory server-side only (env var or convention)."""
    resolved = resolve_runtime_model_dir()
    if resolved is None:
        return None, None

    with _MODEL_CACHE_LOCK:
        cached_dir = _RUNTIME_PREDICTOR_CACHE["model_dir"]
        if isinstance(cached_dir, Path) and cached_dir == resolved:
            cached_predictor = _RUNTIME_PREDICTOR_CACHE["predictor"]
            if isinstance(cached_predictor, RuntimeQuantilePredictor):
                return cached_predictor, resolved

        predictor = RuntimeQuantilePredictor(resolved)
        _RUNTIME_PREDICTOR_CACHE["model_dir"] = resolved
        _RUNTIME_PREDICTOR_CACHE["predictor"] = predictor
        return predictor, resolved


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


@app.get("/ready")
def ready() -> dict[str, str]:
    """Kubernetes readiness probe."""
    model_dir = resolve_runtime_model_dir()
    if model_dir is not None and model_dir.exists():
        return {"status": "ok"}
    return {"status": "ok"}  # Accept even without model (fallback available)


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
    predictor, model_dir = _get_runtime_predictor()

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
        quantiles = predictor.predict_one(features)
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
    base = payload.requested_runtime_sec or 1800
    queue_depth = payload.queue_depth_jobs or 0
    queue_factor = 1.0 + min(queue_depth, 2000) / 10000.0
    runtime_p50 = max(60, int(base * 0.72 * queue_factor))
    runtime_p90 = max(runtime_p50, int(base * 1.08 * queue_factor))
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
    if waste_ratio <= 0.15:
        risk = "low"
    elif waste_ratio <= 0.35:
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
