"""Route handlers for the HPC Workload Optimizer API."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from hpcopt import __version__
from hpcopt.api.errors import error_content, request_trace_id, set_telemetry_headers
from hpcopt.api.model_cache import get_runtime_predictor
from hpcopt.api.models import (
    LogLevelRequest,
    ResourceFitRequest,
    ResourceFitResponse,
    RuntimePredictRequest,
    RuntimePredictResponse,
)
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir
from hpcopt.utils.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

# ---------- Circuit Breaker ----------

_prediction_circuit = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)

# ---------- Fallback heuristic constants ----------

_FALLBACK_BASE_RUNTIME_SEC = 1800
_FALLBACK_QUEUE_DEPTH_CAP = 2000
_FALLBACK_QUEUE_DEPTH_SCALE = 10_000.0
_FALLBACK_P50_FACTOR = 0.72
_FALLBACK_P90_FACTOR = 1.08
_FALLBACK_MIN_RUNTIME_SEC = 60

# ---------- Resource-fit thresholds ----------

_WASTE_RATIO_LOW_RISK = 0.15
_WASTE_RATIO_MEDIUM_RISK = 0.35

# ---------- Health constants ----------

_HEALTH_MIN_DISK_FREE_GB = float(os.getenv("HPCOPT_HEALTH_MIN_DISK_FREE_GB", "0.1"))


def register_routes(app: Any) -> None:
    """Register all API route handlers on the FastAPI app instance."""

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

        # Configurable model-required readiness (for production environments)
        readiness_require_model = os.getenv("HPCOPT_READINESS_REQUIRE_MODEL", "").lower() in ("1", "true", "yes")
        if readiness_require_model:
            model_dir = resolve_runtime_model_dir()
            if model_dir is None or not model_dir.exists():
                return JSONResponse(status_code=503, content={"status": "degraded", "reason": "model_not_loaded"})

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
            "trace_id": request_trace_id(request),
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
            logger.warning("Prediction circuit open; using fallback. trace_id=%s", request_trace_id(request))
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
            set_telemetry_headers(
                response,
                trace_id=request_trace_id(request),
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
        set_telemetry_headers(
            response,
            trace_id=request_trace_id(request),
            model_version=result.predictor_version,
            fallback_used=result.fallback_used,
        )
        return result

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
        set_telemetry_headers(
            response,
            trace_id=request_trace_id(request),
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
