"""Health, readiness, status, and metrics API routes."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from hpcopt import __version__
from hpcopt.api.errors import request_trace_id
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir

logger = logging.getLogger(__name__)

_HEALTH_MIN_DISK_FREE_GB = float(os.getenv("HPCOPT_HEALTH_MIN_DISK_FREE_GB", "0.1"))


def register_health_routes(app: Any) -> None:
    """Register health/readiness/status routes on a FastAPI app instance."""

    @app.get("/health")
    def health() -> dict[str, object]:
        checks: list[str] = []
        status = "ok"

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

        try:
            disk = shutil.disk_usage(".")
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb < _HEALTH_MIN_DISK_FREE_GB:
                checks.append("disk_low")
                status = "degraded"
        except OSError:
            disk_free_gb = None

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
        if getattr(app.state, "shutdown_requested", False):
            return JSONResponse(status_code=503, content={"status": "shutting_down"})

        try:
            disk = shutil.disk_usage(".")
            disk_free_gb = disk.free / (1024**3)
            if disk_free_gb < _HEALTH_MIN_DISK_FREE_GB:
                return JSONResponse(status_code=503, content={"status": "degraded", "reason": "disk_low"})
        except OSError:
            pass

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
        try:
            from hpcopt.api.metrics import get_metrics_response

            return PlainTextResponse(content=get_metrics_response(), media_type="text/plain")
        except ImportError:
            return PlainTextResponse(
                content="# prometheus_client not installed\n",
                media_type="text/plain",
            )
