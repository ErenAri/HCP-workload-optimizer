"""Prediction API routes (runtime and resource-fit)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from fastapi import Request, Response

from hpcopt.api.errors import request_trace_id, set_telemetry_headers
from hpcopt.api.model_cache import get_runtime_predictor
from hpcopt.api.models import (
    ResourceFitRequest,
    ResourceFitResponse,
    RuntimePredictRequest,
    RuntimePredictResponse,
)
from hpcopt.utils.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

_FALLBACK_BASE_RUNTIME_SEC = 1800
_FALLBACK_QUEUE_DEPTH_CAP = 2000
_FALLBACK_QUEUE_DEPTH_SCALE = 10_000.0
_FALLBACK_P50_FACTOR = 0.72
_FALLBACK_P90_FACTOR = 1.08
_FALLBACK_MIN_RUNTIME_SEC = 60

_WASTE_RATIO_LOW_RISK = 0.15
_WASTE_RATIO_MEDIUM_RISK = 0.35


def register_prediction_routes(
    app: Any,
    prediction_circuit: CircuitBreaker,
    runtime_predictor_loader: Callable[[], tuple[Any, Any]] = get_runtime_predictor,
) -> None:
    """Register prediction routes on a FastAPI app instance."""

    @app.post("/v1/runtime/predict", response_model=RuntimePredictResponse)
    def predict_runtime(
        payload: RuntimePredictRequest,
        request: Request,
        response: Response,
    ) -> RuntimePredictResponse:
        try:
            predictor, model_dir = runtime_predictor_loader()
        except prediction_circuit.CircuitOpenError:
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
                prediction_circuit.record_success()
            except Exception:
                prediction_circuit.record_failure()
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

        try:
            from hpcopt.api.metrics import record_fallback

            record_fallback()
        except ImportError:
            pass

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
            fallback_used=True,
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
