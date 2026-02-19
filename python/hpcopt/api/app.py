from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hpcopt import __version__
from hpcopt.models.runtime_quantile import (
    RuntimeQuantilePredictor,
    resolve_runtime_model_dir,
)


app = FastAPI(
    title="HPC Workload Optimizer API",
    version=__version__,
    description="Systems-first API scaffold for runtime/resource-fit baseline predictions.",
)


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
    runtime_model_dir: str | None = Field(
        default=None, description="Optional explicit model directory override"
    )


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


_RUNTIME_PREDICTOR_CACHE: dict[str, RuntimeQuantilePredictor | Path | None] = {
    "model_dir": None,
    "predictor": None,
}


def _get_runtime_predictor(explicit_model_dir: str | None) -> tuple[RuntimeQuantilePredictor | None, Path | None]:
    resolved = resolve_runtime_model_dir(Path(explicit_model_dir) if explicit_model_dir else None)
    if resolved is None:
        return None, None

    cached_dir = _RUNTIME_PREDICTOR_CACHE["model_dir"]
    if isinstance(cached_dir, Path) and cached_dir == resolved:
        cached_predictor = _RUNTIME_PREDICTOR_CACHE["predictor"]
        if isinstance(cached_predictor, RuntimeQuantilePredictor):
            return cached_predictor, resolved

    predictor = RuntimeQuantilePredictor(resolved)
    _RUNTIME_PREDICTOR_CACHE["model_dir"] = resolved
    _RUNTIME_PREDICTOR_CACHE["predictor"] = predictor
    return predictor, resolved


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hpcopt-api", "version": __version__}


@app.post("/v1/runtime/predict", response_model=RuntimePredictResponse)
def predict_runtime(payload: RuntimePredictRequest) -> RuntimePredictResponse:
    predictor, model_dir = _get_runtime_predictor(payload.runtime_model_dir)

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
        return RuntimePredictResponse(
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

    fallback_used = True
    base = payload.requested_runtime_sec or 1800
    queue_depth = payload.queue_depth_jobs or 0
    queue_factor = 1.0 + min(queue_depth, 2000) / 10000.0
    runtime_p50 = max(60, int(base * 0.72 * queue_factor))
    runtime_p90 = max(runtime_p50, int(base * 1.08 * queue_factor))
    runtime_guard = int(runtime_p50 + payload.runtime_guard_k * (runtime_p90 - runtime_p50))
    return RuntimePredictResponse(
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


@app.post("/v1/resource-fit/predict", response_model=ResourceFitResponse)
def predict_resource_fit(payload: ResourceFitRequest) -> ResourceFitResponse:
    candidate_node_cpus = sorted(payload.candidate_node_cpus)

    fit_cpu = candidate_node_cpus[-1]
    for capacity in candidate_node_cpus:
        if capacity >= payload.requested_cpus:
            fit_cpu = capacity
            break

    waste = max(fit_cpu - payload.requested_cpus, 0)
    waste_ratio = waste / fit_cpu if fit_cpu else 0.0
    if waste_ratio <= 0.15:
        risk = "low"
    elif waste_ratio <= 0.35:
        risk = "medium"
    else:
        risk = "high"

    return ResourceFitResponse(
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
