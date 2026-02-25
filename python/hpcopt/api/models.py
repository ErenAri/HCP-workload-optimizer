"""Pydantic request/response models for the HPC Workload Optimizer API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class LogLevelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(..., description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
