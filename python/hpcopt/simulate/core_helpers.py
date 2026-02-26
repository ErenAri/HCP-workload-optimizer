"""Shared helpers for the simulation core."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from hpcopt.simulate.adapter import (
    SchedulerStateSnapshot,
    choose_easy_backfill,
    choose_fifo_strict,
    choose_ml_backfill_p50,
)

logger = logging.getLogger(__name__)


def coerce_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "submit_ts", "runtime_actual_sec"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace dataframe missing required columns: {sorted(missing)}")

    df = trace_df.copy()
    if "requested_cpus" not in df.columns:
        if "allocated_cpus" in df.columns:
            df["requested_cpus"] = df["allocated_cpus"]
        else:
            raise ValueError("Trace requires requested_cpus (or allocated_cpus fallback)")

    if "runtime_requested_sec" not in df.columns:
        df["runtime_requested_sec"] = None
    for col in ["user_id", "group_id", "queue_id", "partition_id", "requested_mem"]:
        if col not in df.columns:
            df[col] = None

    for col in ["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"])
    if df.empty:
        raise ValueError("Trace dataframe has no valid rows after coercion.")

    df["job_id"] = df["job_id"].astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)
    df["runtime_actual_sec"] = df["runtime_actual_sec"].clip(lower=0).astype(int)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)

    df["runtime_requested_sec"] = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
    df["runtime_requested_sec"] = df["runtime_requested_sec"].where(df["runtime_requested_sec"] > 0)
    df["runtime_estimate_sec"] = df["runtime_requested_sec"].fillna(df["runtime_actual_sec"]).astype(int)

    df = df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)
    return df


def choose_decisions(
    snapshot: SchedulerStateSnapshot,
    policy_id: str,
    strict_uncertainty_mode: bool = False,
) -> Any:
    if policy_id == "FIFO_STRICT":
        return choose_fifo_strict(snapshot)
    if policy_id == "EASY_BACKFILL_BASELINE":
        return choose_easy_backfill(snapshot)
    if policy_id == "ML_BACKFILL_P50":
        return choose_ml_backfill_p50(
            snapshot=snapshot,
            strict_uncertainty_mode=strict_uncertainty_mode,
        )
    raise ValueError(f"Unsupported policy_id '{policy_id}'")


def build_prediction_features(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "submit_ts": int(job["submit_ts"]),
        "requested_cpus": int(job["requested_cpus"]),
        "runtime_requested_sec": (
            int(job["runtime_requested_sec"]) if pd.notna(job.get("runtime_requested_sec")) else None
        ),
        "requested_mem": (int(job["requested_mem"]) if pd.notna(job.get("requested_mem")) else None),
        "queue_id": int(job["queue_id"]) if pd.notna(job.get("queue_id")) else None,
        "partition_id": (int(job["partition_id"]) if pd.notna(job.get("partition_id")) else None),
        "user_id": int(job["user_id"]) if pd.notna(job.get("user_id")) else None,
        "group_id": int(job["group_id"]) if pd.notna(job.get("group_id")) else None,
    }


def attach_runtime_estimates(
    jobs_df: pd.DataFrame,
    policy_id: str,
    runtime_predictor: Any | None,
    runtime_guard_k: float,
) -> pd.DataFrame:
    df = jobs_df.copy()
    df["runtime_p50_sec"] = None
    df["runtime_p90_sec"] = None
    df["runtime_guard_sec"] = None
    df["estimate_source"] = None

    for idx, row in df.iterrows():
        requested_runtime = row.get("runtime_requested_sec")
        actual_runtime = int(row["runtime_actual_sec"])

        if policy_id == "ML_BACKFILL_P50":
            predicted = None
            if runtime_predictor is not None:
                try:
                    predicted = runtime_predictor.predict_one(build_prediction_features(row.to_dict()))
                except (ValueError, KeyError, TypeError) as exc:
                    logger.warning("Runtime prediction failed for job %s: %s", row.get("job_id"), exc)
                    predicted = None

            if predicted is not None:
                p50 = int(max(1, round(predicted["p50"])))
                p90 = int(max(p50, round(predicted["p90"])))
                guard = int(round(p50 + runtime_guard_k * (p90 - p50)))
                source = "prediction"
            elif pd.notna(requested_runtime) and float(requested_runtime) > 0:
                p50 = int(requested_runtime)
                p90 = int(requested_runtime)
                guard = int(requested_runtime)
                source = "requested_fallback"
            else:
                p50 = int(actual_runtime)
                p90 = int(actual_runtime)
                guard = int(actual_runtime)
                source = "actual_fallback"

            df.at[idx, "runtime_p50_sec"] = p50
            df.at[idx, "runtime_p90_sec"] = p90
            df.at[idx, "runtime_guard_sec"] = max(1, guard)
            df.at[idx, "runtime_estimate_sec"] = p50
            df.at[idx, "estimate_source"] = source
        else:
            if pd.notna(requested_runtime) and float(requested_runtime) > 0:
                estimate = int(requested_runtime)
                source = "requested_fallback"
            else:
                estimate = int(actual_runtime)
                source = "actual_fallback"
            df.at[idx, "runtime_p50_sec"] = estimate
            df.at[idx, "runtime_p90_sec"] = estimate
            df.at[idx, "runtime_guard_sec"] = estimate
            df.at[idx, "runtime_estimate_sec"] = estimate
            df.at[idx, "estimate_source"] = source
    return df


def check_invariants(
    clock_ts: int,
    capacity_cpus: int,
    free_cpus: int,
    queued_jobs: list[dict[str, Any]],
    running_jobs: list[dict[str, Any]],
) -> list[str]:
    failed: list[str] = []
    if free_cpus < 0:
        failed.append("free_cpus_negative")
    if free_cpus > capacity_cpus:
        failed.append("free_cpus_exceeds_capacity")

    running_cpu = sum(int(job["requested_cpus"]) for job in running_jobs)
    if running_cpu + free_cpus != capacity_cpus:
        failed.append("cpu_conservation_broken")

    queued_ids = {int(job["job_id"]) for job in queued_jobs}
    running_ids = {int(job["job_id"]) for job in running_jobs}
    overlap = queued_ids & running_ids
    if overlap:
        failed.append("job_exists_in_queue_and_running")

    for running in running_jobs:
        if int(running["start_ts"]) < int(running["submit_ts"]):
            failed.append(f"job_start_before_submit:{running['job_id']}")
        if int(running["end_ts"]) < int(running["start_ts"]):
            failed.append(f"job_end_before_start:{running['job_id']}")
        if int(running["requested_cpus"]) <= 0:
            failed.append(f"job_nonpositive_cpu:{running['job_id']}")

    for queued in queued_jobs:
        if int(queued["submit_ts"]) > clock_ts:
            failed.append(f"queued_job_submit_in_future:{queued['job_id']}")

    return failed


def invariant_report(
    run_id: str,
    strict_mode: bool,
    step_count: int,
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "strict_mode": bool(strict_mode),
        "step_count": int(step_count),
        "violations": violations,
    }
