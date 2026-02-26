from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hpcopt.simulate.adapter import (
    AdapterQueuedJob,
    AdapterRunningJob,
    SchedulerStateSnapshot,
    choose_easy_backfill,
    choose_fifo_strict,
    choose_ml_backfill_p50,
    snapshot_state_hash,
)
from hpcopt.simulate.metrics import compute_job_metrics
from hpcopt.simulate.objective import compute_objective_contract_metrics

logger = logging.getLogger(__name__)

SUPPORTED_POLICIES = {"FIFO_STRICT", "EASY_BACKFILL_BASELINE", "ML_BACKFILL_P50"}


@dataclass
class SimulationResult:
    policy_id: str
    jobs_df: pd.DataFrame
    queue_series_df: pd.DataFrame
    metrics: dict[str, float]
    objective_metrics: dict[str, float]
    invariant_report: dict[str, Any]
    fallback_accounting: dict[str, Any]


def _coerce_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
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


def _choose_decisions(
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


def _build_prediction_features(job: dict[str, Any]) -> dict[str, Any]:
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


def _attach_runtime_estimates(
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
                    predicted = runtime_predictor.predict_one(_build_prediction_features(row.to_dict()))
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


def _check_invariants(
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


def _invariant_report(
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


def run_simulation_from_trace(
    trace_df: pd.DataFrame,
    policy_id: str,
    capacity_cpus: int,
    run_id: str,
    strict_invariants: bool = False,
    runtime_predictor: Any | None = None,
    runtime_guard_k: float = 0.5,
    strict_uncertainty_mode: bool = False,
    starvation_wait_cap_sec: int = 172800,
) -> SimulationResult:
    if policy_id not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy: {policy_id}")
    if capacity_cpus <= 0:
        raise ValueError("capacity_cpus must be > 0")

    jobs_df = _coerce_trace_df(trace_df)
    jobs_df = _attach_runtime_estimates(
        jobs_df=jobs_df,
        policy_id=policy_id,
        runtime_predictor=runtime_predictor,
        runtime_guard_k=runtime_guard_k,
    )
    jobs = jobs_df.to_dict(orient="records")
    total_jobs = len(jobs)

    submit_idx = 0
    queue: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    queue_series: list[dict[str, int]] = []
    free_cpus = int(capacity_cpus)
    clock_ts = int(jobs[0]["submit_ts"])

    step_index = 0
    violations: list[dict[str, Any]] = []
    fallback_counts = {
        "prediction_used_count": 0,
        "requested_fallback_count": 0,
        "actual_fallback_count": 0,
    }

    while len(completed) < total_jobs:
        next_submit_ts = int(jobs[submit_idx]["submit_ts"]) if submit_idx < total_jobs else 10**18
        next_complete_ts = min(int(job["end_ts"]) for job in running) if running else 10**18
        next_ts = min(next_submit_ts, next_complete_ts)
        if next_ts >= 10**18:
            raise RuntimeError("Simulation deadlock: no next event; check resource sizing/policy.")
        clock_ts = next_ts

        completed_now = sorted(
            [job for job in running if int(job["end_ts"]) == clock_ts],
            key=lambda job: int(job["job_id"]),
        )
        if completed_now:
            completed_ids = {int(job["job_id"]) for job in completed_now}
            running = [job for job in running if int(job["job_id"]) not in completed_ids]
            for job in completed_now:
                free_cpus += int(job["requested_cpus"])
                completed.append(job)

        while submit_idx < total_jobs and int(jobs[submit_idx]["submit_ts"]) == clock_ts:
            queue.append(jobs[submit_idx])
            submit_idx += 1
        queue.sort(key=lambda job: (int(job["submit_ts"]), int(job["job_id"])))

        snapshot = SchedulerStateSnapshot(
            clock_ts=clock_ts,
            capacity_cpus=capacity_cpus,
            free_cpus=free_cpus,
            queued_jobs=tuple(
                AdapterQueuedJob(
                    job_id=int(job["job_id"]),
                    submit_ts=int(job["submit_ts"]),
                    requested_cpus=int(job["requested_cpus"]),
                    runtime_estimate_sec=int(job["runtime_estimate_sec"]),
                    runtime_p90_sec=int(job["runtime_p90_sec"]),
                    runtime_guard_sec=int(job["runtime_guard_sec"]),
                    estimate_source=str(job["estimate_source"]),
                )
                for job in queue
            ),
            running_jobs=tuple(
                AdapterRunningJob(
                    job_id=int(job["job_id"]),
                    end_ts=int(job["end_ts"]),
                    allocated_cpus=int(job["requested_cpus"]),
                )
                for job in running
            ),
        )
        decision = _choose_decisions(
            snapshot=snapshot,
            policy_id=policy_id,
            strict_uncertainty_mode=strict_uncertainty_mode,
        )
        for dispatch in decision.decisions:
            idx = next(
                (i for i, job in enumerate(queue) if int(job["job_id"]) == int(dispatch.job_id)),
                None,
            )
            if idx is None:
                continue
            job = queue[idx]
            requested = int(job["requested_cpus"])
            if requested > free_cpus:
                continue

            queue.pop(idx)
            start_ts = clock_ts
            runtime = int(job["runtime_actual_sec"])
            end_ts = start_ts + runtime
            estimate_source = str(job.get("estimate_source", "unknown"))
            running.append(
                {
                    "job_id": int(job["job_id"]),
                    "submit_ts": int(job["submit_ts"]),
                    "start_ts": int(start_ts),
                    "end_ts": int(end_ts),
                    "runtime_actual_sec": int(runtime),
                    "requested_cpus": int(requested),
                    "runtime_estimate_sec": int(job["runtime_estimate_sec"]),
                    "runtime_p90_sec": int(job["runtime_p90_sec"]),
                    "runtime_guard_sec": int(job["runtime_guard_sec"]),
                    "estimate_source": estimate_source,
                    "user_id": job.get("user_id"),
                    "group_id": job.get("group_id"),
                    "queue_id": job.get("queue_id"),
                    "partition_id": job.get("partition_id"),
                }
            )
            free_cpus -= requested
            if estimate_source == "prediction":
                fallback_counts["prediction_used_count"] += 1
            elif estimate_source == "requested_fallback":
                fallback_counts["requested_fallback_count"] += 1
            else:
                fallback_counts["actual_fallback_count"] += 1

        failed = _check_invariants(
            clock_ts=clock_ts,
            capacity_cpus=capacity_cpus,
            free_cpus=free_cpus,
            queued_jobs=queue,
            running_jobs=running,
        )
        if failed:
            violation = {
                "step_index": step_index,
                "event_type": "tick",
                "clock_ts": clock_ts,
                "failed_invariants": failed,
                "severity": "error",
                "state_hash": snapshot_state_hash(snapshot),
            }
            violations.append(violation)
            if strict_invariants:
                raise RuntimeError(f"Strict invariant violation: {failed}")

        queue_series.append(
            {
                "ts": int(clock_ts),
                "queue_len_jobs": int(len(queue)),
                "queue_len_cpu_demand": int(sum(int(job["requested_cpus"]) for job in queue)),
            }
        )
        step_index += 1

        if not running and submit_idx >= total_jobs and queue:
            raise RuntimeError(
                "Simulation cannot progress: queued jobs remain but none can be dispatched. "
                "Likely requested_cpus > capacity_cpus."
            )

    completed_df = pd.DataFrame(completed)
    if completed_df.empty:
        completed_df = pd.DataFrame(
            columns=[
                "job_id",
                "submit_ts",
                "start_ts",
                "end_ts",
                "runtime_actual_sec",
                "requested_cpus",
            ]
        )
    completed_df = completed_df.sort_values(["job_id"]).reset_index(drop=True)
    metrics = compute_job_metrics(completed_df, capacity_cpus=capacity_cpus)
    objective_metrics = compute_objective_contract_metrics(
        jobs_df=completed_df,
        capacity_cpus=capacity_cpus,
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )
    queue_series_df = pd.DataFrame(queue_series).sort_values("ts").drop_duplicates("ts", keep="last")
    queue_series_df = queue_series_df.reset_index(drop=True)

    invariant_report = _invariant_report(
        run_id=run_id,
        strict_mode=strict_invariants,
        step_count=step_index,
        violations=violations,
    )
    total_scheduled = int(len(completed_df))
    denominator = total_scheduled if total_scheduled > 0 else 1
    fallback_accounting = {
        **fallback_counts,
        "prediction_used_rate": float(fallback_counts["prediction_used_count"] / denominator),
        "requested_fallback_rate": float(fallback_counts["requested_fallback_count"] / denominator),
        "actual_fallback_rate": float(fallback_counts["actual_fallback_count"] / denominator),
        "total_scheduled_jobs": total_scheduled,
        "runtime_guard_k": float(runtime_guard_k),
        "strict_uncertainty_mode": bool(strict_uncertainty_mode),
    }
    return SimulationResult(
        policy_id=policy_id,
        jobs_df=completed_df,
        queue_series_df=queue_series_df,
        metrics=metrics,
        objective_metrics=objective_metrics,
        invariant_report=invariant_report,
        fallback_accounting=fallback_accounting,
    )


def build_observed_jobs_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "submit_ts", "start_ts", "end_ts"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace missing required observed columns: {sorted(missing)}")

    df = trace_df.copy()
    if "requested_cpus" not in df.columns:
        if "allocated_cpus" in df.columns:
            df["requested_cpus"] = df["allocated_cpus"]
        else:
            raise ValueError("Trace requires requested_cpus (or allocated_cpus fallback)")

    for col in ["job_id", "submit_ts", "start_ts", "end_ts", "requested_cpus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job_id", "submit_ts", "start_ts", "end_ts", "requested_cpus"])
    df["job_id"] = df["job_id"].astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)
    df["start_ts"] = df["start_ts"].astype(int)
    df["end_ts"] = df["end_ts"].astype(int)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)
    return df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)


def build_observed_queue_series(trace_df: pd.DataFrame) -> pd.DataFrame:
    observed_jobs = build_observed_jobs_df(trace_df)
    events: list[tuple[int, int, int]] = []
    for row in observed_jobs.itertuples(index=False):
        events.append((int(row.submit_ts), 0, int(row.requested_cpus)))
        events.append((int(row.start_ts), 1, int(row.requested_cpus)))

    events.sort(key=lambda item: (item[0], item[1]))
    queue_len = 0
    queue_cpu = 0
    out_rows: list[dict[str, int]] = []

    i = 0
    while i < len(events):
        ts = events[i][0]
        while i < len(events) and events[i][0] == ts and events[i][1] == 0:
            queue_len += 1
            queue_cpu += events[i][2]
            i += 1
        while i < len(events) and events[i][0] == ts and events[i][1] == 1:
            queue_len -= 1
            queue_cpu -= events[i][2]
            i += 1

        queue_len = max(queue_len, 0)
        queue_cpu = max(queue_cpu, 0)
        out_rows.append(
            {
                "ts": int(ts),
                "queue_len_jobs": int(queue_len),
                "queue_len_cpu_demand": int(queue_cpu),
            }
        )
    if not out_rows:
        out_rows.append({"ts": 0, "queue_len_jobs": 0, "queue_len_cpu_demand": 0})
    return pd.DataFrame(out_rows).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
