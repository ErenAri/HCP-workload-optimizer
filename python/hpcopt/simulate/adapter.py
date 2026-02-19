from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


EventType = Literal["job_complete", "job_submit", "dispatch"]

_EVENT_ORDER: dict[EventType, int] = {
    "job_complete": 0,
    "job_submit": 1,
    "dispatch": 2,
}


@dataclass(frozen=True)
class AdapterEvent:
    ts: int
    event_type: EventType
    job_id: int | None = None


@dataclass(frozen=True)
class AdapterQueuedJob:
    job_id: int
    submit_ts: int
    requested_cpus: int
    runtime_estimate_sec: int
    runtime_p90_sec: int | None = None
    runtime_guard_sec: int | None = None
    estimate_source: str | None = None


@dataclass(frozen=True)
class AdapterRunningJob:
    job_id: int
    end_ts: int
    allocated_cpus: int


@dataclass(frozen=True)
class SchedulerStateSnapshot:
    clock_ts: int
    capacity_cpus: int
    free_cpus: int
    queued_jobs: tuple[AdapterQueuedJob, ...]
    running_jobs: tuple[AdapterRunningJob, ...]


@dataclass(frozen=True)
class DispatchDecision:
    job_id: int
    requested_cpus: int
    runtime_estimate_sec: int
    estimated_completion_ts: int
    reason: str


@dataclass(frozen=True)
class SchedulerDecision:
    policy_id: str
    reservation_ts: int | None
    decisions: tuple[DispatchDecision, ...]


def _require_fields(payload: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def parse_state_snapshot(payload: dict[str, Any]) -> SchedulerStateSnapshot:
    _require_fields(
        payload,
        {"clock_ts", "capacity_cpus", "free_cpus", "queued_jobs", "running_jobs"},
        "state_snapshot",
    )
    if payload["capacity_cpus"] <= 0:
        raise ValueError("capacity_cpus must be > 0")
    if payload["free_cpus"] < 0:
        raise ValueError("free_cpus must be >= 0")
    if payload["free_cpus"] > payload["capacity_cpus"]:
        raise ValueError("free_cpus cannot exceed capacity_cpus")

    queued_jobs: list[AdapterQueuedJob] = []
    for i, job in enumerate(payload["queued_jobs"]):
        _require_fields(
            job,
            {"job_id", "submit_ts", "requested_cpus", "runtime_estimate_sec"},
            f"queued_jobs[{i}]",
        )
        queued_jobs.append(
            AdapterQueuedJob(
                job_id=int(job["job_id"]),
                submit_ts=int(job["submit_ts"]),
                requested_cpus=int(job["requested_cpus"]),
                runtime_estimate_sec=max(0, int(job["runtime_estimate_sec"])),
                runtime_p90_sec=(
                    max(0, int(job["runtime_p90_sec"]))
                    if job.get("runtime_p90_sec") is not None
                    else None
                ),
                runtime_guard_sec=(
                    max(0, int(job["runtime_guard_sec"]))
                    if job.get("runtime_guard_sec") is not None
                    else None
                ),
                estimate_source=(
                    str(job.get("estimate_source"))
                    if job.get("estimate_source") is not None
                    else None
                ),
            )
        )

    running_jobs: list[AdapterRunningJob] = []
    for i, job in enumerate(payload["running_jobs"]):
        _require_fields(job, {"job_id", "end_ts", "allocated_cpus"}, f"running_jobs[{i}]")
        running_jobs.append(
            AdapterRunningJob(
                job_id=int(job["job_id"]),
                end_ts=int(job["end_ts"]),
                allocated_cpus=int(job["allocated_cpus"]),
            )
        )

    queued_jobs_sorted = tuple(sorted(queued_jobs, key=lambda job: (job.submit_ts, job.job_id)))
    running_jobs_sorted = tuple(sorted(running_jobs, key=lambda job: (job.end_ts, job.job_id)))
    return SchedulerStateSnapshot(
        clock_ts=int(payload["clock_ts"]),
        capacity_cpus=int(payload["capacity_cpus"]),
        free_cpus=int(payload["free_cpus"]),
        queued_jobs=queued_jobs_sorted,
        running_jobs=running_jobs_sorted,
    )


def order_events(events: list[AdapterEvent]) -> list[AdapterEvent]:
    return sorted(
        events,
        key=lambda event: (
            int(event.ts),
            _EVENT_ORDER[event.event_type],
            int(event.job_id) if event.job_id is not None else -1,
        ),
    )


def snapshot_state_hash(snapshot: SchedulerStateSnapshot) -> str:
    payload = {
        "clock_ts": snapshot.clock_ts,
        "capacity_cpus": snapshot.capacity_cpus,
        "free_cpus": snapshot.free_cpus,
        "queued_jobs": [asdict(job) for job in snapshot.queued_jobs],
        "running_jobs": [asdict(job) for job in snapshot.running_jobs],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reservation_ts_for_hol(snapshot: SchedulerStateSnapshot, hol: AdapterQueuedJob) -> int:
    if hol.requested_cpus > snapshot.capacity_cpus:
        return 10**18
    if hol.requested_cpus <= snapshot.free_cpus:
        return snapshot.clock_ts

    free_cpus = snapshot.free_cpus
    for running in snapshot.running_jobs:
        free_cpus += running.allocated_cpus
        if free_cpus >= hol.requested_cpus:
            return max(snapshot.clock_ts, running.end_ts)
    return 10**18


def choose_fifo_strict(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
    decisions: list[DispatchDecision] = []
    available = snapshot.free_cpus
    for job in snapshot.queued_jobs:
        if job.requested_cpus <= 0:
            continue
        if job.requested_cpus <= available:
            decisions.append(
                DispatchDecision(
                    job_id=job.job_id,
                    requested_cpus=job.requested_cpus,
                    runtime_estimate_sec=job.runtime_estimate_sec,
                    estimated_completion_ts=snapshot.clock_ts + job.runtime_estimate_sec,
                    reason="fifo_dispatch",
                )
            )
            available -= job.requested_cpus
        else:
            # Strict FIFO blocks when first non-feasible job is reached.
            break
    return SchedulerDecision(
        policy_id="FIFO_STRICT",
        reservation_ts=None,
        decisions=tuple(decisions),
    )


def choose_easy_backfill(snapshot: SchedulerStateSnapshot) -> SchedulerDecision:
    if not snapshot.queued_jobs:
        return SchedulerDecision(
            policy_id="EASY_BACKFILL_BASELINE",
            reservation_ts=None,
            decisions=tuple(),
        )

    queue = list(snapshot.queued_jobs)
    queue.sort(key=lambda job: (job.submit_ts, job.job_id))
    hol = queue[0]
    reservation_ts = _reservation_ts_for_hol(snapshot, hol)

    available = snapshot.free_cpus
    decisions: list[DispatchDecision] = []

    if hol.requested_cpus > 0 and hol.requested_cpus <= available:
        decisions.append(
            DispatchDecision(
                job_id=hol.job_id,
                requested_cpus=hol.requested_cpus,
                runtime_estimate_sec=hol.runtime_estimate_sec,
                estimated_completion_ts=snapshot.clock_ts + hol.runtime_estimate_sec,
                reason="easy_head_dispatch",
            )
        )
        available -= hol.requested_cpus
        tail = queue[1:]
        for job in tail:
            if job.requested_cpus <= 0:
                continue
            if job.requested_cpus <= available:
                decisions.append(
                    DispatchDecision(
                        job_id=job.job_id,
                        requested_cpus=job.requested_cpus,
                        runtime_estimate_sec=job.runtime_estimate_sec,
                        estimated_completion_ts=snapshot.clock_ts + job.runtime_estimate_sec,
                        reason="easy_follow_dispatch",
                    )
                )
                available -= job.requested_cpus
        return SchedulerDecision(
            policy_id="EASY_BACKFILL_BASELINE",
            reservation_ts=reservation_ts,
            decisions=tuple(decisions),
        )

    for job in queue[1:]:
        if job.requested_cpus <= 0:
            continue
        if job.requested_cpus > available:
            continue
        completion_ts = snapshot.clock_ts + job.runtime_estimate_sec
        if completion_ts <= reservation_ts:
            decisions.append(
                DispatchDecision(
                    job_id=job.job_id,
                    requested_cpus=job.requested_cpus,
                    runtime_estimate_sec=job.runtime_estimate_sec,
                    estimated_completion_ts=completion_ts,
                    reason="easy_backfill",
                )
            )
            available -= job.requested_cpus

    return SchedulerDecision(
        policy_id="EASY_BACKFILL_BASELINE",
        reservation_ts=reservation_ts,
        decisions=tuple(decisions),
    )


def choose_ml_backfill_p50(
    snapshot: SchedulerStateSnapshot,
    strict_uncertainty_mode: bool = False,
) -> SchedulerDecision:
    if not snapshot.queued_jobs:
        return SchedulerDecision(
            policy_id="ML_BACKFILL_P50",
            reservation_ts=None,
            decisions=tuple(),
        )

    queue = list(snapshot.queued_jobs)
    queue.sort(key=lambda job: (job.submit_ts, job.job_id))
    hol = queue[0]
    reservation_ts = _reservation_ts_for_hol(snapshot, hol)

    available = snapshot.free_cpus
    decisions: list[DispatchDecision] = []

    if hol.requested_cpus > 0 and hol.requested_cpus <= available:
        decisions.append(
            DispatchDecision(
                job_id=hol.job_id,
                requested_cpus=hol.requested_cpus,
                runtime_estimate_sec=hol.runtime_estimate_sec,
                estimated_completion_ts=snapshot.clock_ts + hol.runtime_estimate_sec,
                reason="ml_head_dispatch",
            )
        )
        available -= hol.requested_cpus
        for job in queue[1:]:
            if job.requested_cpus <= 0:
                continue
            if job.requested_cpus <= available:
                decisions.append(
                    DispatchDecision(
                        job_id=job.job_id,
                        requested_cpus=job.requested_cpus,
                        runtime_estimate_sec=job.runtime_estimate_sec,
                        estimated_completion_ts=snapshot.clock_ts + job.runtime_estimate_sec,
                        reason=f"ml_follow_dispatch:{job.estimate_source or 'unknown'}",
                    )
                )
                available -= job.requested_cpus
        return SchedulerDecision(
            policy_id="ML_BACKFILL_P50",
            reservation_ts=reservation_ts,
            decisions=tuple(decisions),
        )

    for job in queue[1:]:
        if job.requested_cpus <= 0:
            continue
        if job.requested_cpus > available:
            continue

        runtime_for_gate = (
            job.runtime_p90_sec
            if strict_uncertainty_mode
            else (job.runtime_guard_sec or job.runtime_estimate_sec)
        )
        completion_ts = snapshot.clock_ts + runtime_for_gate
        if completion_ts <= reservation_ts:
            decisions.append(
                DispatchDecision(
                    job_id=job.job_id,
                    requested_cpus=job.requested_cpus,
                    runtime_estimate_sec=runtime_for_gate,
                    estimated_completion_ts=completion_ts,
                    reason=f"ml_backfill:{job.estimate_source or 'unknown'}",
                )
            )
            available -= job.requested_cpus

    return SchedulerDecision(
        policy_id="ML_BACKFILL_P50",
        reservation_ts=reservation_ts,
        decisions=tuple(decisions),
    )
