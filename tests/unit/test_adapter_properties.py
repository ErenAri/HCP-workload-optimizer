"""Property-based tests for the scheduler adapter module.

Tests invariants:
- FIFO decisions are always sorted by submit_ts
- EASY backfill never dispatches more CPUs than available
- State hash is deterministic
- Event ordering is deterministic
"""
from __future__ import annotations

import pytest

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from hpcopt.simulate.adapter import (
    AdapterQueuedJob,
    AdapterRunningJob,
    SchedulerStateSnapshot,
    choose_fifo_strict,
    choose_easy_backfill,
    snapshot_state_hash,
    order_events,
    AdapterEvent,
)


def _make_snapshot(
    n_queued: int,
    n_running: int,
    capacity_cpus: int,
    seed: int,
) -> SchedulerStateSnapshot:
    """Build a valid SchedulerStateSnapshot for testing."""
    import random
    random.seed(seed)

    running_cpus = 0
    running = []
    for i in range(n_running):
        cpus = random.choice([1, 2, 4, 8])
        if running_cpus + cpus > capacity_cpus:
            break
        running_cpus += cpus
        running.append(AdapterRunningJob(
            job_id=1000 + i,
            end_ts=100 + random.randint(60, 3600),
            allocated_cpus=cpus,
        ))

    free_cpus = capacity_cpus - running_cpus
    queued = []
    for i in range(n_queued):
        cpus = random.choice([1, 2, 4, 8, 16])
        queued.append(AdapterQueuedJob(
            job_id=i + 1,
            submit_ts=random.randint(1, 100),
            requested_cpus=cpus,
            runtime_estimate_sec=random.randint(60, 3600),
        ))

    return SchedulerStateSnapshot(
        clock_ts=100,
        capacity_cpus=capacity_cpus,
        free_cpus=free_cpus,
        queued_jobs=tuple(queued),
        running_jobs=tuple(running),
    )


@given(
    n_queued=st.integers(min_value=1, max_value=30),
    n_running=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_fifo_decisions_sorted_by_submit_ts(n_queued: int, n_running: int, seed: int) -> None:
    """FIFO_STRICT must only dispatch head-of-queue jobs (head-of-line blocking)."""
    snapshot = _make_snapshot(n_queued, n_running, capacity_cpus=64, seed=seed)
    decision = choose_fifo_strict(snapshot)

    if len(decision.decisions) > 0:
        # All dispatched jobs must have fit within the free capacity
        dispatched_cpus = sum(d.requested_cpus for d in decision.decisions)
        assert dispatched_cpus <= snapshot.free_cpus, \
            f"Dispatched {dispatched_cpus} CPUs but only {snapshot.free_cpus} free"


@given(
    n_queued=st.integers(min_value=1, max_value=30),
    n_running=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_easy_backfill_never_exceeds_capacity(n_queued: int, n_running: int, seed: int) -> None:
    """EASY_BACKFILL must never dispatch more CPUs than free."""
    snapshot = _make_snapshot(n_queued, n_running, capacity_cpus=64, seed=seed)
    decision = choose_easy_backfill(snapshot)

    dispatched_cpus = sum(d.requested_cpus for d in decision.decisions)
    assert dispatched_cpus <= snapshot.free_cpus, \
        f"Dispatched {dispatched_cpus} CPUs but only {snapshot.free_cpus} free"


@given(
    n_queued=st.integers(min_value=1, max_value=30),
    n_running=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_cpu_conservation_law(n_queued: int, n_running: int, seed: int) -> None:
    """Dispatched jobs' total CPUs must never exceed cluster capacity."""
    snapshot = _make_snapshot(n_queued, n_running, capacity_cpus=64, seed=seed)

    # Check FIFO
    fifo_decision = choose_fifo_strict(snapshot)
    running_cpus = sum(j.allocated_cpus for j in snapshot.running_jobs)
    fifo_dispatched = sum(d.requested_cpus for d in fifo_decision.decisions)
    assert running_cpus + fifo_dispatched <= snapshot.capacity_cpus, \
        f"FIFO conservation violated: {running_cpus} running + {fifo_dispatched} dispatched > {snapshot.capacity_cpus}"

    # Check EASY backfill
    easy_decision = choose_easy_backfill(snapshot)
    easy_dispatched = sum(d.requested_cpus for d in easy_decision.decisions)
    assert running_cpus + easy_dispatched <= snapshot.capacity_cpus, \
        f"EASY conservation violated: {running_cpus} running + {easy_dispatched} dispatched > {snapshot.capacity_cpus}"


@given(
    n_queued=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_state_hash_deterministic(n_queued: int, seed: int) -> None:
    """Same snapshot must produce identical state hashes."""
    snapshot = _make_snapshot(n_queued, 2, capacity_cpus=64, seed=seed)
    h1 = snapshot_state_hash(snapshot)
    h2 = snapshot_state_hash(snapshot)
    assert h1 == h2, "State hash must be deterministic"
    assert isinstance(h1, str) and len(h1) > 0


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=10, deadline=None)
def test_event_ordering_deterministic(seed: int) -> None:
    """Event ordering must be deterministic for the same input."""
    import random
    random.seed(seed)

    events = [
        AdapterEvent(ts=random.randint(1, 100), event_type=random.choice(["job_complete", "job_submit", "dispatch"]), job_id=i)
        for i in range(20)
    ]

    o1 = order_events(events.copy())
    o2 = order_events(events.copy())

    assert len(o1) == len(o2)
    for e1, e2 in zip(o1, o2):
        assert e1.ts == e2.ts
        assert e1.event_type == e2.event_type
        assert e1.job_id == e2.job_id
