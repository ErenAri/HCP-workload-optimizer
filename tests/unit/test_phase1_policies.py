"""End-to-end tests for the Phase 1 policies: CBF, SJF, LJF, FAIRSHARE."""

from __future__ import annotations

import pandas as pd
import pytest

from hpcopt.simulate.adapter import (
    AdapterQueuedJob,
    AdapterRunningJob,
    SchedulerStateSnapshot,
    choose_conservative_backfill,
    choose_fairshare_backfill,
    choose_ljf_backfill,
    choose_sjf_backfill,
)
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace
from hpcopt.simulate.fairshare import compute_fairshare_priorities


# ── adapter-level tests ────────────────────────────────────────────


def _qj(job_id, submit, cpus, runtime, *, priority=None):
    return AdapterQueuedJob(
        job_id=job_id,
        submit_ts=submit,
        requested_cpus=cpus,
        runtime_estimate_sec=runtime,
        priority_score=priority,
    )


def test_sjf_orders_by_shortest_estimate():
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=8,
        free_cpus=8,
        queued_jobs=(_qj(1, 0, 4, 1000), _qj(2, 0, 4, 100), _qj(3, 0, 4, 500)),
        running_jobs=(),
    )
    decision = choose_sjf_backfill(snap)
    # Shortest first (job 2: 100s), then job 3 (500s) fits in remaining cpus.
    dispatched = [d.job_id for d in decision.decisions]
    assert dispatched[0] == 2
    assert 3 in dispatched


def test_ljf_orders_by_longest_estimate():
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=8,
        free_cpus=8,
        queued_jobs=(_qj(1, 0, 4, 100), _qj(2, 0, 4, 1000), _qj(3, 0, 4, 500)),
        running_jobs=(),
    )
    decision = choose_ljf_backfill(snap)
    assert decision.decisions[0].job_id == 2  # longest first


def test_fairshare_dispatches_higher_priority_first():
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=4,
        free_cpus=4,
        queued_jobs=(
            _qj(1, 0, 4, 100, priority=-1000.0),  # high recent usage
            _qj(2, 0, 4, 100, priority=-10.0),    # low usage → wins
        ),
        running_jobs=(),
    )
    decision = choose_fairshare_backfill(snap)
    assert decision.decisions[0].job_id == 2


def test_cbf_dispatches_when_capacity_allows():
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=8,
        free_cpus=8,
        queued_jobs=(_qj(1, 0, 4, 100), _qj(2, 0, 4, 100)),
        running_jobs=(),
    )
    decision = choose_conservative_backfill(snap)
    ids = sorted(d.job_id for d in decision.decisions)
    assert ids == [1, 2]


def test_cbf_protects_all_queued_reservations_not_just_head():
    """Conservative differs from EASY here: a queued job behind the head
    that *would* delay an even-later queued job's reservation must NOT
    backfill, even if it would fit in EASY's shadow window.
    """
    # Capacity=8, free_cpus=4 now. A 4-cpu job at end_ts=100 will release
    # making 8 free at t=100.
    # Queue (FIFO order):
    #   J1 needs 8 cpus, runtime 50  → must wait until t=100, reserved [100,150)
    #   J2 needs 4 cpus, runtime 60  → earliest fit: t=0..60 (free=4 now). OK.
    #   J3 needs 4 cpus, runtime 200 → if we let J3 run now [0,200), it would
    #          eat 4 cpus through t=200, blocking J1's reservation at t=100.
    #          Conservative MUST refuse.
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=8,
        free_cpus=4,
        queued_jobs=(_qj(1, 0, 8, 50), _qj(2, 0, 4, 60), _qj(3, 0, 4, 200)),
        running_jobs=(AdapterRunningJob(job_id=99, end_ts=100, allocated_cpus=4),),
    )
    decision = choose_conservative_backfill(snap)
    dispatched = {d.job_id for d in decision.decisions}
    assert 1 not in dispatched  # head can't run yet
    assert 2 in dispatched      # safe to backfill
    assert 3 not in dispatched  # would push J1 later — forbidden under CBF
    assert decision.reservation_ts == 100


def test_cbf_skips_oversize_jobs_without_blocking_others():
    snap = SchedulerStateSnapshot(
        clock_ts=0,
        capacity_cpus=4,
        free_cpus=4,
        queued_jobs=(_qj(1, 0, 999, 100), _qj(2, 0, 4, 100)),
        running_jobs=(),
    )
    decision = choose_conservative_backfill(snap)
    dispatched = {d.job_id for d in decision.decisions}
    assert dispatched == {2}


# ── fairshare priority computation ─────────────────────────────────


def test_fairshare_priority_favors_low_usage_user():
    df = pd.DataFrame(
        [
            {"job_id": 1, "user_id": 100, "submit_ts": 0, "runtime_actual_sec": 1000, "requested_cpus": 4},
            {"job_id": 2, "user_id": 100, "submit_ts": 2000, "runtime_actual_sec": 100, "requested_cpus": 4},
            {"job_id": 3, "user_id": 200, "submit_ts": 2000, "runtime_actual_sec": 100, "requested_cpus": 4},
        ]
    )
    scores = compute_fairshare_priorities(df, half_life_sec=86400)
    # User 100 has 1000s job completed by t=1000, billed before submit_ts=2000.
    # User 200 has no prior usage → score 0. User 100 → score < 0.
    assert scores.iloc[2] > scores.iloc[1]


def test_fairshare_priority_decays_over_time():
    df = pd.DataFrame(
        [
            {"job_id": 1, "user_id": 100, "submit_ts": 0, "runtime_actual_sec": 1000, "requested_cpus": 4},
            {"job_id": 2, "user_id": 100, "submit_ts": 2000, "runtime_actual_sec": 1, "requested_cpus": 1},
            {"job_id": 3, "user_id": 100, "submit_ts": 10**9, "runtime_actual_sec": 1, "requested_cpus": 1},
        ]
    )
    scores = compute_fairshare_priorities(df, half_life_sec=3600)
    # The far-future job has effectively decayed away → score ~ 0.
    assert abs(scores.iloc[2]) < abs(scores.iloc[1])


# ── full simulation tests ──────────────────────────────────────────


@pytest.fixture
def small_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"job_id": 1, "submit_ts": 0,   "runtime_actual_sec": 100, "requested_cpus": 4, "user_id": 1},
            {"job_id": 2, "submit_ts": 10,  "runtime_actual_sec": 50,  "requested_cpus": 4, "user_id": 2},
            {"job_id": 3, "submit_ts": 20,  "runtime_actual_sec": 30,  "requested_cpus": 2, "user_id": 1},
            {"job_id": 4, "submit_ts": 30,  "runtime_actual_sec": 200, "requested_cpus": 6, "user_id": 2},
            {"job_id": 5, "submit_ts": 40,  "runtime_actual_sec": 10,  "requested_cpus": 2, "user_id": 1},
        ]
    )


@pytest.mark.parametrize(
    "policy",
    [
        "CONSERVATIVE_BACKFILL_BASELINE",
        "SJF_BACKFILL",
        "LJF_BACKFILL",
        "FAIRSHARE_BACKFILL",
    ],
)
def test_phase1_policies_run_end_to_end(small_trace, policy):
    assert policy in SUPPORTED_POLICIES
    result = run_simulation_from_trace(
        trace_df=small_trace,
        policy_id=policy,
        capacity_cpus=8,
        run_id=f"test_{policy}",
        strict_invariants=True,
    )
    assert result.policy_id == policy
    assert len(result.jobs_df) == len(small_trace)
    # No invariant violations under strict mode.
    assert result.invariant_report["violations"] == []


def test_cbf_yields_no_violations_against_easy(small_trace):
    """CBF should produce a feasible schedule (no invariant breaks); we don't
    require it to beat EASY on any specific metric — just to be valid."""
    cbf = run_simulation_from_trace(
        trace_df=small_trace,
        policy_id="CONSERVATIVE_BACKFILL_BASELINE",
        capacity_cpus=8,
        run_id="cbf",
        strict_invariants=True,
    )
    easy = run_simulation_from_trace(
        trace_df=small_trace,
        policy_id="EASY_BACKFILL_BASELINE",
        capacity_cpus=8,
        run_id="easy",
        strict_invariants=True,
    )
    assert len(cbf.jobs_df) == len(easy.jobs_df) == len(small_trace)
