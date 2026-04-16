"""Tests for the Tsafrir/Etsion/Feitelson runtime predictor and policy."""

from __future__ import annotations

import pandas as pd

from hpcopt.models.baseline_tsafrir import (
    compute_tsafrir_estimates,
    predict_one,
)
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace


# ---------------------------------------------------------------------------
# Unit tests for predict_one
# ---------------------------------------------------------------------------


def test_predict_one_cold_start_returns_user_estimate() -> None:
    pred = predict_one(user_id=1, user_estimate_sec=600, history_runtimes_sec=[])
    assert pred.runtime_estimate_sec == 600
    assert pred.completed_history_count == 0
    assert pred.clamped_by_user_estimate is False


def test_predict_one_single_history_returns_that_runtime() -> None:
    pred = predict_one(user_id=1, user_estimate_sec=600, history_runtimes_sec=[100])
    assert pred.runtime_estimate_sec == 100
    assert pred.completed_history_count == 1
    assert pred.clamped_by_user_estimate is False


def test_predict_one_two_history_returns_average() -> None:
    pred = predict_one(user_id=1, user_estimate_sec=600, history_runtimes_sec=[200, 100])
    # avg(200, 100) = 150
    assert pred.runtime_estimate_sec == 150
    assert pred.completed_history_count == 2
    assert pred.clamped_by_user_estimate is False


def test_predict_one_clamps_above_user_estimate() -> None:
    # avg(800, 1000) = 900 > user_estimate 500 -> clamp to 500
    pred = predict_one(user_id=1, user_estimate_sec=500, history_runtimes_sec=[800, 1000])
    assert pred.runtime_estimate_sec == 500
    assert pred.clamped_by_user_estimate is True


def test_predict_one_only_uses_two_most_recent() -> None:
    # The 3rd entry must be ignored.
    pred = predict_one(
        user_id=1,
        user_estimate_sec=10_000,
        history_runtimes_sec=[100, 200, 9999],
    )
    assert pred.runtime_estimate_sec == 150
    assert pred.completed_history_count == 2


def test_predict_one_rejects_nonpositive_user_estimate() -> None:
    try:
        predict_one(user_id=1, user_estimate_sec=0, history_runtimes_sec=[])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Tests for compute_tsafrir_estimates (chronological scan)
# ---------------------------------------------------------------------------


def test_compute_tsafrir_estimates_respects_completion_order() -> None:
    # User 1 submits 3 jobs; the 3rd job's prediction should be the average
    # of jobs 1 and 2's actual runtimes (both completed before job 3 submits).
    df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0,
                "runtime_actual_sec": 100,
                "runtime_estimate_sec": 1000,
                "user_id": 1,
            },
            {
                "job_id": 2,
                "submit_ts": 200,  # after job 1 completes at ts=100
                "runtime_actual_sec": 200,
                "runtime_estimate_sec": 1000,
                "user_id": 1,
            },
            {
                "job_id": 3,
                "submit_ts": 500,  # after job 2 completes at ts=400
                "runtime_actual_sec": 50,
                "runtime_estimate_sec": 1000,
                "user_id": 1,
            },
        ]
    )
    out = compute_tsafrir_estimates(df)
    by_id = {int(r["job_id"]): r for _, r in out.iterrows()}
    # Job 1 has no history -> user_estimate (1000)
    assert int(by_id[1]["tsafrir_runtime_sec"]) == 1000
    assert int(by_id[1]["tsafrir_history_count"]) == 0
    # Job 2 has 1 prior completion (job 1, runtime 100) -> 100
    assert int(by_id[2]["tsafrir_runtime_sec"]) == 100
    assert int(by_id[2]["tsafrir_history_count"]) == 1
    # Job 3 has 2 prior completions (jobs 2 and 1) -> avg(200, 100) = 150
    assert int(by_id[3]["tsafrir_runtime_sec"]) == 150
    assert int(by_id[3]["tsafrir_history_count"]) == 2


def test_compute_tsafrir_estimates_ignores_uncompleted_at_submit() -> None:
    # Job 2 submits BEFORE job 1 completes -> job 2 is cold start.
    df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0,
                "runtime_actual_sec": 1000,
                "runtime_estimate_sec": 5000,
                "user_id": 7,
            },
            {
                "job_id": 2,
                "submit_ts": 10,  # well before job 1 completes at ts=1000
                "runtime_actual_sec": 50,
                "runtime_estimate_sec": 5000,
                "user_id": 7,
            },
        ]
    )
    out = compute_tsafrir_estimates(df)
    by_id = {int(r["job_id"]): r for _, r in out.iterrows()}
    assert int(by_id[2]["tsafrir_history_count"]) == 0
    assert int(by_id[2]["tsafrir_runtime_sec"]) == 5000


def test_compute_tsafrir_estimates_partitions_by_user() -> None:
    df = pd.DataFrame(
        [
            {"job_id": 1, "submit_ts": 0, "runtime_actual_sec": 100, "runtime_estimate_sec": 1000, "user_id": 1},
            {"job_id": 2, "submit_ts": 0, "runtime_actual_sec": 800, "runtime_estimate_sec": 1000, "user_id": 2},
            {"job_id": 3, "submit_ts": 2000, "runtime_actual_sec": 50, "runtime_estimate_sec": 1000, "user_id": 1},
        ]
    )
    out = compute_tsafrir_estimates(df)
    by_id = {int(r["job_id"]): r for _, r in out.iterrows()}
    # Job 3 only sees user 1's history (job 1 -> runtime 100), NOT user 2's job.
    assert int(by_id[3]["tsafrir_runtime_sec"]) == 100
    assert int(by_id[3]["tsafrir_history_count"]) == 1


# ---------------------------------------------------------------------------
# End-to-end simulation tests
# ---------------------------------------------------------------------------


def test_easy_backfill_tsafrir_is_supported_policy() -> None:
    assert "EASY_BACKFILL_TSAFRIR" in SUPPORTED_POLICIES


def test_easy_backfill_tsafrir_runs_end_to_end() -> None:
    trace = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0,
                "runtime_actual_sec": 100,
                "requested_cpus": 4,
                "runtime_requested_sec": 1000,
                "user_id": 1,
            },
            {
                "job_id": 2,
                "submit_ts": 200,
                "runtime_actual_sec": 50,
                "requested_cpus": 4,
                "runtime_requested_sec": 1000,
                "user_id": 1,
            },
            {
                "job_id": 3,
                "submit_ts": 400,
                "runtime_actual_sec": 75,
                "requested_cpus": 4,
                "runtime_requested_sec": 1000,
                "user_id": 1,
            },
        ]
    )
    result = run_simulation_from_trace(
        trace_df=trace,
        policy_id="EASY_BACKFILL_TSAFRIR",
        capacity_cpus=8,
        run_id="tsafrir_e2e",
        strict_invariants=True,
    )
    # All jobs scheduled.
    assert len(result.jobs_df) == 3
    # Tsafrir history counts surface in fallback accounting.
    accounting = result.fallback_accounting
    assert accounting["tsafrir_cold_start_count"] >= 1
    assert accounting["tsafrir_history_count"] + accounting["tsafrir_cold_start_count"] == 3


def test_easy_backfill_tsafrir_estimate_source_labels() -> None:
    trace = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0,
                "runtime_actual_sec": 100,
                "requested_cpus": 2,
                "runtime_requested_sec": 1000,
                "user_id": 9,
            },
            {
                "job_id": 2,
                "submit_ts": 200,
                "runtime_actual_sec": 100,
                "requested_cpus": 2,
                "runtime_requested_sec": 1000,
                "user_id": 9,
            },
            {
                "job_id": 3,
                "submit_ts": 400,
                "runtime_actual_sec": 100,
                "requested_cpus": 2,
                "runtime_requested_sec": 1000,
                "user_id": 9,
            },
        ]
    )
    result = run_simulation_from_trace(
        trace_df=trace,
        policy_id="EASY_BACKFILL_TSAFRIR",
        capacity_cpus=8,
        run_id="tsafrir_labels",
        strict_invariants=True,
    )
    sources = {int(r["job_id"]): str(r["estimate_source"]) for _, r in result.jobs_df.iterrows()}
    assert sources[1] == "tsafrir_cold_start"
    assert sources[2] == "tsafrir_history_1"
    assert sources[3] == "tsafrir_history_2"
