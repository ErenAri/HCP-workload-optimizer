"""Property-based tests for the objective contract module.

Tests invariants:
- Jain fairness index is always in [0, 1]
- Starvation rate is always in [0, 1]
- Empty dataframe returns safe defaults
- Constraint evaluation is deterministic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


def _make_jobs_df(
    n_jobs: int,
    seed: int,
    n_users: int = 5,
) -> pd.DataFrame:
    """Build a synthetic jobs DataFrame for objective contract tests."""
    rng = np.random.default_rng(seed)
    submit_ts = np.cumsum(rng.integers(1, 60, size=n_jobs))
    wait = rng.integers(0, 600, size=n_jobs)
    runtime = rng.integers(60, 7200, size=n_jobs)
    start_ts = submit_ts + wait
    end_ts = start_ts + runtime
    return pd.DataFrame({
        "job_id": range(1, n_jobs + 1),
        "submit_ts": submit_ts,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "requested_cpus": rng.choice([1, 2, 4, 8, 16], size=n_jobs),
        "user_id": rng.integers(1, n_users + 1, size=n_jobs),
    })


@given(
    n_jobs=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=10000),
    n_users=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_fairness_metrics_bounds(n_jobs: int, seed: int, n_users: int) -> None:
    """Jain index is always in [0,1] and starvation rate is always in [0,1]."""
    from hpcopt.simulate.objective import compute_fairness_starvation_metrics

    jobs_df = _make_jobs_df(n_jobs, seed, n_users)
    result = compute_fairness_starvation_metrics(jobs_df)

    assert 0.0 <= result["jain"] <= 1.0, f"Jain index out of bounds: {result['jain']}"
    assert 0.0 <= result["starved_rate"] <= 1.0, f"Starvation rate out of bounds: {result['starved_rate']}"
    assert result["total_jobs"] == n_jobs
    assert result["active_users"] >= 1
    assert result["fairness_dev"] >= 0.0


def test_empty_df_returns_safe_defaults() -> None:
    """Empty dataframe must return safe zero/default values."""
    from hpcopt.simulate.objective import compute_fairness_starvation_metrics

    result = compute_fairness_starvation_metrics(pd.DataFrame())
    assert result["starved_rate"] == 0.0
    assert result["jain"] == 1.0
    assert result["total_jobs"] == 0.0
    assert result["fairness_dev"] == 0.0


@given(
    n_jobs=st.integers(min_value=10, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_constraint_evaluation_deterministic(n_jobs: int, seed: int) -> None:
    """Same inputs must produce identical constraint evaluation results."""
    from hpcopt.simulate.objective import (
        compute_objective_contract_metrics,
        evaluate_constraint_contract,
    )

    jobs_df = _make_jobs_df(n_jobs, seed)
    metrics1 = compute_objective_contract_metrics(jobs_df, capacity_cpus=64)
    metrics2 = compute_objective_contract_metrics(jobs_df, capacity_cpus=64)
    assert metrics1 == metrics2

    baseline = compute_objective_contract_metrics(_make_jobs_df(50, 0), capacity_cpus=64)
    result1 = evaluate_constraint_contract(metrics1, baseline)
    result2 = evaluate_constraint_contract(metrics2, baseline)
    assert result1 == result2


@given(
    n_jobs=st.integers(min_value=10, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_utilization_between_zero_and_one(n_jobs: int, seed: int) -> None:
    """CPU utilization must always be in [0, 1]."""
    from hpcopt.simulate.metrics import compute_job_metrics

    jobs_df = _make_jobs_df(n_jobs, seed)
    metrics = compute_job_metrics(jobs_df, capacity_cpus=64)
    assert 0.0 <= metrics["utilization_cpu"] <= 1.0, \
        f"Utilization out of bounds: {metrics['utilization_cpu']}"
    assert metrics["job_count"] == n_jobs
    assert metrics["p95_bsld"] >= 0.0


@given(
    n_jobs=st.integers(min_value=10, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
    w1=st.floats(min_value=0.0, max_value=10.0),
    w2=st.floats(min_value=0.0, max_value=10.0),
    w3=st.floats(min_value=0.0, max_value=10.0),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_weighted_score_deterministic(n_jobs: int, seed: int, w1: float, w2: float, w3: float) -> None:
    """Weighted score must be deterministic for same inputs."""
    from hpcopt.simulate.objective import (
        compute_objective_contract_metrics,
        compute_weighted_analysis_score,
    )

    candidate = compute_objective_contract_metrics(_make_jobs_df(n_jobs, seed), capacity_cpus=64)
    baseline = compute_objective_contract_metrics(_make_jobs_df(50, 0), capacity_cpus=64)

    s1 = compute_weighted_analysis_score(candidate, baseline, w1=w1, w2=w2, w3=w3)
    s2 = compute_weighted_analysis_score(candidate, baseline, w1=w1, w2=w2, w3=w3)
    assert s1 == s2
