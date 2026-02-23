"""Property-based tests for the simulation core using Hypothesis.

Tests deterministic invariants that must hold for all valid inputs:
- free CPUs never goes negative and never exceeds capacity
- all jobs eventually complete (no deadlocks)
- job start_ts >= submit_ts always
- deterministic replay (same seed -> same outcome)
"""
from __future__ import annotations

import pandas as pd
import pytest

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


def _make_trace_df(jobs: list[dict]) -> pd.DataFrame:
    """Build a canonical trace DataFrame from a list of job dicts."""
    return pd.DataFrame(jobs)


def _job_strategy(min_jobs: int = 5, max_jobs: int = 50):
    """Strategy that generates a valid trace DataFrame for simulation."""
    return st.integers(min_value=min_jobs, max_value=max_jobs).flatmap(
        lambda n: st.fixed_dictionaries({
            "n_jobs": st.just(n),
            "seed": st.integers(min_value=0, max_value=10000),
        })
    )


@given(data=_job_strategy(5, 40))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_simulation_invariants_fifo(data: dict) -> None:
    """FIFO_STRICT simulation: all invariants hold on synthetic data."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.simulate.core import run_simulation_from_trace

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        stress = generate_stress_scenario(
            scenario="heavy_tail",
            out_dir=Path(tmp),
            n_jobs=data["n_jobs"],
            seed=data["seed"],
            params={"alpha": 1.25},
        )
        trace_df = pd.read_parquet(stress.dataset_path)
        assume(len(trace_df) >= 5)

        result = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="FIFO_STRICT",
            capacity_cpus=64,
            run_id=f"prop_fifo_{data['seed']}",
            strict_invariants=True,
        )

        jobs = result.jobs_df
        assert len(jobs) > 0, "Simulation must produce job results"

        # Invariant: every job starts at or after its submission
        assert (jobs["start_ts"] >= jobs["submit_ts"]).all(), "start_ts must be >= submit_ts"

        # Invariant: every job ends at or after its start
        assert (jobs["end_ts"] >= jobs["start_ts"]).all(), "end_ts must be >= start_ts"

        # Invariant: all submitted jobs are completed
        assert len(jobs) == len(trace_df), "All jobs must complete"

        # Invariant: no invariant violations reported
        violations = result.invariant_report.get("violations", [])
        assert len(violations) == 0, f"Invariant violations found: {violations}"


@given(data=_job_strategy(5, 40))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_simulation_invariants_easy_backfill(data: dict) -> None:
    """EASY_BACKFILL_BASELINE simulation: all invariants hold on synthetic data."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.simulate.core import run_simulation_from_trace

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        stress = generate_stress_scenario(
            scenario="heavy_tail",
            out_dir=Path(tmp),
            n_jobs=data["n_jobs"],
            seed=data["seed"],
            params={"alpha": 1.25},
        )
        trace_df = pd.read_parquet(stress.dataset_path)
        assume(len(trace_df) >= 5)

        result = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="EASY_BACKFILL_BASELINE",
            capacity_cpus=64,
            run_id=f"prop_easy_{data['seed']}",
            strict_invariants=True,
        )

        jobs = result.jobs_df
        assert len(jobs) == len(trace_df), "All jobs must complete"
        assert (jobs["start_ts"] >= jobs["submit_ts"]).all(), "start_ts must be >= submit_ts"
        assert (jobs["end_ts"] >= jobs["start_ts"]).all(), "end_ts must be >= start_ts"

        violations = result.invariant_report.get("violations", [])
        assert len(violations) == 0, f"Invariant violations found: {violations}"


@given(
    seed_a=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_simulation_deterministic_replay(seed_a: int) -> None:
    """Same trace + same seed must produce byte-identical results."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.simulate.core import run_simulation_from_trace

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        stress = generate_stress_scenario(
            scenario="heavy_tail",
            out_dir=Path(tmp),
            n_jobs=20,
            seed=seed_a,
            params={"alpha": 1.25},
        )
        trace_df = pd.read_parquet(stress.dataset_path)

        r1 = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="FIFO_STRICT",
            capacity_cpus=64,
            run_id="replay_a",
            strict_invariants=True,
        )
        r2 = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="FIFO_STRICT",
            capacity_cpus=64,
            run_id="replay_b",
            strict_invariants=True,
        )

        pd.testing.assert_frame_equal(r1.jobs_df, r2.jobs_df, check_names=False)
        assert r1.metrics == r2.metrics, "Metrics must be identical for same input"


@given(data=_job_strategy(5, 40))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_temporal_ordering_invariant(data: dict) -> None:
    """For all completed jobs: completion_ts >= start_ts >= submit_ts."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.simulate.core import run_simulation_from_trace

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        stress = generate_stress_scenario(
            scenario="heavy_tail",
            out_dir=Path(tmp),
            n_jobs=data["n_jobs"],
            seed=data["seed"],
            params={"alpha": 1.25},
        )
        trace_df = pd.read_parquet(stress.dataset_path)
        assume(len(trace_df) >= 5)

        for policy in ("FIFO_STRICT", "EASY_BACKFILL_BASELINE"):
            result = run_simulation_from_trace(
                trace_df=trace_df,
                policy_id=policy,
                capacity_cpus=64,
                run_id=f"temporal_{policy}_{data['seed']}",
                strict_invariants=True,
            )
            jobs = result.jobs_df
            assert len(jobs) > 0

            # Full temporal chain: submit_ts <= start_ts <= end_ts
            assert (jobs["start_ts"] >= jobs["submit_ts"]).all(), \
                f"{policy}: start_ts < submit_ts for some jobs"
            assert (jobs["end_ts"] >= jobs["start_ts"]).all(), \
                f"{policy}: end_ts < start_ts for some jobs"
            assert (jobs["end_ts"] >= jobs["submit_ts"]).all(), \
                f"{policy}: end_ts < submit_ts for some jobs (transitivity)"


@given(data=_job_strategy(5, 40))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_metric_monotonicity(data: dict) -> None:
    """Makespan must be >= max(completion_ts) - min(submit_ts)."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.simulate.core import run_simulation_from_trace

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        stress = generate_stress_scenario(
            scenario="heavy_tail",
            out_dir=Path(tmp),
            n_jobs=data["n_jobs"],
            seed=data["seed"],
            params={"alpha": 1.25},
        )
        trace_df = pd.read_parquet(stress.dataset_path)
        assume(len(trace_df) >= 5)

        for policy in ("FIFO_STRICT", "EASY_BACKFILL_BASELINE"):
            result = run_simulation_from_trace(
                trace_df=trace_df,
                policy_id=policy,
                capacity_cpus=64,
                run_id=f"mono_{policy}_{data['seed']}",
                strict_invariants=True,
            )
            jobs = result.jobs_df
            assert len(jobs) > 0

            actual_span = jobs["end_ts"].max() - jobs["submit_ts"].min()
            reported_makespan = result.metrics.get("makespan_sec", actual_span)

            # Reported makespan must be at least as large as the observed span
            assert reported_makespan >= actual_span - 1, \
                f"{policy}: makespan {reported_makespan} < observed span {actual_span}"
