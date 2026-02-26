import pandas as pd
from hpcopt.simulate.core import run_simulation_from_trace


class StubPredictor:
    def predict_one(self, features: dict) -> dict[str, float]:
        user_id = features.get("user_id")
        if user_id in {4, 5}:
            raise ValueError("forced prediction miss for fallback path")
        if user_id == 3:
            return {"p10": 1.0, "p50": 2.0, "p90": 20.0}
        return {"p10": 5.0, "p50": 8.0, "p90": 12.0}


def test_ml_backfill_fallback_accounting_counts() -> None:
    trace = pd.DataFrame(
        [
            {
                "job_id": 10,
                "submit_ts": 0,
                "runtime_actual_sec": 100,
                "requested_cpus": 6,
                "runtime_requested_sec": 120,
                "user_id": 1,
            },
            {
                "job_id": 20,
                "submit_ts": 1,
                "runtime_actual_sec": 50,
                "requested_cpus": 8,
                "runtime_requested_sec": 80,
                "user_id": 2,
            },
            {
                "job_id": 30,
                "submit_ts": 2,
                "runtime_actual_sec": 10,
                "requested_cpus": 2,
                "runtime_requested_sec": 30,
                "user_id": 3,
            },
            {
                "job_id": 40,
                "submit_ts": 3,
                "runtime_actual_sec": 10,
                "requested_cpus": 2,
                "runtime_requested_sec": 20,
                "user_id": 4,
            },
            {
                "job_id": 50,
                "submit_ts": 4,
                "runtime_actual_sec": 10,
                "requested_cpus": 2,
                "runtime_requested_sec": None,
                "user_id": 5,
            },
        ]
    )
    result = run_simulation_from_trace(
        trace_df=trace,
        policy_id="ML_BACKFILL_P50",
        capacity_cpus=8,
        run_id="ml_fallback_counts",
        strict_invariants=True,
        runtime_predictor=StubPredictor(),
        runtime_guard_k=0.5,
        strict_uncertainty_mode=False,
    )
    accounting = result.fallback_accounting
    assert accounting["total_scheduled_jobs"] == 5
    assert accounting["prediction_used_count"] == 3
    assert accounting["requested_fallback_count"] == 1
    assert accounting["actual_fallback_count"] == 1


def test_ml_backfill_strict_uncertainty_changes_eligibility() -> None:
    trace = pd.DataFrame(
        [
            {
                "job_id": 10,
                "submit_ts": 0,
                "runtime_actual_sec": 10,
                "requested_cpus": 6,
                "runtime_requested_sec": 20,
                "user_id": 1,
            },
            {
                "job_id": 20,
                "submit_ts": 1,
                "runtime_actual_sec": 50,
                "requested_cpus": 8,
                "runtime_requested_sec": 80,
                "user_id": 2,
            },
            {
                "job_id": 30,
                "submit_ts": 2,
                "runtime_actual_sec": 5,
                "requested_cpus": 2,
                "runtime_requested_sec": 15,
                "user_id": 3,
            },
        ]
    )
    predictor = StubPredictor()

    non_strict = run_simulation_from_trace(
        trace_df=trace,
        policy_id="ML_BACKFILL_P50",
        capacity_cpus=8,
        run_id="ml_non_strict",
        strict_invariants=True,
        runtime_predictor=predictor,
        runtime_guard_k=0.3,
        strict_uncertainty_mode=False,
    )
    strict = run_simulation_from_trace(
        trace_df=trace,
        policy_id="ML_BACKFILL_P50",
        capacity_cpus=8,
        run_id="ml_strict",
        strict_invariants=True,
        runtime_predictor=predictor,
        runtime_guard_k=0.3,
        strict_uncertainty_mode=True,
    )

    non_strict_start = int(non_strict.jobs_df.loc[non_strict.jobs_df["job_id"] == 30, "start_ts"].iloc[0])
    strict_start = int(strict.jobs_df.loc[strict.jobs_df["job_id"] == 30, "start_ts"].iloc[0])
    assert non_strict_start < strict_start
