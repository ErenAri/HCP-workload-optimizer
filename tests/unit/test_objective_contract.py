import pandas as pd
from hpcopt.simulate.objective import (
    compute_objective_contract_metrics,
    compute_weighted_analysis_score,
    evaluate_constraint_contract,
)


def test_objective_metrics_include_fairness_starvation() -> None:
    jobs = pd.DataFrame(
        [
            {"job_id": 1, "submit_ts": 0, "start_ts": 0, "end_ts": 10, "requested_cpus": 4, "user_id": 1},
            {"job_id": 2, "submit_ts": 0, "start_ts": 100, "end_ts": 120, "requested_cpus": 4, "user_id": 2},
        ]
    )
    metrics = compute_objective_contract_metrics(
        jobs_df=jobs,
        capacity_cpus=8,
        starvation_wait_cap_sec=50,
    )
    assert "p95_bsld" in metrics
    assert metrics["starved_rate"] > 0.0
    assert metrics["fairness_dev"] >= 0.0
    assert 0.0 <= metrics["jain"] <= 1.0


def test_constraint_and_weighted_score_contract() -> None:
    baseline = {
        "p95_bsld": 2.0,
        "utilization_cpu": 0.5,
        "fairness_dev": 0.10,
        "jain": 0.95,
        "starved_rate": 0.01,
    }
    candidate = {
        "p95_bsld": 1.5,
        "utilization_cpu": 0.6,
        "fairness_dev": 0.11,
        "jain": 0.94,
        "starved_rate": 0.01,
    }
    constraints = evaluate_constraint_contract(candidate=candidate, baseline=baseline)
    assert constraints["constraints_passed"] is True

    score = compute_weighted_analysis_score(candidate=candidate, baseline=baseline)
    assert score["delta_p95_bsld"] > 0
    assert score["score"] > 0
