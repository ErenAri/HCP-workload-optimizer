from pathlib import Path

import json

from hpcopt.recommend.engine import generate_recommendation_report


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_recommend_engine_accepts_candidate_with_guardrails(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    fidelity = tmp_path / "fidelity.json"

    _write(
        baseline,
        {
            "run_id": "base1",
            "policy_id": "EASY_BACKFILL_BASELINE",
            "objective_metrics": {
                "p95_bsld": 2.2,
                "utilization_cpu": 0.50,
                "fairness_dev": 0.10,
                "jain": 0.95,
                "starved_rate": 0.01,
            },
        },
    )
    _write(
        candidate,
        {
            "run_id": "cand1",
            "policy_id": "ML_BACKFILL_P50",
            "objective_metrics": {
                "p95_bsld": 1.9,
                "utilization_cpu": 0.56,
                "fairness_dev": 0.12,
                "jain": 0.93,
                "starved_rate": 0.01,
            },
            "fallback_accounting": {
                "prediction_used_rate": 0.9,
                "requested_fallback_rate": 0.1,
                "actual_fallback_rate": 0.0,
            },
        },
    )
    _write(fidelity, {"status": "pass"})

    result = generate_recommendation_report(
        baseline_report_path=baseline,
        candidate_report_paths=[candidate],
        out_path=tmp_path / "recommendation.json",
        fidelity_report_path=fidelity,
    )
    assert result.payload["status"] == "accepted"
    assert result.payload["selected_recommendation"]["policy_id"] == "ML_BACKFILL_P50"


def test_recommend_engine_blocks_on_fidelity_fail(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    fidelity = tmp_path / "fidelity.json"

    _write(
        baseline,
        {
            "run_id": "base1",
            "policy_id": "EASY_BACKFILL_BASELINE",
            "objective_metrics": {
                "p95_bsld": 2.0,
                "utilization_cpu": 0.50,
                "fairness_dev": 0.10,
                "jain": 0.95,
                "starved_rate": 0.01,
            },
        },
    )
    _write(
        candidate,
        {
            "run_id": "cand1",
            "policy_id": "ML_BACKFILL_P50",
            "objective_metrics": {
                "p95_bsld": 1.0,
                "utilization_cpu": 0.70,
                "fairness_dev": 0.10,
                "jain": 0.95,
                "starved_rate": 0.01,
            },
        },
    )
    _write(fidelity, {"status": "fail"})

    result = generate_recommendation_report(
        baseline_report_path=baseline,
        candidate_report_paths=[candidate],
        out_path=tmp_path / "recommendation.json",
        fidelity_report_path=fidelity,
    )
    assert result.payload["status"] == "blocked"
    assert result.payload["selected_recommendation"] is None
