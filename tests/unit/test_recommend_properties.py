"""Property-based tests for the recommendation engine.

Tests invariants:
- Pareto frontier is non-dominated
- Score is deterministic for same inputs
- Fidelity gate blocking always prevents acceptance
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


def _make_sim_report(
    p95_bsld: float,
    utilization_cpu: float,
    fairness_dev: float,
    jain: float,
    starved_rate: float,
    policy_id: str = "TEST_POLICY",
) -> dict[str, Any]:
    """Build a minimal simulation report dict."""
    return {
        "policy_id": policy_id,
        "metrics": {
            "p95_bsld": p95_bsld,
            "utilization_cpu": utilization_cpu,
            "fairness_dev": fairness_dev,
            "jain": jain,
            "starved_rate": starved_rate,
            "mean_wait_sec": 100.0,
            "p95_wait_sec": 500.0,
            "job_count": 100.0,
            "throughput": 0.05,
            "makespan_sec": 10000.0,
            "starved_jobs": 0.0,
            "total_jobs": 100.0,
            "active_users": 5.0,
        },
        "objective_metrics": {
            "p95_bsld": p95_bsld,
            "utilization_cpu": utilization_cpu,
            "fairness_dev": fairness_dev,
            "jain": jain,
            "starved_rate": starved_rate,
            "mean_wait_sec": 100.0,
            "p95_wait_sec": 500.0,
            "job_count": 100.0,
            "throughput": 0.05,
            "makespan_sec": 10000.0,
            "starved_jobs": 0.0,
            "total_jobs": 100.0,
            "active_users": 5.0,
        },
    }


def _write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_fidelity_gate_blocks_acceptance() -> None:
    """When fidelity gate fails, recommendation must be blocked."""
    from hpcopt.recommend.engine import generate_recommendation_report

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        baseline = _make_sim_report(10.0, 0.5, 0.1, 0.9, 0.01, "EASY_BACKFILL_BASELINE")
        candidate = _make_sim_report(5.0, 0.55, 0.1, 0.9, 0.01, "ML_BACKFILL_P50")
        fidelity = {"status": "fail", "checks": {"aggregate": "fail"}}

        baseline_path = _write_report(tmp_path / "baseline.json", baseline)
        candidate_path = _write_report(tmp_path / "candidate.json", candidate)
        fidelity_path = _write_report(tmp_path / "fidelity.json", fidelity)

        result = generate_recommendation_report(
            baseline_report_path=baseline_path,
            candidate_report_paths=[candidate_path],
            out_path=tmp_path / "rec.json",
            fidelity_report_path=fidelity_path,
        )

        assert result.payload["status"] == "blocked"
        assert result.payload["fidelity_status"] == "fail"


@given(
    bsld_baseline=st.floats(min_value=1.0, max_value=50.0),
    bsld_candidate=st.floats(min_value=1.0, max_value=50.0),
    util_baseline=st.floats(min_value=0.1, max_value=0.9),
    util_candidate=st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_score_deterministic(
    bsld_baseline: float,
    bsld_candidate: float,
    util_baseline: float,
    util_candidate: float,
) -> None:
    """Score computation must be deterministic for same inputs."""
    from hpcopt.simulate.objective import compute_weighted_analysis_score

    baseline = {
        "p95_bsld": bsld_baseline,
        "utilization_cpu": util_baseline,
        "fairness_dev": 0.1,
        "jain": 0.9,
    }
    candidate = {
        "p95_bsld": bsld_candidate,
        "utilization_cpu": util_candidate,
        "fairness_dev": 0.1,
        "jain": 0.9,
    }

    s1 = compute_weighted_analysis_score(candidate, baseline)
    s2 = compute_weighted_analysis_score(candidate, baseline)
    assert s1["score"] == s2["score"], "Score must be deterministic"


def test_pareto_frontier_non_dominated() -> None:
    """Pareto frontier points must not be dominated by each other."""
    from hpcopt.recommend.engine import generate_pareto_recommendation

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        baseline = _make_sim_report(10.0, 0.5, 0.1, 0.9, 0.01, "EASY_BACKFILL_BASELINE")
        c1 = _make_sim_report(5.0, 0.55, 0.12, 0.88, 0.01, "CANDIDATE_A")
        c2 = _make_sim_report(8.0, 0.60, 0.08, 0.92, 0.01, "CANDIDATE_B")
        c3 = _make_sim_report(3.0, 0.45, 0.15, 0.85, 0.02, "CANDIDATE_C")

        baseline_path = _write_report(tmp_path / "baseline.json", baseline)
        c1_path = _write_report(tmp_path / "c1.json", c1)
        c2_path = _write_report(tmp_path / "c2.json", c2)
        c3_path = _write_report(tmp_path / "c3.json", c3)

        result = generate_pareto_recommendation(
            baseline_report_path=baseline_path,
            candidate_report_paths=[c1_path, c2_path, c3_path],
            out_path=tmp_path / "pareto.json",
        )

        report = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
        frontier = report.get("pareto_frontier", [])

        # Each point on the frontier must not be dominated by any other frontier point
        for i, point_i in enumerate(frontier):
            for j, point_j in enumerate(frontier):
                if i == j:
                    continue
                scores_i = point_i.get("objective_scores", {})
                scores_j = point_j.get("objective_scores", {})
                # If j dominates i on all objectives, that's invalid
                if scores_i and scores_j:
                    all_worse = all(
                        scores_j.get(k, 0) >= scores_i.get(k, 0)
                        for k in scores_i
                    )
                    any_strict = any(
                        scores_j.get(k, 0) > scores_i.get(k, 0)
                        for k in scores_i
                    )
                    assert not (all_worse and any_strict), \
                        f"Point {i} is dominated by point {j} on the frontier"
