from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpcopt.simulate.objective import (
    compute_weighted_analysis_score,
    evaluate_constraint_contract,
)
from hpcopt.utils.io import write_json


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object json: {path}")
    return payload


@dataclass(frozen=True)
class RecommendationResult:
    report_path: Path
    payload: dict[str, Any]


def _extract_objective(report: dict[str, Any]) -> dict[str, float]:
    obj = report.get("objective_metrics")
    if not isinstance(obj, dict):
        raise ValueError("simulation report missing objective_metrics")
    required = {"p95_bsld", "utilization_cpu", "fairness_dev", "jain", "starved_rate"}
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"objective_metrics missing fields: {missing}")
    return {key: float(obj[key]) for key in obj.keys()}


def _fidelity_gate_ok(fidelity_report: dict[str, Any] | None) -> tuple[bool, str | None]:
    if fidelity_report is None:
        return True, None
    status = str(fidelity_report.get("status", "")).lower()
    if status == "pass":
        return True, None
    return False, "fidelity_failed"


def generate_recommendation_report(
    baseline_report_path: Path,
    candidate_report_paths: list[Path],
    out_path: Path,
    fidelity_report_path: Path | None = None,
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 2.0,
) -> RecommendationResult:
    baseline = _load_json(baseline_report_path)
    fidelity = _load_json(fidelity_report_path) if fidelity_report_path else None
    fidelity_ok, fidelity_reason = _fidelity_gate_ok(fidelity)
    baseline_objective = _extract_objective(baseline)
    baseline_policy = str(baseline.get("policy_id", "baseline"))

    candidates: list[dict[str, Any]] = []
    for path in candidate_report_paths:
        cand = _load_json(path)
        cand_obj = _extract_objective(cand)
        score = compute_weighted_analysis_score(
            candidate=cand_obj,
            baseline=baseline_objective,
            w1=w1,
            w2=w2,
            w3=w3,
        )
        constraints = evaluate_constraint_contract(
            candidate=cand_obj,
            baseline=baseline_objective,
        )
        primary_improved = score["delta_p95_bsld"] > 0.0
        accepted = fidelity_ok and constraints["constraints_passed"] and primary_improved
        rejection_reasons: list[str] = []
        if not fidelity_ok and fidelity_reason:
            rejection_reasons.append(fidelity_reason)
        if not constraints["constraints_passed"]:
            rejection_reasons.extend(constraints["violations"])
        if not primary_improved:
            rejection_reasons.append("primary_kpi_not_improved")

        candidates.append(
            {
                "candidate_report_path": str(path),
                "policy_id": cand.get("policy_id"),
                "run_id": cand.get("run_id"),
                "objective_metrics": cand_obj,
                "fallback_accounting": cand.get("fallback_accounting"),
                "score": score,
                "constraints": constraints,
                "primary_improved": primary_improved,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            }
        )

    candidates_sorted = sorted(
        candidates,
        key=lambda item: (item["accepted"], item["score"]["score"]),
        reverse=True,
    )
    winner = candidates_sorted[0] if candidates_sorted else None
    accepted_winner = winner if winner and winner["accepted"] else None

    no_improvement_narrative = None
    if accepted_winner is None and winner is not None:
        no_improvement_narrative = {
            "summary": "No candidate passed guardrails and primary objective improvement criteria.",
            "likely_causes": winner["rejection_reasons"],
        }

    payload = {
        "status": "accepted" if accepted_winner else "blocked",
        "baseline": {
            "policy_id": baseline_policy,
            "run_id": baseline.get("run_id"),
            "objective_metrics": baseline_objective,
            "report_path": str(baseline_report_path),
        },
        "fidelity_status": fidelity.get("status") if fidelity else "not_provided",
        "candidates": candidates_sorted,
        "selected_recommendation": accepted_winner,
        "failure_modes": [
            {
                "policy_id": item["policy_id"],
                "run_id": item["run_id"],
                "rejection_reasons": item["rejection_reasons"],
            }
            for item in candidates_sorted
            if not item["accepted"]
        ],
        "no_improvement_narrative": no_improvement_narrative,
    }
    write_json(out_path, payload)
    return RecommendationResult(report_path=out_path, payload=payload)
