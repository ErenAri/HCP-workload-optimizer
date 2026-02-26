"""Policy sensitivity sweep: vary runtime_guard_k and measure metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor
from hpcopt.simulate.core import run_simulation_from_trace
from hpcopt.simulate.objective import evaluate_constraint_contract
from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensitivitySweepResult:
    report_path: Path
    metrics_df: pd.DataFrame
    payload: dict[str, Any]


def run_guard_k_sweep(
    trace_df: pd.DataFrame,
    capacity_cpus: int,
    k_values: list[float] | None = None,
    model_dir: Path | None = None,
    seed: int = 42,
    baseline_policy: str = "EASY_BACKFILL_BASELINE",
    strict_invariants: bool = True,
) -> dict[str, Any]:
    """Run simulation for each k value and collect metrics."""
    if k_values is None:
        k_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    predictor = None
    if model_dir is not None and model_dir.exists():
        predictor = RuntimeQuantilePredictor(model_dir)

    # Run baseline once
    baseline_sim = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=baseline_policy,
        capacity_cpus=capacity_cpus,
        run_id="sensitivity_baseline",
        strict_invariants=strict_invariants,
    )
    baseline_obj = baseline_sim.objective_metrics

    sweep_rows: list[dict[str, Any]] = []
    for k in k_values:
        logger.info("Sweep: k=%.2f", k)
        try:
            sim = run_simulation_from_trace(
                trace_df=trace_df,
                policy_id="ML_BACKFILL_P50",
                capacity_cpus=capacity_cpus,
                run_id=f"sensitivity_k{k:.2f}",
                strict_invariants=strict_invariants,
                runtime_predictor=predictor,
                runtime_guard_k=k,
            )
            obj = sim.objective_metrics
            constraints = evaluate_constraint_contract(candidate=obj, baseline=baseline_obj)
            delta_p95 = float(baseline_obj["p95_bsld"] - obj["p95_bsld"])
            delta_util = float(obj["utilization_cpu"] - baseline_obj["utilization_cpu"])

            sweep_rows.append(
                {
                    "guard_k": float(k),
                    "p95_bsld": float(obj["p95_bsld"]),
                    "utilization_cpu": float(obj["utilization_cpu"]),
                    "fairness_dev": float(obj["fairness_dev"]),
                    "jain": float(obj["jain"]),
                    "starved_rate": float(obj["starved_rate"]),
                    "p95_wait_sec": float(obj["p95_wait_sec"]),
                    "delta_p95_bsld": delta_p95,
                    "delta_utilization": delta_util,
                    "constraints_passed": constraints["constraints_passed"],
                    "violations": constraints["violations"],
                    "prediction_used_rate": float(sim.fallback_accounting.get("prediction_used_rate", 0)),
                    "status": "ok",
                }
            )
        except (ValueError, OSError) as exc:
            logger.warning("Sweep failed for k=%.2f: %s", k, exc)
            sweep_rows.append(
                {
                    "guard_k": float(k),
                    "status": "error",
                    "error": str(exc),
                }
            )

    return {
        "baseline": {
            "policy_id": baseline_policy,
            "objective_metrics": baseline_obj,
        },
        "sweep": sweep_rows,
    }


def build_sensitivity_report(
    sweep_results: dict[str, Any],
    out_path: Path,
) -> SensitivitySweepResult:
    """Build structured sensitivity report from sweep results."""
    ensure_dir(out_path.parent)

    sweep_rows = sweep_results["sweep"]
    baseline = sweep_results["baseline"]

    # Find optimal k region: best delta_p95_bsld with constraints passing
    ok_rows = [r for r in sweep_rows if r.get("status") == "ok" and r.get("constraints_passed")]
    best_k: float | None = None
    best_delta: float = -float("inf")
    if ok_rows:
        for row in ok_rows:
            d = float(row.get("delta_p95_bsld", 0))
            if d > best_delta:
                best_delta = d
                best_k = float(row["guard_k"])

    # Compute effect sizes
    ok_deltas = [float(r.get("delta_p95_bsld", 0)) for r in sweep_rows if r.get("status") == "ok"]
    effect_size_range = (min(ok_deltas), max(ok_deltas)) if ok_deltas else (0.0, 0.0)

    metrics_df = pd.DataFrame([r for r in sweep_rows if r.get("status") == "ok"])

    payload = {
        "baseline": baseline,
        "sweep_results": sweep_rows,
        "analysis": {
            "k_values_tested": len(sweep_rows),
            "k_values_passed_constraints": len(ok_rows),
            "optimal_k": best_k,
            "optimal_delta_p95_bsld": best_delta if best_k is not None else None,
            "effect_size_range": {
                "min_delta_p95_bsld": effect_size_range[0],
                "max_delta_p95_bsld": effect_size_range[1],
            },
        },
    }

    write_json(out_path, payload)
    return SensitivitySweepResult(
        report_path=out_path,
        metrics_df=metrics_df,
        payload=payload,
    )
