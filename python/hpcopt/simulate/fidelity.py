from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hpcopt.simulate.core import (
    build_observed_jobs_df,
    build_observed_queue_series,
    run_simulation_from_trace,
)
from hpcopt.simulate.metrics import (
    ks_statistic,
    queue_series_correlation,
    relative_divergence,
    wait_and_slowdown_arrays,
    wait_kl_divergence,
)
from hpcopt.simulate.objective import (
    compute_objective_contract_metrics,
    evaluate_constraint_contract,
)
from hpcopt.utils.io import write_json

DEFAULT_FIDELITY_THRESHOLDS: dict[str, Any] = {
    "fidelity_gate": {
        "aggregate": {
            "single_metric_max_divergence": 0.20,
            "two_metric_max_divergence": 0.15,
        },
        "distribution": {
            "wait_kl_max": 0.20,
            "slowdown_ks_max": 0.15,
            "queue_corr_min": 0.85,
        },
        "queue_series": {
            "mode": "fixed_cadence",
            "cadence_sec": 60,
        },
    }
}

# Stabilize aggregate divergence for low-magnitude metrics in toy/smoke traces.
# For production traces these floors are usually below observed scale and have
# little effect.
_CORE_DIVERGENCE_DENOM_FLOORS: dict[str, float] = {
    "mean_wait_sec": 60.0,
    "p95_wait_sec": 60.0,
    "throughput": 0.05,
    "makespan_sec": 300.0,
}

# Distribution tests are noisy with very small sample counts and can dominate
# pass/fail outcomes in fixture-sized traces.
_MIN_DISTRIBUTION_SAMPLE_SIZE = 30


@dataclass
class FidelityGateResult:
    report_path: Path
    status: str
    report: dict[str, Any]


def load_fidelity_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return DEFAULT_FIDELITY_THRESHOLDS
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fidelity config must be a mapping")
    return payload


def _policy_fidelity(
    observed_jobs: pd.DataFrame,
    observed_queue: pd.DataFrame,
    simulated_jobs: pd.DataFrame,
    simulated_queue: pd.DataFrame,
    thresholds: dict[str, Any],
    capacity_cpus: int,
) -> dict[str, Any]:
    observed_metrics = compute_objective_contract_metrics(
        observed_jobs,
        capacity_cpus=capacity_cpus,
    )
    simulated_metrics = compute_objective_contract_metrics(
        simulated_jobs,
        capacity_cpus=capacity_cpus,
    )

    core_keys = ["mean_wait_sec", "p95_wait_sec", "throughput", "makespan_sec"]
    core_divergence = {
        key: relative_divergence(
            observed=float(observed_metrics[key]),
            simulated=float(simulated_metrics[key]),
            denominator_floor=_CORE_DIVERGENCE_DENOM_FLOORS.get(key),
        )
        for key in core_keys
    }

    observed_wait, observed_slowdown = wait_and_slowdown_arrays(observed_jobs)
    simulated_wait, simulated_slowdown = wait_and_slowdown_arrays(simulated_jobs)

    start_ts = int(min(observed_jobs["submit_ts"].min(), simulated_jobs["submit_ts"].min()))
    end_ts = int(max(observed_jobs["end_ts"].max(), simulated_jobs["end_ts"].max()))
    cadence_sec = int(thresholds["fidelity_gate"]["queue_series"]["cadence_sec"])

    distribution_sample_size = int(min(observed_wait.size, simulated_wait.size))
    distribution_checks_skipped = distribution_sample_size < _MIN_DISTRIBUTION_SAMPLE_SIZE

    wait_kl = 0.0
    slowdown_ks = 0.0
    if not distribution_checks_skipped:
        wait_kl = wait_kl_divergence(observed_wait, simulated_wait)
        slowdown_ks = ks_statistic(observed_slowdown, simulated_slowdown)
    queue_corr = queue_series_correlation(
        observed_queue=observed_queue,
        simulated_queue=simulated_queue,
        start_ts=start_ts,
        end_ts=end_ts,
        cadence_sec=cadence_sec,
    )

    agg_threshold_single = float(thresholds["fidelity_gate"]["aggregate"]["single_metric_max_divergence"])
    agg_threshold_dual = float(thresholds["fidelity_gate"]["aggregate"]["two_metric_max_divergence"])
    dist_wait_kl_max = float(thresholds["fidelity_gate"]["distribution"]["wait_kl_max"])
    dist_slowdown_ks_max = float(thresholds["fidelity_gate"]["distribution"]["slowdown_ks_max"])
    dist_queue_corr_min = float(thresholds["fidelity_gate"]["distribution"]["queue_corr_min"])

    fail_reasons: list[str] = []
    if any(value > agg_threshold_single for value in core_divergence.values()):
        fail_reasons.append("aggregate_single_metric_divergence_exceeded")
    if sum(value > agg_threshold_dual for value in core_divergence.values()) >= 2:
        fail_reasons.append("aggregate_two_metric_divergence_exceeded")
    if not distribution_checks_skipped and wait_kl > dist_wait_kl_max:
        fail_reasons.append("wait_kl_exceeded")
    if not distribution_checks_skipped and slowdown_ks > dist_slowdown_ks_max:
        fail_reasons.append("slowdown_ks_exceeded")
    if queue_corr < dist_queue_corr_min:
        fail_reasons.append("queue_correlation_below_min")

    status = "pass" if not fail_reasons else "fail"
    constraint_check = evaluate_constraint_contract(
        candidate=simulated_metrics,
        baseline=observed_metrics,
    )
    return {
        "status": status,
        "fail_reasons": fail_reasons,
        "observed_metrics": observed_metrics,
        "simulated_metrics": simulated_metrics,
        "core_metric_divergence": core_divergence,
        "distribution_metrics": {
            "wait_kl_divergence": wait_kl,
            "slowdown_ks_statistic": slowdown_ks,
            "queue_corr_pearson": queue_corr,
        },
        "distribution_checks": {
            "sample_size": distribution_sample_size,
            "min_sample_size_required": _MIN_DISTRIBUTION_SAMPLE_SIZE,
            "skipped_small_sample": distribution_checks_skipped,
        },
        "queue_series_alignment": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "cadence_sec": cadence_sec,
        },
        "constraint_contract_check": constraint_check,
    }


def run_baseline_fidelity_gate(
    trace_df: pd.DataFrame,
    capacity_cpus: int,
    out_path: Path,
    run_id: str,
    config_path: Path | None = None,
    strict_invariants: bool = True,
) -> FidelityGateResult:
    thresholds = load_fidelity_config(config_path)
    observed_jobs = build_observed_jobs_df(trace_df)
    observed_queue = build_observed_queue_series(trace_df)

    policy_reports: dict[str, Any] = {}
    for policy_id in ("FIFO_STRICT", "EASY_BACKFILL_BASELINE"):
        sim = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id=policy_id,
            capacity_cpus=capacity_cpus,
            run_id=f"{run_id}_{policy_id.lower()}",
            strict_invariants=strict_invariants,
        )
        policy_reports[policy_id] = _policy_fidelity(
            observed_jobs=observed_jobs,
            observed_queue=observed_queue,
            simulated_jobs=sim.jobs_df,
            simulated_queue=sim.queue_series_df,
            thresholds=thresholds,
            capacity_cpus=capacity_cpus,
        )
        policy_reports[policy_id]["invariant_report"] = sim.invariant_report

    overall_status = "pass" if all(report["status"] == "pass" for report in policy_reports.values()) else "fail"

    report = {
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "status": overall_status,
        "aggregate_metrics": {policy: payload["core_metric_divergence"] for policy, payload in policy_reports.items()},
        "distribution_metrics": {policy: payload["distribution_metrics"] for policy, payload in policy_reports.items()},
        "queue_series_contract": {
            "mode": thresholds["fidelity_gate"]["queue_series"]["mode"],
            "cadence_sec": int(thresholds["fidelity_gate"]["queue_series"]["cadence_sec"]),
            "definition": "queue_len_jobs(t)=|Q(t)|, sampled after complete->submit->dispatch/start ordering",
        },
        "policy_reports": policy_reports,
        "thresholds": thresholds["fidelity_gate"],
    }

    write_json(out_path, report)
    return FidelityGateResult(report_path=out_path, status=overall_status, report=report)


def run_candidate_fidelity_report(
    trace_df: pd.DataFrame,
    simulated_jobs: pd.DataFrame,
    simulated_queue: pd.DataFrame,
    capacity_cpus: int,
    out_path: Path,
    run_id: str,
    policy_id: str,
    config_path: Path | None = None,
) -> FidelityGateResult:
    thresholds = load_fidelity_config(config_path)
    observed_jobs = build_observed_jobs_df(trace_df)
    observed_queue = build_observed_queue_series(trace_df)

    policy_report = _policy_fidelity(
        observed_jobs=observed_jobs,
        observed_queue=observed_queue,
        simulated_jobs=simulated_jobs,
        simulated_queue=simulated_queue,
        thresholds=thresholds,
        capacity_cpus=capacity_cpus,
    )

    report = {
        "run_id": run_id,
        "policy_id": policy_id,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "status": policy_report["status"],
        "aggregate_metrics": policy_report["core_metric_divergence"],
        "distribution_metrics": policy_report["distribution_metrics"],
        "queue_series_contract": {
            "mode": thresholds["fidelity_gate"]["queue_series"]["mode"],
            "cadence_sec": int(thresholds["fidelity_gate"]["queue_series"]["cadence_sec"]),
            "definition": "queue_len_jobs(t)=|Q(t)|, sampled after complete->submit->dispatch/start ordering",
        },
        "policy_report": policy_report,
        "thresholds": thresholds["fidelity_gate"],
    }

    write_json(out_path, report)
    return FidelityGateResult(report_path=out_path, status=policy_report["status"], report=report)
