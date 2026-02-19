from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hpcopt.simulate.metrics import compute_job_metrics


def compute_fairness_starvation_metrics(
    jobs_df: pd.DataFrame,
    starvation_wait_cap_sec: int = 172800,
) -> dict[str, float]:
    if jobs_df.empty:
        return {
            "starved_rate": 0.0,
            "starved_jobs": 0.0,
            "total_jobs": 0.0,
            "fairness_dev": 0.0,
            "jain": 1.0,
            "active_users": 0.0,
        }

    wait = (jobs_df["start_ts"] - jobs_df["submit_ts"]).clip(lower=0).to_numpy(dtype=float)
    starved = wait > float(starvation_wait_cap_sec)
    starved_rate = float(starved.mean() if starved.size else 0.0)
    starved_jobs = float(starved.sum())

    runtime = (jobs_df["end_ts"] - jobs_df["start_ts"]).clip(lower=0).to_numpy(dtype=float)
    requested = jobs_df["requested_cpus"].to_numpy(dtype=float)
    user_series = jobs_df.get("user_id")
    if user_series is None:
        user_series = pd.Series([-1] * len(jobs_df))
    users = pd.to_numeric(user_series, errors="coerce").fillna(-1).astype(int)

    cpu_sec = requested * runtime
    user_cpu = pd.DataFrame({"user_id": users, "cpu_sec": cpu_sec}).groupby("user_id", as_index=False)[
        "cpu_sec"
    ].sum()
    total_cpu = float(user_cpu["cpu_sec"].sum())
    if total_cpu <= 0:
        fairness_dev = 0.0
        jain = 1.0
        active_users = 0.0
    else:
        shares = (user_cpu["cpu_sec"] / total_cpu).to_numpy(dtype=float)
        active_users = float(len(shares))
        target = 1.0 / len(shares) if len(shares) else 0.0
        fairness_dev = float(0.5 * np.sum(np.abs(shares - target))) if len(shares) else 0.0
        denom = float(len(shares) * np.sum(shares ** 2))
        jain = float((np.sum(shares) ** 2) / denom) if denom > 0 else 1.0

    return {
        "starved_rate": starved_rate,
        "starved_jobs": starved_jobs,
        "total_jobs": float(len(jobs_df)),
        "fairness_dev": fairness_dev,
        "jain": jain,
        "active_users": active_users,
    }


def compute_objective_contract_metrics(
    jobs_df: pd.DataFrame,
    capacity_cpus: int,
    starvation_wait_cap_sec: int = 172800,
) -> dict[str, float]:
    base = compute_job_metrics(jobs_df=jobs_df, capacity_cpus=capacity_cpus)
    fairness = compute_fairness_starvation_metrics(
        jobs_df=jobs_df,
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )
    return {**base, **fairness}


def evaluate_constraint_contract(
    candidate: dict[str, float],
    baseline: dict[str, float],
    starvation_rate_max: float = 0.02,
    fairness_dev_delta_max: float = 0.05,
    jain_delta_max: float = 0.03,
) -> dict[str, Any]:
    violations: list[str] = []
    if candidate["starved_rate"] > starvation_rate_max:
        violations.append("starved_rate_exceeded")
    if (candidate["fairness_dev"] - baseline["fairness_dev"]) > fairness_dev_delta_max:
        violations.append("fairness_dev_delta_exceeded")
    if (baseline["jain"] - candidate["jain"]) > jain_delta_max:
        violations.append("jain_delta_exceeded")
    return {
        "constraints_passed": len(violations) == 0,
        "violations": violations,
        "thresholds": {
            "starvation_rate_max": starvation_rate_max,
            "fairness_dev_delta_max": fairness_dev_delta_max,
            "jain_delta_max": jain_delta_max,
        },
    }


def compute_weighted_analysis_score(
    candidate: dict[str, float],
    baseline: dict[str, float],
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 2.0,
) -> dict[str, float]:
    delta_p95_bsld = float(baseline["p95_bsld"] - candidate["p95_bsld"])
    delta_utilization = float(candidate["utilization_cpu"] - baseline["utilization_cpu"])
    fairness_penalty = float(
        max(0.0, candidate["fairness_dev"] - baseline["fairness_dev"])
        + max(0.0, baseline["jain"] - candidate["jain"])
    )
    score = float((w1 * delta_p95_bsld) + (w2 * delta_utilization) - (w3 * fairness_penalty))
    return {
        "score": score,
        "delta_p95_bsld": delta_p95_bsld,
        "delta_utilization": delta_utilization,
        "fairness_penalty": fairness_penalty,
        "weights": {"w1": w1, "w2": w2, "w3": w3},
    }
