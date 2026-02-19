from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.95))


def compute_job_metrics(jobs_df: pd.DataFrame, capacity_cpus: int) -> dict[str, float]:
    if jobs_df.empty:
        return {
            "job_count": 0.0,
            "mean_wait_sec": 0.0,
            "p95_wait_sec": 0.0,
            "throughput": 0.0,
            "makespan_sec": 0.0,
            "utilization_cpu": 0.0,
            "p95_bsld": 0.0,
        }

    wait = (jobs_df["start_ts"] - jobs_df["submit_ts"]).clip(lower=0).to_numpy(dtype=float)
    runtime = (jobs_df["end_ts"] - jobs_df["start_ts"]).clip(lower=0).to_numpy(dtype=float)
    submit_min = int(jobs_df["submit_ts"].min())
    end_max = int(jobs_df["end_ts"].max())
    makespan = max(end_max - submit_min, 0)
    evaluation_duration = float(makespan)

    if evaluation_duration <= 0.0:
        throughput = 0.0
        utilization = 0.0
    else:
        throughput = float(len(jobs_df) / evaluation_duration)
        cpu_sec = (
            jobs_df["requested_cpus"].to_numpy(dtype=float) * runtime
        ).sum()
        utilization = float(cpu_sec / (capacity_cpus * evaluation_duration))
        utilization = max(0.0, min(1.0, utilization))

    bsld = (wait + runtime) / np.maximum(runtime, 60.0)

    return {
        "job_count": float(len(jobs_df)),
        "mean_wait_sec": float(wait.mean() if wait.size else 0.0),
        "p95_wait_sec": _p95(wait),
        "throughput": throughput,
        "makespan_sec": float(makespan),
        "utilization_cpu": utilization,
        "p95_bsld": _p95(bsld),
    }


def relative_divergence(observed: float, simulated: float) -> float:
    denom = abs(observed) if abs(observed) > 1e-9 else 1.0
    return float(abs(simulated - observed) / denom)


def wait_kl_divergence(observed_wait: np.ndarray, simulated_wait: np.ndarray) -> float:
    if observed_wait.size == 0 or simulated_wait.size == 0:
        return 0.0

    max_wait = float(max(observed_wait.max(), simulated_wait.max(), 1.0))
    bins = np.linspace(0.0, max_wait, 51)
    if np.unique(bins).size < 2:
        return 0.0

    p_hist, _ = np.histogram(observed_wait, bins=bins)
    q_hist, _ = np.histogram(simulated_wait, bins=bins)

    epsilon = 1e-9
    p = (p_hist + epsilon).astype(float)
    q = (q_hist + epsilon).astype(float)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def ks_statistic(observed: np.ndarray, simulated: np.ndarray) -> float:
    if observed.size == 0 or simulated.size == 0:
        return 0.0
    observed_sorted = np.sort(observed)
    simulated_sorted = np.sort(simulated)
    union = np.sort(np.concatenate([observed_sorted, simulated_sorted]))
    cdf_obs = np.searchsorted(observed_sorted, union, side="right") / observed_sorted.size
    cdf_sim = np.searchsorted(simulated_sorted, union, side="right") / simulated_sorted.size
    return float(np.max(np.abs(cdf_obs - cdf_sim)))


def resample_queue_series_right_hold(
    queue_series: pd.DataFrame,
    start_ts: int,
    end_ts: int,
    cadence_sec: int = 60,
) -> pd.DataFrame:
    if cadence_sec <= 0:
        raise ValueError("cadence_sec must be > 0")
    if end_ts < start_ts:
        end_ts = start_ts

    sample_ts = np.arange(start_ts, end_ts + 1, cadence_sec, dtype=int)
    sample_df = pd.DataFrame({"ts": sample_ts.astype("int64")})
    source = queue_series.sort_values("ts")[["ts", "queue_len_jobs"]].copy()
    source["ts"] = source["ts"].astype("int64")
    merged = pd.merge_asof(sample_df, source, on="ts", direction="backward")
    merged["queue_len_jobs"] = merged["queue_len_jobs"].fillna(0.0)
    return merged


def queue_series_correlation(
    observed_queue: pd.DataFrame,
    simulated_queue: pd.DataFrame,
    start_ts: int,
    end_ts: int,
    cadence_sec: int = 60,
) -> float:
    obs = resample_queue_series_right_hold(observed_queue, start_ts, end_ts, cadence_sec)
    sim = resample_queue_series_right_hold(simulated_queue, start_ts, end_ts, cadence_sec)

    obs_values = obs["queue_len_jobs"].to_numpy(dtype=float)
    sim_values = sim["queue_len_jobs"].to_numpy(dtype=float)

    obs_std = float(obs_values.std())
    sim_std = float(sim_values.std())

    if obs_std == 0.0 and sim_std == 0.0:
        return 1.0 if np.allclose(obs_values, sim_values) else 0.0
    if obs_std == 0.0 or sim_std == 0.0:
        return 0.0

    obs_norm = (obs_values - obs_values.mean()) / obs_std
    sim_norm = (sim_values - sim_values.mean()) / sim_std
    corr = np.corrcoef(obs_norm, sim_norm)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def wait_and_slowdown_arrays(jobs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if jobs_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    wait = (jobs_df["start_ts"] - jobs_df["submit_ts"]).clip(lower=0).to_numpy(dtype=float)
    runtime = (jobs_df["end_ts"] - jobs_df["start_ts"]).clip(lower=0).to_numpy(dtype=float)
    slowdown = (wait + runtime) / np.maximum(runtime, 60.0)
    return wait, slowdown
