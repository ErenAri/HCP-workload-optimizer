from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.utils.io import ensure_dir, write_json


@dataclass
class StressScenarioResult:
    dataset_path: Path
    metadata_path: Path


def _base_dataframe(n_jobs: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    submit_ts = []
    current_ts = 0
    for _ in range(n_jobs):
        current_ts += random.randint(1, 30)
        submit_ts.append(current_ts)

    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(submit_ts, start=1):
        runtime = random.randint(60, 3600)
        wait = random.randint(0, 300)
        start_ts = ts + wait
        rows.append(
            {
                "job_id": i,
                "submit_ts": ts,
                "start_ts": start_ts,
                "end_ts": start_ts + runtime,
                "wait_sec": wait,
                "runtime_actual_sec": runtime,
                "runtime_requested_sec": int(runtime * random.uniform(1.0, 2.5)),
                "allocated_cpus": random.choice([1, 2, 4, 8, 16]),
                "requested_cpus": random.choice([1, 2, 4, 8, 16]),
                "requested_mem": None,
                "status": 1,
                "user_id": random.randint(1, 40),
                "group_id": random.randint(1, 10),
                "queue_id": 1,
                "partition_id": 1,
                "runtime_overrequest_ratio": None,
            }
        )
    df = pd.DataFrame(rows)
    df["runtime_overrequest_ratio"] = (
        df["runtime_requested_sec"] / df["runtime_actual_sec"]
    )
    return df


def generate_stress_scenario(
    scenario: str,
    out_dir: Path,
    n_jobs: int = 5000,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> StressScenarioResult:
    params = params or {}
    ensure_dir(out_dir)
    df = _base_dataframe(n_jobs=n_jobs, seed=seed)

    if scenario == "heavy_tail":
        alpha = float(params.get("alpha", 1.2))
        # Pareto-like long tail on runtime
        df["runtime_actual_sec"] = (
            60 * (1 + (pd.Series([random.paretovariate(alpha) for _ in range(len(df))])))
        ).astype(int).clip(upper=7 * 24 * 3600)
        df["runtime_requested_sec"] = (df["runtime_actual_sec"] * 1.5).astype(int)
    elif scenario == "low_congestion":
        target_util = float(params.get("target_util", 0.35))
        spacing = int(max(10, (1.0 / max(target_util, 0.05)) * 30))
        df["submit_ts"] = [i * spacing for i in range(1, len(df) + 1)]
        df["wait_sec"] = 0
        df["start_ts"] = df["submit_ts"]
        df["end_ts"] = df["start_ts"] + df["runtime_actual_sec"]
    elif scenario == "user_skew":
        top_user_share = float(params.get("top_user_share", 0.65))
        top_user_count = int(len(df) * top_user_share)
        df.loc[: top_user_count - 1, "user_id"] = 1
    elif scenario == "burst_shock":
        burst_factor = int(params.get("burst_factor", 4))
        burst_duration_sec = int(params.get("burst_duration_sec", 1800))
        burst_start = int(df["submit_ts"].quantile(0.4))
        mask = (df["submit_ts"] >= burst_start) & (
            df["submit_ts"] < burst_start + burst_duration_sec
        )
        df.loc[mask, "submit_ts"] = burst_start + (
            (df.loc[mask, "submit_ts"] - burst_start) // burst_factor
        )
        df = df.sort_values("submit_ts").reset_index(drop=True)
        df["job_id"] = range(1, len(df) + 1)
    else:
        raise ValueError(
            f"Unsupported scenario '{scenario}'. Use heavy_tail, low_congestion, user_skew, or burst_shock."
        )

    dataset_path = out_dir / f"stress_{scenario}.parquet"
    metadata_path = out_dir / f"stress_{scenario}_metadata.json"
    df.to_parquet(dataset_path, index=False)
    write_json(
        metadata_path,
        {
            "scenario": scenario,
            "n_jobs": int(len(df)),
            "seed": seed,
            "params": params,
            "dataset_path": str(dataset_path),
        },
    )
    return StressScenarioResult(dataset_path=dataset_path, metadata_path=metadata_path)
