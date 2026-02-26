from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hpcopt.utils.io import ensure_dir, write_json


@dataclass
class ProfileResult:
    profile_path: Path
    row_count: int


def _queue_series(df: pd.DataFrame) -> pd.DataFrame:
    events: list[tuple[int, int]] = []
    for row in df.itertuples(index=False):
        events.append((int(row.submit_ts), +1))
        events.append((int(row.start_ts), -1))
    event_df = pd.DataFrame(events, columns=["ts", "delta"]).groupby("ts", as_index=False)["delta"].sum()
    event_df = event_df.sort_values("ts").reset_index(drop=True)
    event_df["queue_len_jobs"] = event_df["delta"].cumsum().clip(lower=0)
    return event_df


def _float_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def build_trace_profile(dataset_path: Path, report_dir: Path, dataset_id: str) -> ProfileResult:
    ensure_dir(report_dir)
    df = pd.read_parquet(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty; cannot build profile.")

    runtime = df["runtime_actual_sec"].fillna(0)
    req_runtime = df["runtime_requested_sec"]
    queue_series = _queue_series(df)

    overreq = (
        df.loc[
            (req_runtime.notna()) & (runtime > 0),
            "runtime_requested_sec",
        ]
        / df.loc[(req_runtime.notna()) & (runtime > 0), "runtime_actual_sec"]
    )
    user_counts = df["user_id"].fillna(-1).astype(int).value_counts()
    total_jobs = len(df)
    top_user_share = float(user_counts.iloc[0] / total_jobs) if len(user_counts) else 0.0
    top5_share = float(user_counts.iloc[:5].sum() / total_jobs) if len(user_counts) else 0.0
    shares = user_counts / user_counts.sum()
    user_hhi = float((shares**2).sum()) if len(user_counts) else 0.0

    profile = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "row_count": int(total_jobs),
        "time_range": {
            "submit_min": int(df["submit_ts"].min()),
            "submit_max": int(df["submit_ts"].max()),
            "start_min": int(df["start_ts"].min()),
            "end_max": int(df["end_ts"].max()),
        },
        "job_size_distribution": {
            "requested_cpus_p50": _float_or_none(df["requested_cpus"].quantile(0.5)),
            "requested_cpus_p90": _float_or_none(df["requested_cpus"].quantile(0.9)),
            "requested_cpus_p99": _float_or_none(df["requested_cpus"].quantile(0.99)),
            "requested_cpus_max": _float_or_none(df["requested_cpus"].max()),
        },
        "runtime_heavy_tail": {
            "runtime_p50_sec": _float_or_none(runtime.quantile(0.5)),
            "runtime_p90_sec": _float_or_none(runtime.quantile(0.9)),
            "runtime_p95_sec": _float_or_none(runtime.quantile(0.95)),
            "runtime_p99_sec": _float_or_none(runtime.quantile(0.99)),
            "tail_ratio_p99_over_p50": _float_or_none(
                runtime.quantile(0.99) / runtime.quantile(0.5) if runtime.quantile(0.5) > 0 else None
            ),
        },
        "overrequest_distribution": {
            "sample_size": int(overreq.shape[0]),
            "ratio_p50": _float_or_none(overreq.quantile(0.5)) if not overreq.empty else None,
            "ratio_p90": _float_or_none(overreq.quantile(0.9)) if not overreq.empty else None,
            "ratio_p95": _float_or_none(overreq.quantile(0.95)) if not overreq.empty else None,
        },
        "congestion_regime": {
            "queue_len_mean": _float_or_none(queue_series["queue_len_jobs"].mean()),
            "queue_len_p95": _float_or_none(queue_series["queue_len_jobs"].quantile(0.95)),
            "queue_len_max": _float_or_none(queue_series["queue_len_jobs"].max()),
            "event_count": int(queue_series.shape[0]),
        },
        "user_skew": {
            "unique_users": int(user_counts.shape[0]),
            "top_user_share": top_user_share,
            "top_5_users_share": top5_share,
            "user_hhi": user_hhi,
        },
    }
    profile_path = report_dir / f"{dataset_id}_trace_profile.json"
    write_json(profile_path, profile)
    return ProfileResult(profile_path=profile_path, row_count=int(total_jobs))
