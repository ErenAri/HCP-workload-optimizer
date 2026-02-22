from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpcopt.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_dataset_path: Path
    split_manifest_path: Path
    feature_report_path: Path
    row_count: int
    fold_count: int


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature build missing required columns: {missing}")


def _coerce_frame(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"job_id", "submit_ts", "start_ts", "end_ts", "runtime_actual_sec", "requested_cpus"}
    _require_columns(trace_df, required)
    df = trace_df.copy()

    for col in ("runtime_requested_sec", "requested_mem", "user_id", "group_id", "queue_id", "partition_id"):
        if col not in df.columns:
            df[col] = None

    for col in ("job_id", "submit_ts", "start_ts", "end_ts", "runtime_actual_sec", "requested_cpus"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["job_id", "submit_ts", "start_ts", "end_ts", "runtime_actual_sec", "requested_cpus"])
    if df.empty:
        raise ValueError("feature build produced empty frame after coercion")

    df["job_id"] = df["job_id"].astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)
    df["start_ts"] = df["start_ts"].astype(int)
    df["end_ts"] = df["end_ts"].astype(int)
    df["runtime_actual_sec"] = df["runtime_actual_sec"].clip(lower=0).astype(float)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(int)
    df["runtime_requested_sec"] = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
    df["requested_mem"] = pd.to_numeric(df["requested_mem"], errors="coerce")

    df = df.sort_values(["submit_ts", "job_id"], kind="mergesort").reset_index(drop=True)
    df["submit_dt"] = pd.to_datetime(df["submit_ts"], unit="s", utc=True)
    return df


def _queue_at_submit_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute queue depth at each job's submit time without future leakage.

    Uses NumPy arrays and ``np.lexsort`` instead of Python lists/dicts for
    lower overhead on large traces.  Submit events are processed before start
    events at equal timestamps (event_type 0 < 1).
    """
    n = len(df)
    submit_ts = df["submit_ts"].to_numpy(dtype=np.int64)
    start_ts = df["start_ts"].to_numpy(dtype=np.int64)
    cpus = df["requested_cpus"].to_numpy(dtype=np.int64)

    # Build a (2n,) event array: (timestamp, event_type, row_id, cpus).
    timestamps = np.concatenate([submit_ts, start_ts])
    event_types = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    row_ids = np.concatenate([np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64)])
    event_cpus = np.concatenate([cpus, cpus])

    # Stable sort by (timestamp, event_type, row_id).
    order = np.lexsort((row_ids, event_types, timestamps))

    # Walk events sequentially (inherently serial due to running state).
    queue_len_arr = np.zeros(n, dtype=np.int64)
    queue_cpu_arr = np.zeros(n, dtype=np.int64)
    queue_len = 0
    queue_cpu = 0
    for idx in order:
        if event_types[idx] == 0:  # submit
            rid = row_ids[idx]
            queue_len_arr[rid] = queue_len
            queue_cpu_arr[rid] = queue_cpu
            queue_len += 1
            queue_cpu += event_cpus[idx]
        else:  # start
            queue_len = max(0, queue_len - 1)
            queue_cpu = max(0, queue_cpu - int(event_cpus[idx]))

    return (
        pd.Series(queue_len_arr, index=df.index, dtype="int64"),
        pd.Series(queue_cpu_arr, index=df.index, dtype="int64"),
    )


def _job_size_class(requested_cpus: pd.Series) -> pd.Series:
    # Fixed thresholds avoid dataset-leakage from future quantiles.
    values = requested_cpus.to_numpy(dtype=int)
    out = np.where(
        values <= 4,
        0,
        np.where(values <= 16, 1, np.where(values <= 64, 2, 3)),
    )
    return pd.Series(out, index=requested_cpus.index, dtype="int64")


def _segment_summary(df: pd.DataFrame, start: int, end_exclusive: int) -> dict[str, int]:
    view = df.iloc[start:end_exclusive]
    if view.empty:
        return {
            "start_idx": start,
            "end_idx_exclusive": end_exclusive,
            "row_count": 0,
            "submit_ts_min": 0,
            "submit_ts_max": 0,
        }
    return {
        "start_idx": start,
        "end_idx_exclusive": end_exclusive,
        "row_count": int(len(view)),
        "submit_ts_min": int(view["submit_ts"].min()),
        "submit_ts_max": int(view["submit_ts"].max()),
    }


def build_chronological_splits(
    features_df: pd.DataFrame,
    n_folds: int = 3,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0,1)")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    n = len(features_df)
    if n < 3:
        raise ValueError("need at least 3 rows to build chronological splits")

    base_train_end = max(1, int(round(n * train_fraction)))
    base_val_size = max(1, int(round(n * val_fraction)))
    if base_train_end + base_val_size >= n:
        base_train_end = max(1, n - 2)
        base_val_size = 1

    available_test = max(1, n - base_train_end - base_val_size)
    fold_test_size = max(1, available_test // n_folds)

    folds: list[dict[str, Any]] = []
    for fold_idx in range(n_folds):
        train_end = base_train_end + (fold_idx * fold_test_size)
        if train_end >= n - 1:
            break
        val_start = train_end
        val_end = min(n - 1, val_start + base_val_size)
        test_start = val_end
        if test_start >= n:
            break
        test_end = n if fold_idx == n_folds - 1 else min(n, test_start + fold_test_size)
        if test_end <= test_start:
            continue

        train = _segment_summary(features_df, 0, train_end)
        val = _segment_summary(features_df, val_start, val_end)
        test = _segment_summary(features_df, test_start, test_end)
        if train["row_count"] == 0 or val["row_count"] == 0 or test["row_count"] == 0:
            continue

        chronology_ok = (
            train["submit_ts_max"] <= val["submit_ts_min"]
            and val["submit_ts_max"] <= test["submit_ts_min"]
        )
        folds.append(
            {
                "fold_id": int(fold_idx + 1),
                "mode": "anchored_expanding",
                "chronology_ok": bool(chronology_ok),
                "train": train,
                "validation": val,
                "test": test,
            }
        )

    if not folds:
        # Safe fallback for tiny datasets.
        train_end = max(1, n - 2)
        val_end = n - 1
        folds.append(
            {
                "fold_id": 1,
                "mode": "single_fallback",
                "chronology_ok": True,
                "train": _segment_summary(features_df, 0, train_end),
                "validation": _segment_summary(features_df, train_end, val_end),
                "test": _segment_summary(features_df, val_end, n),
            }
        )
    return folds


def build_feature_dataset(
    dataset_path: Path,
    out_dir: Path,
    report_dir: Path,
    dataset_id: str,
    n_folds: int = 3,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> FeatureBuildResult:
    ensure_dir(out_dir)
    ensure_dir(report_dir)
    source_df = pd.read_parquet(dataset_path)
    df = _coerce_frame(source_df)

    # Time and size signals.
    df["submit_hour"] = df["submit_dt"].dt.hour.astype(int)
    df["submit_dow"] = df["submit_dt"].dt.dayofweek.astype(int)
    df["is_peak_hours"] = ((df["submit_hour"] >= 8) & (df["submit_hour"] <= 18)).astype("int64")
    df["job_size_class"] = _job_size_class(df["requested_cpus"])
    df["time_since_prev_submit_sec"] = df["submit_ts"].diff().fillna(0).clip(lower=0).astype("int64")

    # Queue congestion at submit time (no future leakage).
    queue_jobs, queue_cpu = _queue_at_submit_features(df)
    df["queue_congestion_at_submit_jobs"] = queue_jobs.astype("int64")
    df["queue_congestion_at_submit_cpu"] = queue_cpu.astype("int64")

    # User lookback features.
    df["user_key"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype(int)
    group_runtime = df.groupby("user_key", sort=False)["runtime_actual_sec"]
    df["user_runtime_median_lookback"] = group_runtime.transform(
        lambda s: s.shift(1).expanding().median()
    )
    df["user_runtime_var_lookback"] = group_runtime.transform(
        lambda s: s.shift(1).expanding().var(ddof=0)
    )
    df["user_job_count_lookback"] = df.groupby("user_key", sort=False).cumcount().astype("int64")
    df["user_submit_gap_sec_lookback"] = (
        df.groupby("user_key", sort=False)["submit_ts"].diff().fillna(0).clip(lower=0).astype("int64")
    )

    overrequest_ratio = (
        df["runtime_requested_sec"] / df["runtime_actual_sec"].replace(0.0, np.nan)
    )
    group_or = overrequest_ratio.groupby(df["user_key"], sort=False)
    df["user_overrequest_mean_lookback"] = group_or.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    global_runtime_median = float(df["runtime_actual_sec"].median()) if not df.empty else 0.0
    global_runtime_var = float(df["runtime_actual_sec"].var(ddof=0)) if len(df) > 1 else 0.0
    global_or_median = float(overrequest_ratio.dropna().median()) if overrequest_ratio.notna().any() else 1.0

    df["user_runtime_median_lookback"] = df["user_runtime_median_lookback"].fillna(global_runtime_median)
    df["user_runtime_var_lookback"] = df["user_runtime_var_lookback"].fillna(global_runtime_var)
    df["user_overrequest_mean_lookback"] = df["user_overrequest_mean_lookback"].fillna(global_or_median)

    # User behavior class from lookback over-request tendency.
    df["user_behavior_pattern"] = np.where(
        df["user_overrequest_mean_lookback"] > 1.30,
        2,
        np.where(df["user_overrequest_mean_lookback"] < 0.90, 0, 1),
    ).astype("int64")

    feature_cols = [
        "job_id",
        "submit_ts",
        "start_ts",
        "end_ts",
        "runtime_actual_sec",
        "runtime_requested_sec",
        "requested_cpus",
        "requested_mem",
        "user_id",
        "group_id",
        "queue_id",
        "partition_id",
        "submit_hour",
        "submit_dow",
        "is_peak_hours",
        "job_size_class",
        "time_since_prev_submit_sec",
        "queue_congestion_at_submit_jobs",
        "queue_congestion_at_submit_cpu",
        "user_runtime_median_lookback",
        "user_runtime_var_lookback",
        "user_job_count_lookback",
        "user_submit_gap_sec_lookback",
        "user_overrequest_mean_lookback",
        "user_behavior_pattern",
    ]

    features_df = df[feature_cols].copy()
    folds = build_chronological_splits(
        features_df=features_df,
        n_folds=n_folds,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    feature_dataset_path = out_dir / f"{dataset_id}_features.parquet"
    split_manifest_path = report_dir / f"{dataset_id}_feature_splits.json"
    feature_report_path = report_dir / f"{dataset_id}_feature_report.json"

    features_df.to_parquet(feature_dataset_path, index=False)
    write_json(
        split_manifest_path,
        {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "feature_dataset_path": str(feature_dataset_path),
            "n_rows": int(len(features_df)),
            "n_folds_requested": int(n_folds),
            "n_folds_generated": int(len(folds)),
            "folds": folds,
            "split_contract": {
                "mode": "chronological_no_leakage",
                "train_fraction": float(train_fraction),
                "val_fraction": float(val_fraction),
            },
        },
    )
    write_json(
        feature_report_path,
        {
            "dataset_id": dataset_id,
            "feature_dataset_path": str(feature_dataset_path),
            "row_count": int(len(features_df)),
            "column_count": int(len(features_df.columns)),
            "null_rates": {
                col: float(features_df[col].isna().mean())
                for col in features_df.columns
            },
            "lookback_features": [
                "user_runtime_median_lookback",
                "user_runtime_var_lookback",
                "user_job_count_lookback",
                "user_submit_gap_sec_lookback",
                "user_overrequest_mean_lookback",
                "user_behavior_pattern",
            ],
        },
    )
    return FeatureBuildResult(
        feature_dataset_path=feature_dataset_path,
        split_manifest_path=split_manifest_path,
        feature_report_path=feature_report_path,
        row_count=int(len(features_df)),
        fold_count=int(len(folds)),
    )
