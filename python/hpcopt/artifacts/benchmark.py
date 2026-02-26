from __future__ import annotations

import datetime as dt
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpcopt.features.pipeline import build_feature_dataset
from hpcopt.ingest.swf import ingest_swf
from hpcopt.models.runtime_quantile import train_runtime_quantile_models
from hpcopt.profile.trace_profile import build_trace_profile
from hpcopt.simulate.core import run_simulation_from_trace
from hpcopt.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class BenchmarkRunResult:
    report_path: Path
    history_path: Path
    status: str
    regression_fail: bool
    payload: dict[str, Any]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values, dtype=float), 0.95))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.array(values, dtype=float)))


def _summary(samples: list[dict[str, float]], rate_key: str) -> dict[str, float]:
    durations = [float(sample["duration_sec"]) for sample in samples]
    rates = [float(sample[rate_key]) for sample in samples]
    return {
        "samples": int(len(samples)),
        "duration_median_sec": _median(durations),
        "duration_p95_sec": _p95(durations),
        f"{rate_key}_median": _median(rates),
        f"{rate_key}_p95": _p95(rates),
    }


def _load_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw in history_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _append_history(history_path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(history_path.parent)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parse_benchmark(raw_trace: Path | None, samples: int) -> dict[str, Any]:
    if raw_trace is None:
        return {"status": "skipped", "reason": "raw_trace_not_provided"}

    sample_rows: list[dict[str, float]] = []
    for idx in range(samples):
        with tempfile.TemporaryDirectory(prefix="hpcopt_bench_parse_") as tmp:
            tmp_path = Path(tmp)
            start = time.perf_counter()
            result = ingest_swf(
                input_path=raw_trace,
                out_dir=tmp_path / "curated",
                dataset_id=f"bench_parse_{idx}",
                report_dir=tmp_path / "reports",
            )
            duration = max(time.perf_counter() - start, 1e-9)
            jobs_per_sec = float(result.row_count / duration)
            sample_rows.append(
                {
                    "duration_sec": float(duration),
                    "rows": float(result.row_count),
                    "jobs_per_sec": jobs_per_sec,
                }
            )

    return {
        "status": "ok",
        "samples": sample_rows,
        "summary": _summary(sample_rows, rate_key="jobs_per_sec"),
    }


def _simulation_benchmark(
    trace_df: pd.DataFrame,
    policy_id: str,
    capacity_cpus: int,
    samples: int,
) -> dict[str, Any]:
    sample_rows: list[dict[str, float]] = []
    for idx in range(samples):
        start = time.perf_counter()
        sim = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id=policy_id,
            capacity_cpus=capacity_cpus,
            run_id=f"bench_sim_{idx}",
            strict_invariants=True,
        )
        duration = max(time.perf_counter() - start, 1e-9)
        step_count = int(sim.invariant_report.get("step_count", 0))
        events_per_sec = float(step_count / duration)
        jobs_per_sec = float(len(sim.jobs_df) / duration)
        sample_rows.append(
            {
                "duration_sec": float(duration),
                "events": float(step_count),
                "jobs": float(len(sim.jobs_df)),
                "events_per_sec": events_per_sec,
                "jobs_per_sec": jobs_per_sec,
            }
        )

    events_summary = _summary(sample_rows, rate_key="events_per_sec")
    jobs_summary = _summary(sample_rows, rate_key="jobs_per_sec")
    return {
        "status": "ok",
        "samples": sample_rows,
        "events_summary": events_summary,
        "jobs_summary": jobs_summary,
    }


def _pipeline_benchmark(
    dataset_path: Path,
    samples: int,
) -> dict[str, Any]:
    sample_rows: list[dict[str, float]] = []
    for idx in range(samples):
        with tempfile.TemporaryDirectory(prefix="hpcopt_bench_pipeline_") as tmp:
            tmp_path = Path(tmp)
            start = time.perf_counter()
            profile = build_trace_profile(
                dataset_path=dataset_path,
                report_dir=tmp_path / "reports",
                dataset_id=f"bench_profile_{idx}",
            )
            features = build_feature_dataset(
                dataset_path=dataset_path,
                out_dir=tmp_path / "features",
                report_dir=tmp_path / "reports",
                dataset_id=f"bench_features_{idx}",
                n_folds=3,
                train_fraction=0.70,
                val_fraction=0.15,
            )
            train = train_runtime_quantile_models(
                dataset_path=dataset_path,
                out_dir=tmp_path / "models",
                model_id=f"bench_model_{idx}",
                seed=42 + idx,
            )
            duration = max(time.perf_counter() - start, 1e-9)
            rows = int(features.row_count)
            rows_per_sec = float(rows / duration)
            sample_rows.append(
                {
                    "duration_sec": float(duration),
                    "rows": float(rows),
                    "rows_per_sec": rows_per_sec,
                    "profile_path_exists": 1.0 if profile.profile_path.exists() else 0.0,
                    "model_metrics_exists": 1.0 if train.metrics_path.exists() else 0.0,
                }
            )

    return {
        "status": "ok",
        "samples": sample_rows,
        "summary": _summary(sample_rows, rate_key="rows_per_sec"),
    }


def _detect_workload_class(trace_df: pd.DataFrame) -> str:
    """Auto-detect workload class from trace profile characteristics."""
    runtime = trace_df["runtime_actual_sec"].fillna(0)
    p50 = float(runtime.quantile(0.5))
    p99 = float(runtime.quantile(0.99))
    tail_ratio = p99 / p50 if p50 > 0 else 1.0

    if tail_ratio > 50.0:
        return "heavy_tail"

    if "start_ts" in trace_df.columns and "submit_ts" in trace_df.columns:
        wait = (trace_df["start_ts"] - trace_df["submit_ts"]).clip(lower=0)
        mean_wait = float(wait.mean())
        if mean_wait < 10.0:
            return "low_congestion"

    return "balanced"


def _load_regression_thresholds(
    config_path: Path | None,
    workload_class: str | None,
) -> tuple[float, int]:
    """Load per-workload-class regression thresholds from config."""
    default_drop = 0.10
    default_window = 5
    if config_path is None or not config_path.exists():
        return default_drop, default_window

    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = cfg.get("default", {})
    drop = float(defaults.get("regression_max_drop", default_drop))
    window = int(defaults.get("history_window", default_window))

    if workload_class:
        per_class = cfg.get("per_workload_class", {}).get(workload_class, {})
        drop = float(per_class.get("regression_max_drop", drop))
        window = int(per_class.get("history_window", window))

    return drop, window


def run_benchmark_suite(
    trace_dataset: Path,
    report_path: Path,
    history_path: Path,
    raw_trace: Path | None = None,
    policy_id: str = "FIFO_STRICT",
    capacity_cpus: int = 64,
    samples: int = 3,
    regression_max_drop: float = 0.10,
    history_window: int = 5,
    workload_class: str | None = None,
    regression_config: Path | None = None,
) -> BenchmarkRunResult:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if capacity_cpus <= 0:
        raise ValueError("capacity_cpus must be > 0")
    if not (0.0 < regression_max_drop < 1.0):
        raise ValueError("regression_max_drop must be in (0,1)")
    if history_window < 1:
        raise ValueError("history_window must be >= 1")

    trace_df = pd.read_parquet(trace_dataset)

    # Auto-detect workload class if not provided
    detected_class = workload_class or _detect_workload_class(trace_df)
    if regression_config is not None:
        cfg_drop, cfg_window = _load_regression_thresholds(regression_config, detected_class)
        regression_max_drop = cfg_drop
        history_window = cfg_window

    parse_result = _parse_benchmark(raw_trace=raw_trace, samples=samples)
    simulation_result = _simulation_benchmark(
        trace_df=trace_df,
        policy_id=policy_id,
        capacity_cpus=capacity_cpus,
        samples=samples,
    )
    pipeline_result = _pipeline_benchmark(
        dataset_path=trace_dataset,
        samples=samples,
    )

    profile_id = f"{trace_dataset.name}:{policy_id}:cap{capacity_cpus}:samples{samples}"
    current_sim_rate = float(simulation_result["events_summary"]["events_per_sec_median"])
    history_records = _load_history(history_path)
    prior = [
        record
        for record in history_records
        if record.get("profile_id") == profile_id and record.get("status") == "pass"
    ]
    prior_tail = prior[-history_window:]
    prior_rates = [float(record["simulation_events_per_sec_median"]) for record in prior_tail]
    baseline_median = _median(prior_rates)

    regression_fail = False
    regression_details: dict[str, Any] = {
        "profile_id": profile_id,
        "history_window": history_window,
        "regression_max_drop": regression_max_drop,
        "prior_pass_records": int(len(prior)),
        "prior_window_records": int(len(prior_tail)),
        "baseline_median_events_per_sec": baseline_median,
        "current_events_per_sec": current_sim_rate,
        "allowed_floor_events_per_sec": (
            float(baseline_median * (1.0 - regression_max_drop)) if baseline_median > 0 else None
        ),
    }
    if baseline_median > 0:
        allowed_floor = baseline_median * (1.0 - regression_max_drop)
        regression_fail = current_sim_rate < allowed_floor
    regression_details["regression_fail"] = regression_fail

    status = "fail" if regression_fail else "pass"
    payload = {
        "run_id": report_path.stem.replace("_benchmark_report", ""),
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "status": status,
        "profile_id": profile_id,
        "workload_class": detected_class,
        "input_trace_dataset": str(trace_dataset),
        "raw_trace_for_parse_benchmark": str(raw_trace) if raw_trace is not None else None,
        "policy_id": policy_id,
        "capacity_cpus": int(capacity_cpus),
        "samples": int(samples),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "parse_benchmark": parse_result,
        "simulation_benchmark": simulation_result,
        "pipeline_benchmark": pipeline_result,
        "regression_gate": regression_details,
    }
    write_json(report_path, payload)

    history_entry = {
        "timestamp_utc": payload["timestamp_utc"],
        "profile_id": profile_id,
        "status": status,
        "simulation_events_per_sec_median": current_sim_rate,
    }
    _append_history(history_path=history_path, payload=history_entry)
    return BenchmarkRunResult(
        report_path=report_path,
        history_path=history_path,
        status=status,
        regression_fail=regression_fail,
        payload=payload,
    )
