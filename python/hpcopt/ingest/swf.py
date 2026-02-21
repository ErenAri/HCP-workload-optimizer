from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from hpcopt.utils.io import ensure_dir, sha256_path as _sha256_path, write_json

SWF_FIELDS = [
    "job_number",
    "submit_time",
    "wait_time",
    "run_time",
    "allocated_processors",
    "avg_cpu_time_used",
    "used_memory",
    "requested_processors",
    "requested_time",
    "requested_memory",
    "status",
    "user_id",
    "group_id",
    "executable_number",
    "queue_number",
    "partition_number",
    "preceding_job_number",
    "think_time_from_preceding_job",
]


@dataclass
class IngestResult:
    dataset_path: Path
    quality_report_path: Path
    dataset_metadata_path: Path
    row_count: int


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _to_number(token: str) -> float | None:
    token = token.strip()
    if token in {"", "-1"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _as_int(value: float | None) -> int | None:
    if value is None or math.isnan(value):
        return None
    return int(value)


def _iter_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "total_lines": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "parsed_rows": 0,
    }

    with _open_text(path) as handle:
        for raw_line in handle:
            stats["total_lines"] += 1
            line = raw_line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            if line.startswith(";") or line.startswith("#"):
                stats["comment_lines"] += 1
                continue

            tokens = line.split()
            if len(tokens) != len(SWF_FIELDS):
                stats["malformed_lines"] += 1
                continue

            raw = {field: _to_number(value) for field, value in zip(SWF_FIELDS, tokens)}
            submit_ts = _as_int(raw["submit_time"])
            wait_sec = _as_int(raw["wait_time"])
            runtime_actual_sec = _as_int(raw["run_time"])
            job_id = _as_int(raw["job_number"])
            allocated_cpus = _as_int(raw["allocated_processors"])
            requested_cpus = _as_int(raw["requested_processors"])

            if None in {submit_ts, wait_sec, runtime_actual_sec, job_id, allocated_cpus}:
                stats["malformed_lines"] += 1
                continue

            start_ts = submit_ts + max(wait_sec, 0)
            end_ts = start_ts + max(runtime_actual_sec, 0)

            row = {
                "job_id": job_id,
                "submit_ts": submit_ts,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "wait_sec": max(wait_sec, 0),
                "runtime_actual_sec": max(runtime_actual_sec, 0),
                "runtime_requested_sec": _as_int(raw["requested_time"]),
                "allocated_cpus": allocated_cpus,
                "requested_cpus": requested_cpus if requested_cpus is not None else allocated_cpus,
                "requested_mem": _as_int(raw["requested_memory"]),
                "status": _as_int(raw["status"]),
                "user_id": _as_int(raw["user_id"]),
                "group_id": _as_int(raw["group_id"]),
                "queue_id": _as_int(raw["queue_number"]),
                "partition_id": _as_int(raw["partition_number"]),
                "runtime_overrequest_ratio": None,
            }
            if row["runtime_requested_sec"] and row["runtime_actual_sec"] > 0:
                row["runtime_overrequest_ratio"] = (
                    row["runtime_requested_sec"] / row["runtime_actual_sec"]
                )

            rows.append(row)
            stats["parsed_rows"] += 1

    return rows, stats


def ingest_swf(input_path: Path, out_dir: Path, dataset_id: str, report_dir: Path) -> IngestResult:
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    rows, stats = _iter_rows(input_path)
    if not rows:
        raise ValueError("No parsable SWF rows were produced from input trace.")

    df = pd.DataFrame(rows)
    dataset_path = out_dir / f"{dataset_id}.parquet"
    df.to_parquet(dataset_path, index=False)

    quality_report = {
        "dataset_id": dataset_id,
        "input_path": str(input_path),
        "output_dataset_path": str(dataset_path),
        **stats,
        "requested_mem_null_rows": int(df["requested_mem"].isna().sum()),
        "requested_mem_null_rate": float(df["requested_mem"].isna().mean()),
        "requested_cpu_fallback_rows": int(
            (df["requested_cpus"] == df["allocated_cpus"]).sum()
        ),
        "runtime_requested_null_rows": int(df["runtime_requested_sec"].isna().sum()),
        "row_count": int(len(df)),
    }
    quality_report_path = report_dir / f"{dataset_id}_quality_report.json"
    write_json(quality_report_path, quality_report)

    dataset_metadata = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_path(dataset_path),
        "source_trace_path": str(input_path),
        "source_trace_filename": input_path.name,
        "source_trace_sha256": _sha256_path(input_path),
        "row_count": int(len(df)),
    }
    dataset_metadata_path = out_dir / f"{dataset_id}.metadata.json"
    write_json(dataset_metadata_path, dataset_metadata)

    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
