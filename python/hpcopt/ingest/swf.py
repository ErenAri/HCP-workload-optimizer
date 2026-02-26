from __future__ import annotations

import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.ingest import finalize_ingest
from hpcopt.utils.io import ensure_dir

MAX_INPUT_FILE_BYTES = 2 * 1024**3  # 2 GB
MAX_LINE_LENGTH = 1_000_000
MAX_ROWS = 50_000_000

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


def _open_text(path: Path) -> Any:
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
            if len(raw_line) > MAX_LINE_LENGTH:
                stats["malformed_lines"] += 1
                continue
            line = raw_line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            if line.startswith(";") or line.startswith("#"):
                stats["comment_lines"] += 1
                continue

            if stats["parsed_rows"] >= MAX_ROWS:
                break

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

            if (
                submit_ts is None
                or wait_sec is None
                or runtime_actual_sec is None
                or job_id is None
                or allocated_cpus is None
            ):
                stats["malformed_lines"] += 1
                continue

            submit_ts_i = int(submit_ts)
            wait_sec_i = max(int(wait_sec), 0)
            runtime_actual_sec_i = max(int(runtime_actual_sec), 0)
            job_id_i = int(job_id)
            allocated_cpus_i = int(allocated_cpus)
            requested_cpus_i = int(requested_cpus) if requested_cpus is not None else allocated_cpus_i

            start_ts = submit_ts_i + wait_sec_i
            end_ts = start_ts + runtime_actual_sec_i
            runtime_requested_sec = _as_int(raw["requested_time"])

            row: dict[str, Any] = {
                "job_id": job_id_i,
                "submit_ts": submit_ts_i,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "wait_sec": wait_sec_i,
                "runtime_actual_sec": runtime_actual_sec_i,
                "runtime_requested_sec": runtime_requested_sec,
                "allocated_cpus": allocated_cpus_i,
                "requested_cpus": requested_cpus_i,
                "requested_mem": _as_int(raw["requested_memory"]),
                "status": _as_int(raw["status"]),
                "user_id": _as_int(raw["user_id"]),
                "group_id": _as_int(raw["group_id"]),
                "queue_id": _as_int(raw["queue_number"]),
                "partition_id": _as_int(raw["partition_number"]),
                "runtime_overrequest_ratio": None,
            }
            if runtime_requested_sec is not None and runtime_actual_sec_i > 0:
                row["runtime_overrequest_ratio"] = runtime_requested_sec / runtime_actual_sec_i

            rows.append(row)
            stats["parsed_rows"] += 1

    return rows, stats


def ingest_swf(input_path: Path, out_dir: Path, dataset_id: str, report_dir: Path) -> IngestResult:
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    file_size = input_path.stat().st_size
    if file_size > MAX_INPUT_FILE_BYTES:
        raise ValueError(
            f"Input file too large ({file_size / (1024**3):.1f} GB). "
            f"Maximum allowed: {MAX_INPUT_FILE_BYTES / (1024**3):.0f} GB."
        )

    rows, stats = _iter_rows(input_path)
    if not rows:
        raise ValueError("No parsable SWF rows were produced from input trace.")

    df = pd.DataFrame(rows)
    extra_quality = {
        "requested_cpu_fallback_rows": int((df["requested_cpus"] == df["allocated_cpus"]).sum()),
    }
    dataset_path, quality_report_path, dataset_metadata_path = finalize_ingest(
        df=df,
        dataset_id=dataset_id,
        input_path=input_path,
        out_dir=out_dir,
        report_dir=report_dir,
        parse_stats=stats,
        source_format="swf",
        extra_quality_fields=extra_quality,
    )

    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
