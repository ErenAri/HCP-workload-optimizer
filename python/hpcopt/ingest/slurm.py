from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.ingest import finalize_ingest
from hpcopt.utils.io import ensure_dir

logger = logging.getLogger(__name__)

MAX_INPUT_FILE_BYTES = 2 * 1024**3  # 2 GB
MAX_LINE_LENGTH = 1_000_000
MAX_ROWS = 50_000_000

# ---------------------------------------------------------------------------
# Re-use the canonical IngestResult from the SWF module
# ---------------------------------------------------------------------------

from hpcopt.ingest.swf import IngestResult  # noqa: E402

# ---------------------------------------------------------------------------
# Field map: sacct --parsable2 column names -> internal names
# ---------------------------------------------------------------------------

SACCT_COLUMNS = [
    "JobID",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "AllocCPUS",
    "ReqCPUS",
    "ReqMem",
    "User",
    "Group",
    "Partition",
    "State",
]

# Regex patterns for JobID variants
_ARRAY_JOB_RE = re.compile(r"^(\d+)_(\d+)$")      # 12345_0
_JOB_STEP_RE = re.compile(r"^(\d+)\.(.+)$")        # 12345.batch, 12345.0
_ARRAY_STEP_RE = re.compile(r"^(\d+)_(\d+)\.(.+)$")  # 12345_0.batch
_PLAIN_JOB_RE = re.compile(r"^(\d+)$")              # 12345

# Slurm date format (may include 'T' separator)
_SLURM_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_slurm_datetime(value: str) -> int | None:
    """Parse a Slurm datetime string into a Unix timestamp (UTC).

    Returns ``None`` for sentinel values such as ``Unknown`` or ``None``.
    """
    value = value.strip()
    if not value or value.lower() in {"unknown", "none", "n/a", ""}:
        return None

    for fmt in _SLURM_DT_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            continue
    logger.debug("Unparseable Slurm datetime: '%s'", value)
    return None


def _parse_elapsed(value: str) -> int | None:
    """Convert Slurm ``Elapsed`` field (``[DD-]HH:MM:SS``) to seconds."""
    value = value.strip()
    if not value or value.lower() in {"unknown", "none", "n/a"}:
        return None

    days = 0
    if "-" in value:
        day_part, time_part = value.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    else:
        time_part = value

    parts = time_part.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            hours, minutes, seconds = 0, 0, int(parts[0])
        else:
            return None
    except ValueError:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_reqmem(value: str) -> int | None:
    """Parse ``ReqMem`` field (e.g. ``4000Mc``, ``8Gn``) into megabytes.

    The trailing ``c`` means per-CPU, ``n`` means per-node.  We report the
    raw numeric value in MB without multiplying by CPU/node count because
    canonical schema stores the request as stated.
    """
    value = value.strip()
    if not value or value == "0" or value.lower() in {"unknown", "none", "n/a"}:
        return None

    # Strip per-cpu / per-node suffix
    unit_suffix = value[-1] if value[-1] in ("c", "n") else ""
    numeric_part = value[: -1] if unit_suffix else value

    multiplier: float = 1.0  # assume megabytes
    if numeric_part.endswith("G") or numeric_part.endswith("g"):
        multiplier = 1024
        numeric_part = numeric_part[:-1]
    elif numeric_part.endswith("M") or numeric_part.endswith("m"):
        multiplier = 1
        numeric_part = numeric_part[:-1]
    elif numeric_part.endswith("K") or numeric_part.endswith("k"):
        multiplier = 1.0 / 1024
        numeric_part = numeric_part[:-1]
    elif numeric_part.endswith("T") or numeric_part.endswith("t"):
        multiplier = 1024 * 1024
        numeric_part = numeric_part[:-1]

    try:
        return int(float(numeric_part) * multiplier)
    except ValueError:
        return None


def _classify_job_id(raw_id: str) -> tuple[str, str | None, bool]:
    """Return (canonical_job_id, array_index_or_none, is_job_step).

    Job steps (e.g. ``12345.batch``) are flagged so we can optionally skip
    them and keep only allocation-level records.
    """
    raw_id = raw_id.strip()

    m = _ARRAY_STEP_RE.match(raw_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}", m.group(2), True

    m = _JOB_STEP_RE.match(raw_id)
    if m:
        return m.group(1), None, True

    m = _ARRAY_JOB_RE.match(raw_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}", m.group(2), False

    m = _PLAIN_JOB_RE.match(raw_id)
    if m:
        return m.group(1), None, False

    # Fallback: use the raw string as-is.
    return raw_id, None, False


def _parse_state(value: str) -> str:
    """Normalise Slurm job state to an upper-case root state.

    Slurm states can include qualifiers (e.g. ``CANCELLED by 1000``).
    """
    value = value.strip().upper()
    # Take root word only.
    return value.split()[0] if value else "UNKNOWN"


# ---------------------------------------------------------------------------
# Row-level parsing
# ---------------------------------------------------------------------------


def _iter_rows(
    path: Path,
    skip_job_steps: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse sacct ``--parsable2`` (pipe-delimited) output.

    Parameters
    ----------
    path:
        Input file path.
    skip_job_steps:
        If ``True``, rows for job steps (``JobID.step``) are dropped.

    Returns
    -------
    tuple
        (rows, parse_stats)
    """
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "total_lines": 0,
        "header_lines": 0,
        "blank_lines": 0,
        "step_lines_skipped": 0,
        "malformed_lines": 0,
        "parsed_rows": 0,
    }

    with path.open("r", encoding="utf-8", errors="strict") as fh:
        header_seen = False
        col_indices: dict[str, int] = {}

        for raw_line in fh:
            stats["total_lines"] += 1
            if len(raw_line) > MAX_LINE_LENGTH:
                stats["malformed_lines"] += 1
                continue
            line = raw_line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue

            if stats["parsed_rows"] >= MAX_ROWS:
                break

            tokens = line.split("|")

            # Auto-detect the header row.
            if not header_seen:
                if "JobID" in tokens or "jobid" in [t.lower() for t in tokens]:
                    col_indices = {
                        col.strip(): idx for idx, col in enumerate(tokens)
                    }
                    header_seen = True
                    stats["header_lines"] += 1
                    continue
                # If no header, assume default column order.
                if not col_indices:
                    col_indices = {col: idx for idx, col in enumerate(SACCT_COLUMNS)}

            if len(tokens) < len(col_indices):
                stats["malformed_lines"] += 1
                continue

            def _col(name: str) -> str:
                idx = col_indices.get(name)
                if idx is None or idx >= len(tokens):
                    return ""
                return tokens[idx].strip()

            # ----------------------------------------------------------
            # JobID classification
            # ----------------------------------------------------------
            raw_job_id = _col("JobID")
            if not raw_job_id:
                stats["malformed_lines"] += 1
                continue

            canonical_id, array_index, is_step = _classify_job_id(raw_job_id)
            if is_step and skip_job_steps:
                stats["step_lines_skipped"] += 1
                continue

            # ----------------------------------------------------------
            # Timestamps
            # ----------------------------------------------------------
            submit_ts = _parse_slurm_datetime(_col("Submit"))
            start_ts = _parse_slurm_datetime(_col("Start"))
            end_ts = _parse_slurm_datetime(_col("End"))

            elapsed_sec = _parse_elapsed(_col("Elapsed"))

            # Derive missing end_ts from start + elapsed.
            if end_ts is None and start_ts is not None and elapsed_sec is not None:
                end_ts = start_ts + elapsed_sec

            # Derive runtime: prefer elapsed, then end-start.
            if elapsed_sec is not None:
                runtime_actual_sec = max(elapsed_sec, 0)
            elif start_ts is not None and end_ts is not None:
                runtime_actual_sec = max(end_ts - start_ts, 0)
            else:
                runtime_actual_sec = None

            if submit_ts is None:
                stats["malformed_lines"] += 1
                continue

            # If start is unknown, fall back to submit.
            if start_ts is None:
                start_ts = submit_ts
            if end_ts is None and runtime_actual_sec is not None:
                end_ts = start_ts + runtime_actual_sec
            if end_ts is None:
                end_ts = start_ts
            if runtime_actual_sec is None:
                runtime_actual_sec = max(end_ts - start_ts, 0)

            wait_sec = max(start_ts - submit_ts, 0)

            # ----------------------------------------------------------
            # Resource fields
            # ----------------------------------------------------------
            try:
                alloc_cpus = int(_col("AllocCPUS")) if _col("AllocCPUS") else None
            except ValueError:
                alloc_cpus = None

            try:
                req_cpus = int(_col("ReqCPUS")) if _col("ReqCPUS") else None
            except ValueError:
                req_cpus = None

            requested_mem = _parse_reqmem(_col("ReqMem"))

            # Use ReqCPUS, fall back to AllocCPUS.
            cpus = req_cpus if req_cpus is not None else alloc_cpus
            if cpus is None:
                cpus = 1  # absolute fallback

            allocated_cpus = alloc_cpus if alloc_cpus is not None else cpus

            # ----------------------------------------------------------
            # Metadata fields
            # ----------------------------------------------------------
            user_raw = _col("User")
            group_raw = _col("Group")
            partition_raw = _col("Partition")
            state = _parse_state(_col("State"))

            row: dict[str, Any] = {
                "job_id": canonical_id,
                "submit_ts": submit_ts,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "wait_sec": wait_sec,
                "runtime_actual_sec": runtime_actual_sec,
                "runtime_requested_sec": None,  # sacct does not expose Timelimit by default
                "allocated_cpus": allocated_cpus,
                "requested_cpus": cpus,
                "requested_mem": requested_mem,
                "status": state,
                "user_id": user_raw if user_raw else None,
                "group_id": group_raw if group_raw else None,
                "queue_id": partition_raw if partition_raw else None,
                "partition_id": partition_raw if partition_raw else None,
                "array_index": array_index,
                "runtime_overrequest_ratio": None,
            }
            rows.append(row)
            stats["parsed_rows"] += 1

    return rows, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_slurm(
    input_path: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    report_dir: Path | str,
    skip_job_steps: bool = True,
) -> IngestResult:
    """Ingest a ``sacct --parsable2`` dump into canonical parquet.

    Parameters
    ----------
    input_path:
        Path to the pipe-delimited sacct output file.
    out_dir:
        Directory where the output parquet will be written.
    dataset_id:
        Identifier used to name output artifacts.
    report_dir:
        Directory for quality-report JSON.
    skip_job_steps:
        Whether to skip job-step rows (default ``True``).

    Returns
    -------
    IngestResult
        Canonical result with dataset path, quality report path, and row count.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    report_dir = Path(report_dir)
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    file_size = input_path.stat().st_size
    if file_size > MAX_INPUT_FILE_BYTES:
        raise ValueError(
            f"Input file too large ({file_size / (1024**3):.1f} GB). "
            f"Maximum allowed: {MAX_INPUT_FILE_BYTES / (1024**3):.0f} GB."
        )

    rows, stats = _iter_rows(input_path, skip_job_steps=skip_job_steps)
    if not rows:
        raise ValueError(
            f"No parsable Slurm rows were produced from input file: {input_path}"
        )

    df = pd.DataFrame(rows)

    # Drop the array_index helper column before writing canonical parquet.
    if "array_index" in df.columns:
        df = df.drop(columns=["array_index"])

    dataset_path, quality_report_path, dataset_metadata_path = finalize_ingest(
        df=df,
        dataset_id=dataset_id,
        input_path=input_path,
        out_dir=out_dir,
        report_dir=report_dir,
        parse_stats=stats,
        source_format="slurm_sacct_parsable2",
    )

    logger.info(
        "Slurm ingest complete: %d rows written to %s", len(df), dataset_path
    )
    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
