from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.ingest.swf import IngestResult
from hpcopt.utils.io import ensure_dir, write_json
from hpcopt.utils.io import sha256_path as _sha256_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PBS/Torque accounting log format
# ---------------------------------------------------------------------------
# Each record is a single line:
#   <timestamp>;<type>;<id>;<attributes>
#
# Type is one of: S (started), E (ended/exited), Q (queued), D (deleted), R (rerun), etc.
# Attributes is a space-separated list of key=value pairs.
# We primarily care about 'E' (exit/end) records which carry the full accounting data.
# ---------------------------------------------------------------------------

_PBS_LINE_RE = re.compile(
    r"^(?P<timestamp>[^;]+);(?P<record_type>[A-Z]);(?P<job_id>[^;]+);(?P<attrs>.*)$"
)

# PBS datetime format: MM/DD/YYYY HH:MM:SS
_PBS_DT_FORMATS = [
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M",
]

# Common PBS attribute names for resource/time fields
_ATTR_CTIME = "ctime"       # creation (submit) time
_ATTR_QTIME = "qtime"       # queue time
_ATTR_ETIME = "etime"       # eligible time
_ATTR_START = "start"       # start time
_ATTR_END = "end"           # end time
_ATTR_RESOURCES_USED_WALLTIME = "resources_used.walltime"
_ATTR_RESOURCES_USED_CPUT = "resources_used.cput"
_ATTR_RESOURCES_USED_MEM = "resources_used.mem"
_ATTR_RESOURCE_LIST_WALLTIME = "Resource_List.walltime"
_ATTR_RESOURCE_LIST_NCPUS = "Resource_List.ncpus"
_ATTR_RESOURCE_LIST_NODES = "Resource_List.nodes"
_ATTR_RESOURCE_LIST_MEM = "Resource_List.mem"
_ATTR_USER = "user"
_ATTR_GROUP = "group"
_ATTR_QUEUE = "queue"
_ATTR_EXIT_STATUS = "Exit_status"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_pbs_timestamp(value: str) -> int | None:
    """Parse a PBS timestamp into a Unix epoch (UTC).

    PBS can store timestamps as either formatted strings or raw epoch integers.
    """
    value = value.strip()
    if not value:
        return None

    # Try raw epoch integer first.
    try:
        ts = int(value)
        if ts > 0:
            return ts
    except ValueError:
        pass

    for fmt in _PBS_DT_FORMATS:
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            continue
    return None


def _parse_walltime(value: str) -> int | None:
    """Convert a PBS walltime string (``HH:MM:SS`` or ``DD:HH:MM:SS``) to seconds."""
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
        if len(parts) == 4:
            d, h, m, s = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return d * 86400 + h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return m * 60 + s
    except ValueError:
        pass
    return None


def _parse_mem_kb(value: str) -> int | None:
    """Parse a PBS memory value (e.g. ``1234kb``, ``4gb``) into megabytes."""
    value = value.strip().lower()
    if not value:
        return None

    multiplier = 1.0 / 1024  # default: assume bytes -> MB
    if value.endswith("kb"):
        multiplier = 1.0 / 1024
        value = value[:-2]
    elif value.endswith("mb"):
        multiplier = 1.0
        value = value[:-2]
    elif value.endswith("gb"):
        multiplier = 1024.0
        value = value[:-2]
    elif value.endswith("tb"):
        multiplier = 1024.0 * 1024
        value = value[:-2]
    elif value.endswith("b"):
        multiplier = 1.0 / (1024 * 1024)
        value = value[:-1]

    try:
        return int(float(value) * multiplier)
    except ValueError:
        return None


def _parse_ncpus_from_nodes(value: str) -> int | None:
    """Extract CPU count from PBS ``nodes`` spec.

    Formats: ``1:ppn=8``, ``2:ppn=4``, ``nodes=1:ppn=4``, ``1``.
    """
    value = value.strip()
    if not value:
        return None

    # Strip leading ``nodes=`` if present.
    if value.lower().startswith("nodes="):
        value = value[6:]

    total_cpus = 0
    for chunk in value.split("+"):
        chunk = chunk.strip()
        node_count = 1
        ppn = 1

        parts = chunk.split(":")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("ppn="):
                try:
                    ppn = int(part[4:])
                except ValueError:
                    pass
            else:
                try:
                    node_count = int(part)
                except ValueError:
                    pass
        total_cpus += node_count * ppn

    return total_cpus if total_cpus > 0 else None


def _parse_attrs(raw: str) -> dict[str, str]:
    """Parse space-separated ``key=value`` PBS attribute string.

    Values can themselves contain spaces in rare cases but the standard
    accounting log uses spaces only as a delimiter.
    """
    attrs: dict[str, str] = {}
    for token in raw.split():
        if "=" in token:
            key, _, val = token.partition("=")
            attrs[key.strip()] = val.strip()
    return attrs


def _canonical_job_id(raw_id: str) -> str:
    """Normalise PBS job ID (strip server suffix).

    ``12345.pbs-server`` -> ``12345``
    ``12345[0].pbs-server`` -> ``12345_0``  (array job)
    """
    # Strip server suffix
    base = raw_id.split(".")[0] if "." in raw_id else raw_id

    # Convert array bracket notation to underscore.
    base = base.replace("[", "_").replace("]", "")
    return base.strip()


# ---------------------------------------------------------------------------
# Row-level parsing
# ---------------------------------------------------------------------------


def _iter_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a PBS/Torque accounting log file.

    Only ``E`` (exit) records are kept because they contain the full set of
    accounting attributes.
    """
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "total_lines": 0,
        "blank_lines": 0,
        "non_exit_records": 0,
        "malformed_lines": 0,
        "parsed_rows": 0,
    }

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            stats["total_lines"] += 1
            line = raw_line.strip()
            if not line or line.startswith("#"):
                stats["blank_lines"] += 1
                continue

            m = _PBS_LINE_RE.match(line)
            if not m:
                stats["malformed_lines"] += 1
                continue

            record_type = m.group("record_type")
            if record_type != "E":
                stats["non_exit_records"] += 1
                continue

            record_ts_str = m.group("timestamp")
            raw_job_id = m.group("job_id")
            attrs = _parse_attrs(m.group("attrs"))

            # ----------------------------------------------------------
            # Timestamps
            # ----------------------------------------------------------
            submit_ts = _parse_pbs_timestamp(attrs.get(_ATTR_CTIME, ""))
            if submit_ts is None:
                submit_ts = _parse_pbs_timestamp(attrs.get(_ATTR_QTIME, ""))
            if submit_ts is None:
                submit_ts = _parse_pbs_timestamp(record_ts_str)
            if submit_ts is None:
                stats["malformed_lines"] += 1
                continue

            start_ts = _parse_pbs_timestamp(attrs.get(_ATTR_START, ""))
            end_ts = _parse_pbs_timestamp(attrs.get(_ATTR_END, ""))

            walltime_used = _parse_walltime(
                attrs.get(_ATTR_RESOURCES_USED_WALLTIME, "")
            )
            walltime_requested = _parse_walltime(
                attrs.get(_ATTR_RESOURCE_LIST_WALLTIME, "")
            )

            # Derive missing timestamps.
            if end_ts is None and start_ts is not None and walltime_used is not None:
                end_ts = start_ts + walltime_used
            if start_ts is None:
                start_ts = submit_ts
            if end_ts is None:
                if walltime_used is not None:
                    end_ts = start_ts + walltime_used
                else:
                    end_ts = start_ts

            runtime_actual_sec = (
                walltime_used if walltime_used is not None else max(end_ts - start_ts, 0)
            )
            wait_sec = max(start_ts - submit_ts, 0)

            # ----------------------------------------------------------
            # Resource fields
            # ----------------------------------------------------------
            ncpus: int | None = None
            ncpus_raw = attrs.get(_ATTR_RESOURCE_LIST_NCPUS, "")
            if ncpus_raw:
                try:
                    ncpus = int(ncpus_raw)
                except ValueError:
                    pass

            if ncpus is None:
                ncpus = _parse_ncpus_from_nodes(
                    attrs.get(_ATTR_RESOURCE_LIST_NODES, "")
                )
            if ncpus is None:
                ncpus = 1

            requested_mem = _parse_mem_kb(attrs.get(_ATTR_RESOURCE_LIST_MEM, ""))

            # ----------------------------------------------------------
            # Metadata
            # ----------------------------------------------------------
            user_id = attrs.get(_ATTR_USER) or None
            group_id = attrs.get(_ATTR_GROUP) or None
            queue_id = attrs.get(_ATTR_QUEUE) or None
            exit_status = attrs.get(_ATTR_EXIT_STATUS) or None

            runtime_overrequest_ratio: float | None = None
            if walltime_requested and runtime_actual_sec and runtime_actual_sec > 0:
                runtime_overrequest_ratio = walltime_requested / runtime_actual_sec

            row: dict[str, Any] = {
                "job_id": _canonical_job_id(raw_job_id),
                "submit_ts": submit_ts,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "wait_sec": wait_sec,
                "runtime_actual_sec": runtime_actual_sec,
                "runtime_requested_sec": walltime_requested,
                "allocated_cpus": ncpus,
                "requested_cpus": ncpus,
                "requested_mem": requested_mem,
                "status": exit_status,
                "user_id": user_id,
                "group_id": group_id,
                "queue_id": queue_id,
                "partition_id": queue_id,
                "runtime_overrequest_ratio": runtime_overrequest_ratio,
            }
            rows.append(row)
            stats["parsed_rows"] += 1

    return rows, stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_pbs(
    input_path: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    report_dir: Path | str,
) -> IngestResult:
    """Ingest a PBS/Torque accounting log into canonical parquet.

    Parameters
    ----------
    input_path:
        Path to the PBS accounting log file.
    out_dir:
        Directory where the output parquet will be written.
    dataset_id:
        Identifier used to name output artifacts.
    report_dir:
        Directory for quality-report JSON.

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

    rows, stats = _iter_rows(input_path)
    if not rows:
        raise ValueError(
            f"No parsable PBS records were produced from input file: {input_path}"
        )

    df = pd.DataFrame(rows)
    dataset_path = out_dir / f"{dataset_id}.parquet"
    df.to_parquet(dataset_path, index=False)

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------
    quality_report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "input_path": str(input_path),
        "output_dataset_path": str(dataset_path),
        **stats,
        "requested_mem_null_rows": int(df["requested_mem"].isna().sum()),
        "requested_mem_null_rate": float(df["requested_mem"].isna().mean()),
        "runtime_requested_null_rows": int(df["runtime_requested_sec"].isna().sum()),
        "row_count": int(len(df)),
    }
    quality_report_path = report_dir / f"{dataset_id}_quality_report.json"
    write_json(quality_report_path, quality_report)

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------
    dataset_metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_path(dataset_path),
        "source_trace_path": str(input_path),
        "source_trace_filename": input_path.name,
        "source_trace_sha256": _sha256_path(input_path),
        "source_format": "pbs_torque_accounting",
        "row_count": int(len(df)),
    }
    dataset_metadata_path = out_dir / f"{dataset_id}.metadata.json"
    write_json(dataset_metadata_path, dataset_metadata)

    logger.info(
        "PBS ingest complete: %d rows written to %s", len(df), dataset_path
    )
    return IngestResult(
        dataset_path=dataset_path,
        quality_report_path=quality_report_path,
        dataset_metadata_path=dataset_metadata_path,
        row_count=int(len(df)),
    )
