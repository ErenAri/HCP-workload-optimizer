"""Ingestion package.

Shared helpers for the SWF, Slurm, and PBS ingest modules live here to
avoid duplicating the quality-report / metadata / parquet-write pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.utils.io import ensure_dir, write_json
from hpcopt.utils.io import sha256_path as _sha256_path

logger = logging.getLogger(__name__)


def finalize_ingest(
    df: pd.DataFrame,
    dataset_id: str,
    input_path: Path,
    out_dir: Path,
    report_dir: Path,
    parse_stats: dict[str, Any],
    source_format: str | None = None,
    extra_quality_fields: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write canonical parquet, quality report, and dataset metadata.

    Returns ``(dataset_path, quality_report_path, dataset_metadata_path)``.
    """
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    dataset_path = out_dir / f"{dataset_id}.parquet"
    df.to_parquet(dataset_path, index=False)

    # -- Quality report ---------------------------------------------------
    quality_report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "input_path": str(input_path),
        "output_dataset_path": str(dataset_path),
        **parse_stats,
        "requested_mem_null_rows": int(df["requested_mem"].isna().sum()),
        "requested_mem_null_rate": float(df["requested_mem"].isna().mean()),
        "runtime_requested_null_rows": int(df["runtime_requested_sec"].isna().sum()),
        "row_count": int(len(df)),
    }
    if extra_quality_fields:
        quality_report.update(extra_quality_fields)
    quality_report_path = report_dir / f"{dataset_id}_quality_report.json"
    write_json(quality_report_path, quality_report)

    # -- Dataset metadata -------------------------------------------------
    dataset_metadata: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_path(dataset_path),
        "source_trace_path": str(input_path),
        "source_trace_filename": input_path.name,
        "source_trace_sha256": _sha256_path(input_path),
        "row_count": int(len(df)),
    }
    if source_format:
        dataset_metadata["source_format"] = source_format
    dataset_metadata_path = out_dir / f"{dataset_id}.metadata.json"
    write_json(dataset_metadata_path, dataset_metadata)

    return dataset_path, quality_report_path, dataset_metadata_path
