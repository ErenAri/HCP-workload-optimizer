"""Tests for the shadow ingestion daemon."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpcopt.ingest.shadow import (
    PollResult,
    ShadowIngestionDaemon,
    WatermarkState,
    _load_watermark,
    _save_watermark,
)


def test_watermark_default() -> None:
    state = WatermarkState()
    assert state.last_processed_ts is None
    assert state.poll_count == 0
    assert state.rows_ingested_total == 0


def test_watermark_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "watermark.json"
    state = WatermarkState(
        last_processed_ts=1000,
        last_poll_utc="2024-01-01T00:00:00",
        rows_ingested_total=500,
        poll_count=5,
        source_type="slurm",
        source_path="/data/sacct.txt",
    )
    _save_watermark(path, state)
    loaded = _load_watermark(path)
    assert loaded.last_processed_ts == 1000
    assert loaded.poll_count == 5
    assert loaded.rows_ingested_total == 500


def test_watermark_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not valid json", encoding="utf-8")
    state = _load_watermark(path)
    assert state.last_processed_ts is None
    assert state.poll_count == 0


def test_watermark_missing_file(tmp_path: Path) -> None:
    state = _load_watermark(tmp_path / "nonexistent.json")
    assert state.last_processed_ts is None


def test_watermark_from_dict() -> None:
    data = {
        "last_processed_ts": 42,
        "poll_count": 3,
        "rows_ingested_total": 100,
    }
    state = WatermarkState.from_dict(data)
    assert state.last_processed_ts == 42
    assert state.poll_count == 3


def test_daemon_poll_once_slurm(tmp_path: Path) -> None:
    """Test a single poll cycle with Slurm source."""
    sacct_content = (
        "JobID|Submit|Start|End|Elapsed|AllocCPUS|ReqCPUS|ReqMem|User|Group|Partition|State\n"
        "1001|2024-01-01T00:00:00|2024-01-01T00:01:00|2024-01-01T00:11:00|00:10:00|4|4|4000Mc|alice|grp1|batch|COMPLETED\n"
    )
    source = tmp_path / "sacct.txt"
    source.write_text(sacct_content, encoding="utf-8")

    daemon = ShadowIngestionDaemon(
        out_dir=tmp_path / "out",
        report_dir=tmp_path / "rep",
        watermark_path=tmp_path / "wm.json",
    )
    daemon._source_type = "slurm"
    daemon._source_path = source
    result = daemon.poll_once()
    assert result.success is True
    assert result.rows_ingested >= 0


def test_daemon_unsupported_source_type(tmp_path: Path) -> None:
    daemon = ShadowIngestionDaemon(
        out_dir=tmp_path / "out",
        report_dir=tmp_path / "rep",
        watermark_path=tmp_path / "wm.json",
    )
    daemon._source_type = "unsupported"
    daemon._source_path = tmp_path / "fake"
    result = daemon.poll_once()
    assert result.success is False
    assert "Unsupported" in (result.error or "")


def test_daemon_stop(tmp_path: Path) -> None:
    daemon = ShadowIngestionDaemon(
        out_dir=tmp_path / "out",
        report_dir=tmp_path / "rep",
    )
    # Calling stop without start should not error
    daemon.stop()
