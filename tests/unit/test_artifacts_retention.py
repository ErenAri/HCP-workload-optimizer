"""Tests for the artifact retention/cleanup module."""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import pytest

from hpcopt.artifacts.retention import cleanup_artifacts


def _make_old_file(path: Path, days_old: int = 100) -> None:
    """Create a file and set its mtime to *days_old* days ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old artifact", encoding="utf-8")
    old_ts = time.time() - (days_old * 86400)
    os.utime(path, (old_ts, old_ts))


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _make_old_file(outputs / "old_report.json")
    result = cleanup_artifacts(outputs_dir=outputs, max_age_days=30, dry_run=True)
    assert (outputs / "old_report.json").exists()
    assert len(result.get("would_delete", [])) >= 1


def test_real_run_deletes(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _make_old_file(outputs / "stale.json")
    result = cleanup_artifacts(outputs_dir=outputs, max_age_days=30, dry_run=False)
    assert not (outputs / "stale.json").exists()
    assert len(result.get("deleted", [])) >= 1


def test_recent_files_not_deleted(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    recent = outputs / "recent.json"
    recent.write_text("fresh", encoding="utf-8")
    result = cleanup_artifacts(outputs_dir=outputs, max_age_days=30, dry_run=True)
    assert len(result.get("would_delete", [])) == 0


def test_nonexistent_directory(tmp_path: Path) -> None:
    result = cleanup_artifacts(outputs_dir=tmp_path / "nonexistent", max_age_days=30)
    assert result["summary"]["candidates_count"] == 0


def test_registry_file_protected(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    reg = outputs / "models" / "registry.jsonl"
    _make_old_file(reg, days_old=200)
    result = cleanup_artifacts(outputs_dir=outputs, max_age_days=30, dry_run=True)
    assert str(reg) in result.get("protected", []) or len(result.get("would_delete", [])) == 0
