"""Tests for run report export (JSON + Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hpcopt.artifacts.report_export import (
    _validate_run_id,
    export_run_report,
)


def test_validate_run_id_valid() -> None:
    _validate_run_id("run_2024_001")
    _validate_run_id("stress-test-v2")
    _validate_run_id("abc123")


def test_validate_run_id_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_run_id("")


def test_validate_run_id_path_traversal() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _validate_run_id("../etc/passwd")
    with pytest.raises(ValueError, match="forbidden"):
        _validate_run_id("foo/bar")
    with pytest.raises(ValueError, match="forbidden"):
        _validate_run_id("foo\\bar")


def test_export_with_matching_files(tmp_path: Path) -> None:
    run_id = "test_run_001"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / f"{run_id}_quality.json").write_text(json.dumps({"quality": "ok"}), encoding="utf-8")
    sim_dir = tmp_path / "simulations"
    sim_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    out_dir = tmp_path / "export"
    result = export_run_report(
        run_id=run_id,
        out_dir=out_dir,
        report_dir=report_dir,
        simulation_dir=sim_dir,
        model_dir=model_dir,
    )
    assert result.json_path.exists()
    assert result.md_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert len(payload["artifacts"]["report_files"]) == 1


def test_export_no_matching_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "export"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    sim_dir = tmp_path / "simulations"
    sim_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    result = export_run_report(
        run_id="no_match",
        out_dir=out_dir,
        report_dir=report_dir,
        simulation_dir=sim_dir,
        model_dir=model_dir,
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert len(payload["artifacts"]["report_files"]) == 0


def test_markdown_output_contains_headers(tmp_path: Path) -> None:
    out_dir = tmp_path / "export"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    result = export_run_report(
        run_id="md_test",
        out_dir=out_dir,
        report_dir=report_dir,
        simulation_dir=tmp_path / "sim",
        model_dir=tmp_path / "mod",
    )
    md_content = result.md_path.read_text(encoding="utf-8")
    assert "# Run Export: md_test" in md_content
    assert "## Report Files" in md_content
