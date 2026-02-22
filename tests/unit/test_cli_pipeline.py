"""CLI tests for profiling and feature pipeline commands."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hpcopt.cli.main import app


def test_profile_trace_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "profile", "trace",
            "--dataset", str(stress_dataset.dataset_path),
            "--out", str(report_dir),
            "--dataset-id", "profile_cli_test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Profile report:" in result.output
    assert "Rows:" in result.output


def test_features_build_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "features"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "features", "build",
            "--dataset", str(stress_dataset.dataset_path),
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--dataset-id", "features_cli_test",
            "--n-folds", "2",
            "--train-fraction", "0.60",
            "--val-fraction", "0.20",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Feature dataset:" in result.output
