"""CLI tests for training commands."""

from __future__ import annotations

from pathlib import Path

from hpcopt.cli.main import app
from typer.testing import CliRunner


def test_train_runtime_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "train",
            "runtime",
            "--dataset",
            str(stress_dataset.dataset_path),
            "--out",
            str(model_dir),
            "--report-out",
            str(report_dir),
            "--model-id",
            "cli_test_model",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Model dir:" in result.output
    assert "Train manifest:" in result.output
