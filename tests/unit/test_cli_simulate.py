"""CLI tests for simulation commands."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hpcopt.cli.main import app


def test_simulate_run_fifo_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "sim_out"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "simulate", "run",
            "--trace", str(stress_dataset.dataset_path),
            "--policy", "FIFO_STRICT",
            "--capacity-cpus", "64",
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--run-id", "sim_cli_test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Simulation report:" in result.output
    assert "Manifest:" in result.output


def test_simulate_replay_baselines_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "sim_out"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "simulate", "replay-baselines",
            "--trace", str(stress_dataset.dataset_path),
            "--capacity-cpus", "64",
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--run-id", "replay_cli_test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Baseline replay report:" in result.output
