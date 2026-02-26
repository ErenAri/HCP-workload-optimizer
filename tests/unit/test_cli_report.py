"""CLI tests for report and artifact commands."""

from __future__ import annotations

from pathlib import Path

from hpcopt.cli.main import app
from hpcopt.simulate.stress import generate_stress_scenario
from typer.testing import CliRunner


def test_report_benchmark_cli(tmp_path: Path) -> None:
    # Generate a small trace for benchmarking
    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path / "data",
        n_jobs=100,
        seed=5,
        params={"alpha": 1.2},
    )

    runner = CliRunner()
    out_dir = tmp_path / "reports"
    history_path = tmp_path / "bench_history.jsonl"
    result = runner.invoke(
        app,
        [
            "report",
            "benchmark",
            "--trace",
            str(stress.dataset_path),
            "--policy",
            "FIFO_STRICT",
            "--capacity-cpus",
            "64",
            "--samples",
            "1",
            "--out",
            str(out_dir),
            "--history",
            str(history_path),
            "--run-id",
            "bench_cli_test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Benchmark status:" in result.output
    assert "Benchmark report:" in result.output
