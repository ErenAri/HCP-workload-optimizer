"""CLI tests for model management commands."""

from __future__ import annotations

from pathlib import Path

from hpcopt.cli.main import app
from typer.testing import CliRunner


def test_model_list_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HPCOPT_REGISTRY_PATH", str(tmp_path / "empty_registry.jsonl"))
    runner = CliRunner()
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "No models registered" in result.output


def test_stress_gen_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stress",
            "gen",
            "--scenario",
            "heavy_tail",
            "--out",
            str(tmp_path),
            "--n-jobs",
            "200",
            "--seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Stress dataset:" in result.output
