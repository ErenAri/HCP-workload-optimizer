"""CLI tests for report and artifact commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


def _write_object(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_recommend_generate_cli_standard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.report as cli_report

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "cand.json"
    _write_object(baseline, {"ok": True})
    _write_object(candidate, {"ok": True})

    monkeypatch.setattr(
        cli_report,
        "generate_recommendation_report",
        lambda **_kwargs: SimpleNamespace(
            report_path=tmp_path / "reports" / "recommendation.json",
            payload={"status": "accepted"},
        ),
    )
    monkeypatch.setattr(cli_report, "build_manifest", lambda **_kwargs: {"run_id": "r1"})
    monkeypatch.setattr(cli_report, "write_manifest", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "generate",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--out",
            str(tmp_path / "reports"),
            "--run-id",
            "rec_std",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recommendation status: accepted" in result.output


def test_recommend_generate_cli_pareto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.report as cli_report
    import hpcopt.recommend.engine as recommend_engine

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "cand.json"
    _write_object(baseline, {"ok": True})
    _write_object(candidate, {"ok": True})

    monkeypatch.setattr(
        recommend_engine,
        "generate_pareto_recommendation",
        lambda **_kwargs: SimpleNamespace(
            report_path=tmp_path / "reports" / "pareto.json",
            payload={"status": "pareto"},
        ),
    )
    monkeypatch.setattr(cli_report, "build_manifest", lambda **_kwargs: {"run_id": "r2"})
    monkeypatch.setattr(cli_report, "write_manifest", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "generate",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--out",
            str(tmp_path / "reports"),
            "--run-id",
            "rec_pareto",
            "--pareto",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recommendation status: pareto" in result.output


def test_report_export_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.report as cli_report

    monkeypatch.setattr(
        cli_report,
        "export_run_report",
        lambda **_kwargs: SimpleNamespace(
            json_path=tmp_path / "run.json",
            md_path=tmp_path / "run.md",
        ),
    )

    runner = CliRunner()
    both = runner.invoke(app, ["report", "export", "--run-id", "run-1", "--out", str(tmp_path), "--format", "both"])
    md_only = runner.invoke(app, ["report", "export", "--run-id", "run-2", "--out", str(tmp_path), "--format", "md"])
    invalid = runner.invoke(
        app,
        ["report", "export", "--run-id", "run-3", "--out", str(tmp_path), "--format", "yaml"],
    )

    assert both.exit_code == 0, both.output
    assert "Export json:" in both.output and "Export md:" in both.output
    assert md_only.exit_code == 0, md_only.output
    assert "Export md:" in md_only.output and "Export json:" not in md_only.output
    assert invalid.exit_code != 0
    assert "format must be one of: json|md|both" in invalid.output


def test_report_benchmark_strict_regression_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.report as cli_report

    trace = tmp_path / "trace.parquet"
    trace.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        cli_report,
        "run_benchmark_suite",
        lambda **_kwargs: SimpleNamespace(
            status="ok",
            regression_fail=True,
            report_path=tmp_path / "reports" / "bench.json",
            history_path=tmp_path / "reports" / "history.jsonl",
        ),
    )
    monkeypatch.setattr(cli_report, "build_manifest", lambda **_kwargs: {"run_id": "b1"})
    monkeypatch.setattr(cli_report, "write_manifest", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "report",
            "benchmark",
            "--trace",
            str(trace),
            "--policy",
            "FIFO_STRICT",
            "--out",
            str(tmp_path / "reports"),
            "--history",
            str(tmp_path / "reports" / "history.jsonl"),
            "--strict-regression",
        ],
    )
    assert result.exit_code == 1
    assert "Benchmark status: ok" in result.output


def test_report_benchmark_invalid_policy(tmp_path: Path) -> None:
    trace = tmp_path / "trace.parquet"
    trace.write_text("x", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["report", "benchmark", "--trace", str(trace), "--policy", "NOT_A_POLICY"],
    )
    assert result.exit_code != 0
    assert "Unsupported policy" in result.output


def test_artifacts_cleanup_preview_and_real(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.artifacts.retention as retention_mod

    monkeypatch.setattr(
        retention_mod,
        "cleanup_artifacts",
        lambda **kwargs: {
            "summary": {
                "would_delete_count": 4,
                "deleted_count": 2,
            },
            "dry_run": kwargs.get("dry_run", True),
        },
    )
    runner = CliRunner()
    preview = runner.invoke(app, ["artifacts", "cleanup", "--outputs-dir", str(tmp_path)])
    real = runner.invoke(app, ["artifacts", "cleanup", "--outputs-dir", str(tmp_path), "--no-dry-run"])

    assert preview.exit_code == 0, preview.output
    assert "Cleanup preview: 4 artifacts" in preview.output
    assert real.exit_code == 0, real.output
    assert "Cleanup completed: 2 artifacts" in real.output
