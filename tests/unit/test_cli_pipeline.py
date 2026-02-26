"""CLI tests for profiling and feature pipeline commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from hpcopt.cli.main import app
from typer.testing import CliRunner


def test_profile_trace_cli(tmp_path: Path, stress_dataset) -> None:
    runner = CliRunner()
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "profile",
            "trace",
            "--dataset",
            str(stress_dataset.dataset_path),
            "--out",
            str(report_dir),
            "--dataset-id",
            "profile_cli_test",
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
            "features",
            "build",
            "--dataset",
            str(stress_dataset.dataset_path),
            "--out",
            str(out_dir),
            "--report-out",
            str(report_dir),
            "--dataset-id",
            "features_cli_test",
            "--n-folds",
            "2",
            "--train-fraction",
            "0.60",
            "--val-fraction",
            "0.20",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Feature dataset:" in result.output


def test_features_build_cli_rejects_invalid_split(tmp_path: Path) -> None:
    dataset = tmp_path / "trace.parquet"
    pd.DataFrame(
        [{"job_id": 1, "submit_ts": 1, "runtime_actual_sec": 1, "requested_cpus": 1}],
    ).to_parquet(dataset, index=False)

    result = CliRunner().invoke(
        app,
        [
            "features",
            "build",
            "--dataset",
            str(dataset),
            "--train-fraction",
            "0.80",
            "--val-fraction",
            "0.25",
        ],
    )
    assert result.exit_code != 0
    assert "must be < 1.0" in result.output


def test_profile_trace_cli_uses_default_dataset_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.pipeline as cli_pipeline

    dataset = tmp_path / "my_dataset.parquet"
    dataset.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        cli_pipeline,
        "build_trace_profile",
        lambda **_kwargs: SimpleNamespace(profile_path=tmp_path / "reports" / "profile.json", row_count=12),
    )
    monkeypatch.setattr(cli_pipeline, "build_manifest", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(cli_pipeline, "write_manifest", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        ["profile", "trace", "--dataset", str(dataset), "--out", str(tmp_path / "reports")],
    )
    assert result.exit_code == 0, result.output
    assert "Profile report:" in result.output


def test_analysis_sensitivity_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.analysis.sensitivity as sensitivity_mod
    import hpcopt.cli.pipeline as cli_pipeline

    trace = tmp_path / "trace.parquet"
    pd.DataFrame(
        [
            {"job_id": 1, "submit_ts": 1, "runtime_actual_sec": 10, "requested_cpus": 1},
            {"job_id": 2, "submit_ts": 2, "runtime_actual_sec": 11, "requested_cpus": 1},
        ],
    ).to_parquet(trace, index=False)

    monkeypatch.setattr(cli_pipeline, "resolve_runtime_model_dir", lambda _arg: tmp_path / "models" / "m1")
    monkeypatch.setattr(sensitivity_mod, "run_guard_k_sweep", lambda **_kwargs: [{"k": 0.1, "ok": True}])
    monkeypatch.setattr(
        sensitivity_mod,
        "build_sensitivity_report",
        lambda **_kwargs: SimpleNamespace(
            report_path=tmp_path / "reports" / "sensitivity.json",
            payload={"analysis": {"optimal_k": 0.1}},
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "analysis",
            "sensitivity-sweep",
            "--trace",
            str(trace),
            "--out",
            str(tmp_path / "reports"),
            "--k-values",
            "0.1,0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Sensitivity report:" in result.output
    assert "Optimal k: 0.1" in result.output


def test_analysis_feature_importance_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.analysis.feature_importance as fi_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = tmp_path / "trace.parquet"
    dataset.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        fi_mod,
        "build_importance_report",
        lambda **_kwargs: SimpleNamespace(report_path=tmp_path / "reports" / "importance.json"),
    )

    result = CliRunner().invoke(
        app,
        [
            "analysis",
            "feature-importance",
            "--model-dir",
            str(model_dir),
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Feature importance report:" in result.output


def test_credibility_run_suite_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.orchestrate.credibility as credibility_mod

    suite = SimpleNamespace(
        status="ok",
        per_trace=[
            SimpleNamespace(trace_id="trace_a", status="ok", fidelity_status="pass", recommendation_status="accepted"),
            SimpleNamespace(trace_id="trace_b", status="ok", fidelity_status="pass", recommendation_status="blocked"),
        ],
    )
    monkeypatch.setattr(credibility_mod, "run_suite_credibility", lambda **_kwargs: suite)

    result = CliRunner().invoke(
        app,
        [
            "credibility",
            "run-suite",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--raw-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "credibility"),
            "--reference-suite-config",
            str(tmp_path / "ref.yaml"),
            "--fidelity-config",
            str(tmp_path / "fidelity.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Suite status: ok" in result.output
    assert "trace_a: ok" in result.output
    assert "trace_b: ok" in result.output


def test_credibility_dossier_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.artifacts.credibility_dossier as dossier_mod

    monkeypatch.setattr(
        dossier_mod,
        "assemble_credibility_dossier",
        lambda **_kwargs: SimpleNamespace(
            json_path=tmp_path / "dossier.json",
            md_path=tmp_path / "dossier.md",
        ),
    )
    result = CliRunner().invoke(
        app,
        [
            "credibility",
            "dossier",
            "--input-dir",
            str(tmp_path / "cred"),
            "--out",
            str(tmp_path / "dossier"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dossier JSON:" in result.output
    assert "Dossier MD:" in result.output
