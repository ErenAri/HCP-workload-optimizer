"""CLI tests for training commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_train_runtime_cli_with_hyperparams_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.train as cli_train

    dataset = tmp_path / "train.parquet"
    dataset.write_text("x", encoding="utf-8")
    hp_config = tmp_path / "hp.yaml"
    hp_config.write_text("n_estimators: 64\nlearning_rate: 0.08\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model_dir=tmp_path / "models" / "m1",
            metrics_path=tmp_path / "models" / "m1" / "metrics.json",
            metadata_path=tmp_path / "models" / "m1" / "metadata.json",
        )

    monkeypatch.setattr(cli_train, "train_runtime_quantile_models", _fake_train)
    monkeypatch.setattr(cli_train, "build_manifest", lambda **_kwargs: {"run_id": "train-1"})
    monkeypatch.setattr(cli_train, "write_manifest", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "train",
            "runtime",
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "models"),
            "--report-out",
            str(tmp_path / "reports"),
            "--model-id",
            "runtime_x",
            "--seed",
            "9",
            "--hyperparams-config",
            str(hp_config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["model_id"] == "runtime_x"
    assert captured["seed"] == 9
    assert captured["hyperparams"] == {"n_estimators": 64, "learning_rate": 0.08}


def test_train_tune_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.models.tuning as tuning_mod

    dataset = tmp_path / "train.parquet"
    dataset.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        tuning_mod,
        "build_tuning_report",
        lambda **_kwargs: SimpleNamespace(
            best_params=SimpleNamespace(to_dict=lambda: {"max_depth": 4}),
            best_score=0.123456,
            report_path=tmp_path / "reports" / "tuning.json",
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "train",
            "tune",
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "reports"),
            "--quantile",
            "0.9",
            "--seed",
            "5",
            "--n-trials",
            "4",
            "--n-folds",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Best params:" in result.output
    assert "Best score:" in result.output
    assert "Tuning report:" in result.output


def test_train_resource_fit_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.models.resource_fit as resource_fit_mod

    dataset = tmp_path / "train.parquet"
    dataset.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        resource_fit_mod,
        "train_resource_fit_model",
        lambda **_kwargs: SimpleNamespace(
            model_dir=tmp_path / "models" / "rf1",
            metrics_path=tmp_path / "models" / "rf1" / "metrics.json",
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "train",
            "resource-fit",
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "models"),
            "--model-id",
            "rf_model",
            "--seed",
            "17",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Model dir:" in result.output
    assert "Metrics:" in result.output
