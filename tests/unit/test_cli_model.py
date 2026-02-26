"""CLI tests for model management commands."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
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


def test_model_list_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.models.registry as registry_mod

    class _FakeRegistry:
        def list(self) -> list[dict[str, str]]:
            return [{"model_id": "m-prod", "status": "production", "registered_at": "2026-02-26T00:00:00Z"}]

    monkeypatch.setattr(registry_mod, "ModelRegistry", _FakeRegistry)

    result = CliRunner().invoke(app, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "m-prod (production) [PRODUCTION]" in result.output


def test_model_promote_and_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.models.registry as registry_mod

    calls: dict[str, list[str]] = {"promote": [], "archive": []}

    class _FakeRegistry:
        def promote(self, model_id: str) -> None:
            calls["promote"].append(model_id)

        def archive(self, model_id: str) -> None:
            calls["archive"].append(model_id)

    monkeypatch.setattr(registry_mod, "ModelRegistry", _FakeRegistry)
    runner = CliRunner()

    promoted = runner.invoke(app, ["model", "promote", "--model-id", "m1"])
    archived = runner.invoke(app, ["model", "archive", "--model-id", "m2"])

    assert promoted.exit_code == 0, promoted.output
    assert archived.exit_code == 0, archived.output
    assert calls["promote"] == ["m1"]
    assert calls["archive"] == ["m2"]


def test_model_drift_check_no_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.model as cli_model

    eval_dataset = tmp_path / "eval.parquet"
    eval_dataset.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli_model, "resolve_runtime_model_dir", lambda _arg: None)

    result = CliRunner().invoke(
        app,
        ["model", "drift-check", "--eval-dataset", str(eval_dataset), "--out", str(tmp_path / "reports")],
    )
    assert result.exit_code != 0
    assert "No model directory found." in result.output


def test_model_drift_check_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.model as cli_model
    import hpcopt.models.drift as drift_mod

    eval_dataset = tmp_path / "eval.parquet"
    eval_dataset.write_text("x", encoding="utf-8")
    model_dir = tmp_path / "model_a"
    model_dir.mkdir()

    class _Report:
        overall_drift_detected = True

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"overall_drift_detected": True}

    monkeypatch.setattr(cli_model, "resolve_runtime_model_dir", lambda _arg: model_dir)
    monkeypatch.setattr(drift_mod, "compute_drift_report", lambda **_kwargs: _Report())

    result = CliRunner().invoke(
        app,
        ["model", "drift-check", "--eval-dataset", str(eval_dataset), "--out", str(tmp_path / "reports")],
    )
    assert result.exit_code == 0, result.output
    assert "Drift status: drift_detected" in result.output
    assert "Drift report:" in result.output


def test_lock_reference_suite_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.cli.model as cli_model

    config = tmp_path / "reference_suite.yaml"
    config.write_text("suite: []\n", encoding="utf-8")
    monkeypatch.setattr(
        cli_model,
        "lock_reference_suite_hashes",
        lambda **_kwargs: {"updated": 2, "missing_files": ["a", "b", "c"]},
    )

    result = CliRunner().invoke(
        app,
        [
            "data",
            "lock-reference-suite",
            "--config",
            str(config),
            "--raw-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "lock_report.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Suite lock updated: 2" in result.output
    assert "Missing files: 3" in result.output


def test_serve_api_cli_calls_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, int, bool]] = []

    def _fake_run(app_path: str, host: str, port: int, reload: bool) -> None:
        calls.append((app_path, host, port, reload))

    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = _fake_run  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)

    result = CliRunner().invoke(app, ["serve", "api", "--host", "0.0.0.0", "--port", "9999"])
    assert result.exit_code == 0, result.output
    assert calls == [("hpcopt.api.app:app", "0.0.0.0", 9999, False)]
