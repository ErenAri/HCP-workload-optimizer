"""Tests for Rust simulator bridge wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from hpcopt.simulate import rust_bridge


def test_find_rust_binary_uses_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_bridge.shutil, "which", lambda _: "/usr/local/bin/sim-runner")
    assert rust_bridge.find_rust_binary() == Path("/usr/local/bin/sim-runner")


def test_find_rust_binary_uses_search_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_bridge.shutil, "which", lambda _: None)
    candidate = tmp_path / "sim-runner.exe"
    candidate.write_text("", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "_BINARY_SEARCH_PATHS", [candidate])
    assert rust_bridge.find_rust_binary() == candidate


def test_rust_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: None)
    assert rust_bridge.rust_available() is False


def test_rust_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: Path("/bin/sim-runner"))
    assert rust_bridge.rust_available() is True


def test_run_rust_simulation_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: None)
    with pytest.raises(FileNotFoundError, match="sim-runner binary not found"):
        rust_bridge.run_rust_simulation(trace_json_path=trace)


def test_run_rust_simulation_missing_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "sim-runner"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: binary)
    with pytest.raises(FileNotFoundError, match="Trace file not found"):
        rust_bridge.run_rust_simulation(trace_json_path=tmp_path / "missing.json")


def test_run_rust_simulation_process_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "sim-runner"
    trace = tmp_path / "trace.json"
    binary.write_text("", encoding="utf-8")
    trace.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: binary)

    def _fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(rust_bridge.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="sim-runner failed"):
        rust_bridge.run_rust_simulation(trace_json_path=trace)


def test_run_rust_simulation_temp_output_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "sim-runner"
    trace = tmp_path / "trace.json"
    binary.write_text("", encoding="utf-8")
    trace.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: binary)
    seen_output: dict[str, Path] = {}

    def _fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(json.dumps({"metrics": {"p95_bsld": 1.0}}), encoding="utf-8")
        seen_output["path"] = out_path
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rust_bridge.subprocess, "run", _fake_run)

    report = rust_bridge.run_rust_simulation(
        trace_json_path=trace,
        policy="EASY_BACKFILL_BASELINE",
        capacity_cpus=128,
        strict_invariants=True,
    )
    assert report["metrics"]["p95_bsld"] == 1.0
    assert "path" in seen_output
    assert seen_output["path"].exists() is False


def test_run_rust_simulation_with_explicit_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "sim-runner"
    trace = tmp_path / "trace.json"
    output = tmp_path / "result.json"
    binary.write_text("", encoding="utf-8")
    trace.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(rust_bridge, "find_rust_binary", lambda: binary)

    def _fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(json.dumps({"run_id": "abc"}), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rust_bridge.subprocess, "run", _fake_run)

    report = rust_bridge.run_rust_simulation(trace_json_path=trace, output_path=output)
    assert report == {"run_id": "abc"}
    assert output.exists()
