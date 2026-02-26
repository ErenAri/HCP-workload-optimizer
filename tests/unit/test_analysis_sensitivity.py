"""Tests for the sensitivity sweep analysis module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from hpcopt.analysis.sensitivity import (
    SensitivitySweepResult,
    build_sensitivity_report,
    run_guard_k_sweep,
)


def test_build_sensitivity_report_from_sweep(tmp_path: Path) -> None:
    """Test building a report from pre-built sweep results."""
    sweep_results = {
        "baseline": {
            "policy_id": "EASY_BACKFILL_BASELINE",
            "objective_metrics": {"p95_bsld": 2.5, "utilization_cpu": 0.7},
        },
        "sweep": [
            {
                "guard_k": 0.5,
                "p95_bsld": 2.0,
                "utilization_cpu": 0.72,
                "delta_p95_bsld": 0.5,
                "delta_utilization": 0.02,
                "constraints_passed": True,
                "violations": [],
                "status": "ok",
            },
            {
                "guard_k": 1.0,
                "p95_bsld": 1.8,
                "utilization_cpu": 0.68,
                "delta_p95_bsld": 0.7,
                "delta_utilization": -0.02,
                "constraints_passed": True,
                "violations": [],
                "status": "ok",
            },
            {
                "guard_k": 1.5,
                "status": "error",
                "error": "diverged",
            },
        ],
    }
    out_path = tmp_path / "sensitivity_report.json"
    result = build_sensitivity_report(sweep_results, out_path)

    assert isinstance(result, SensitivitySweepResult)
    assert result.report_path.exists()
    assert isinstance(result.metrics_df, pd.DataFrame)
    assert len(result.metrics_df) == 2  # only "ok" rows
    assert result.payload["analysis"]["optimal_k"] == 1.0  # best delta
    assert result.payload["analysis"]["k_values_tested"] == 3
    assert result.payload["analysis"]["k_values_passed_constraints"] == 2


def test_build_sensitivity_report_no_ok_rows(tmp_path: Path) -> None:
    """Test when all sweep rows are errors."""
    sweep_results = {
        "baseline": {"policy_id": "EASY", "objective_metrics": {}},
        "sweep": [
            {"guard_k": 0.5, "status": "error", "error": "fail"},
        ],
    }
    out_path = tmp_path / "report.json"
    result = build_sensitivity_report(sweep_results, out_path)
    assert result.payload["analysis"]["optimal_k"] is None
    assert len(result.metrics_df) == 0


def test_run_guard_k_sweep_success_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import hpcopt.analysis.sensitivity as sensitivity_mod

    trace_df = pd.DataFrame(
        [
            {"job_id": 1, "submit_ts": 1, "runtime_actual_sec": 10, "requested_cpus": 1},
            {"job_id": 2, "submit_ts": 2, "runtime_actual_sec": 11, "requested_cpus": 2},
        ]
    )

    def _fake_run_simulation_from_trace(**kwargs):
        run_id = kwargs["run_id"]
        if run_id == "sensitivity_baseline":
            return SimpleNamespace(
                objective_metrics={
                    "p95_bsld": 10.0,
                    "utilization_cpu": 0.50,
                    "fairness_dev": 0.01,
                    "jain": 0.90,
                    "starved_rate": 0.01,
                    "p95_wait_sec": 100.0,
                },
                fallback_accounting={},
            )
        if abs(float(kwargs["runtime_guard_k"]) - 0.2) < 1e-9:
            raise ValueError("sim failed")
        return SimpleNamespace(
            objective_metrics={
                "p95_bsld": 8.0,
                "utilization_cpu": 0.55,
                "fairness_dev": 0.02,
                "jain": 0.89,
                "starved_rate": 0.01,
                "p95_wait_sec": 80.0,
            },
            fallback_accounting={"prediction_used_rate": 0.75},
        )

    monkeypatch.setattr(sensitivity_mod, "run_simulation_from_trace", _fake_run_simulation_from_trace)
    monkeypatch.setattr(
        sensitivity_mod,
        "evaluate_constraint_contract",
        lambda **_kwargs: {"constraints_passed": True, "violations": []},
    )

    payload = run_guard_k_sweep(
        trace_df=trace_df,
        capacity_cpus=16,
        k_values=[0.1, 0.2],
        model_dir=None,
        baseline_policy="EASY_BACKFILL_BASELINE",
        strict_invariants=True,
    )

    assert payload["baseline"]["policy_id"] == "EASY_BACKFILL_BASELINE"
    assert len(payload["sweep"]) == 2
    assert payload["sweep"][0]["status"] == "ok"
    assert payload["sweep"][0]["delta_p95_bsld"] == 2.0
    assert payload["sweep"][0]["prediction_used_rate"] == 0.75
    assert payload["sweep"][1]["status"] == "error"
    assert payload["sweep"][1]["error"] == "sim failed"


def test_run_guard_k_sweep_uses_predictor_when_model_dir_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hpcopt.analysis.sensitivity as sensitivity_mod

    trace_df = pd.DataFrame(
        [{"job_id": 1, "submit_ts": 1, "runtime_actual_sec": 10, "requested_cpus": 1}],
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    fake_predictor = object()
    captured_predictors: list[object | None] = []

    monkeypatch.setattr(sensitivity_mod, "RuntimeQuantilePredictor", lambda _path: fake_predictor)

    def _fake_run_simulation_from_trace(**kwargs):
        if kwargs["run_id"] == "sensitivity_baseline":
            captured_predictors.append(kwargs.get("runtime_predictor"))
            return SimpleNamespace(
                objective_metrics={
                    "p95_bsld": 10.0,
                    "utilization_cpu": 0.50,
                    "fairness_dev": 0.01,
                    "jain": 0.90,
                    "starved_rate": 0.01,
                    "p95_wait_sec": 100.0,
                },
                fallback_accounting={},
            )
        captured_predictors.append(kwargs.get("runtime_predictor"))
        return SimpleNamespace(
            objective_metrics={
                "p95_bsld": 9.0,
                "utilization_cpu": 0.60,
                "fairness_dev": 0.01,
                "jain": 0.90,
                "starved_rate": 0.01,
                "p95_wait_sec": 90.0,
            },
            fallback_accounting={"prediction_used_rate": 1.0},
        )

    monkeypatch.setattr(sensitivity_mod, "run_simulation_from_trace", _fake_run_simulation_from_trace)
    monkeypatch.setattr(
        sensitivity_mod,
        "evaluate_constraint_contract",
        lambda **_kwargs: {"constraints_passed": True, "violations": []},
    )

    payload = run_guard_k_sweep(
        trace_df=trace_df,
        capacity_cpus=8,
        k_values=[0.5],
        model_dir=model_dir,
    )

    assert payload["sweep"][0]["status"] == "ok"
    assert captured_predictors[0] is None  # baseline call
    assert captured_predictors[1] is fake_predictor
