"""Tests for the sensitivity sweep analysis module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpcopt.analysis.sensitivity import (
    SensitivitySweepResult,
    build_sensitivity_report,
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
