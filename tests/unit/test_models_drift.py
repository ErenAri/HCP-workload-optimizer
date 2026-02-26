"""Tests for the drift detection module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hpcopt.models.drift import (
    DEFAULT_METRIC_DEGRADATION_FACTOR,
    DEFAULT_N_BINS,
    DEFAULT_PSI_THRESHOLD,
    _compute_psi,
    _pinball_loss,
)


def test_psi_identical_distributions() -> None:
    """PSI should be ~0 for identical distributions."""
    rng = np.random.default_rng(42)
    data = rng.normal(100, 10, size=1000)
    psi, train_counts, eval_counts = _compute_psi(data, data, n_bins=10)
    assert psi < 0.01  # very close to 0


def test_psi_different_distributions() -> None:
    """PSI should be high for very different distributions."""
    rng = np.random.default_rng(42)
    train = rng.normal(100, 10, size=1000)
    evaluation = rng.normal(200, 10, size=1000)
    psi, _, _ = _compute_psi(train, evaluation, n_bins=10)
    assert psi > DEFAULT_PSI_THRESHOLD


def test_psi_degenerate_single_value() -> None:
    """Single unique value in training data should return 0 PSI."""
    train = np.array([5.0] * 100)
    evaluation = np.array([5.0] * 100)
    psi, _, _ = _compute_psi(train, evaluation)
    assert psi == 0.0


def test_pinball_loss_exact_prediction() -> None:
    """Pinball loss should be 0 when predictions match exactly."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert _pinball_loss(y_true, y_pred, alpha=0.5) == pytest.approx(0.0)


def test_pinball_loss_underprediction() -> None:
    """Under-predictions should have higher loss for higher quantiles."""
    y_true = np.array([100.0])
    y_pred = np.array([50.0])
    loss_p50 = _pinball_loss(y_true, y_pred, alpha=0.5)
    loss_p90 = _pinball_loss(y_true, y_pred, alpha=0.9)
    assert loss_p90 > loss_p50  # higher alpha penalizes under-prediction more


def test_pinball_loss_overprediction() -> None:
    """Over-predictions should have higher loss for lower quantiles."""
    y_true = np.array([50.0])
    y_pred = np.array([100.0])
    loss_p10 = _pinball_loss(y_true, y_pred, alpha=0.1)
    loss_p50 = _pinball_loss(y_true, y_pred, alpha=0.5)
    assert loss_p10 > loss_p50


def test_default_thresholds() -> None:
    assert DEFAULT_PSI_THRESHOLD == 0.20
    assert DEFAULT_METRIC_DEGRADATION_FACTOR == 1.50
    assert DEFAULT_N_BINS == 10


def test_compute_drift_report_end_to_end(tmp_path: Path) -> None:
    """Full drift report should work with a trained model and eval data."""
    from hpcopt.models.drift import compute_drift_report
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=100,
        seed=1,
        params={"alpha": 1.25},
    )
    result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path,
        out_dir=tmp_path / "m",
        model_id="drift_test",
        seed=1,
    )

    report = compute_drift_report(
        model_dir=result.model_dir,
        eval_dataset_path=stress.dataset_path,
    )
    assert isinstance(report.overall_drift_detected, bool)
    assert isinstance(report.summary, dict)
    assert len(report.metric_degradations) > 0
