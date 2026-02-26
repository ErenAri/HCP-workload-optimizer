"""Tests for hyperparameter tuning module."""

from __future__ import annotations

from pathlib import Path


def test_hyperparams_defaults() -> None:
    from hpcopt.models.tuning import HyperParams

    hp = HyperParams()
    assert hp.n_estimators > 0
    assert 0 < hp.learning_rate <= 1.0
    assert hp.max_depth >= 1


def test_hyperparams_roundtrip() -> None:
    from hpcopt.models.tuning import HyperParams

    hp = HyperParams(n_estimators=50, learning_rate=0.05, max_depth=4)
    d = hp.to_dict()
    hp2 = HyperParams.from_dict(d)
    assert hp2.n_estimators == 50
    assert hp2.learning_rate == 0.05
    assert hp2.max_depth == 4


def test_build_tuning_report(tmp_path: Path) -> None:
    """End-to-end tuning on small synthetic data with random search."""
    from hpcopt.models.tuning import build_tuning_report
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=80,
        seed=42,
        params={"alpha": 1.25},
    )

    result = build_tuning_report(
        dataset_path=stress.dataset_path,
        out_path=tmp_path / "tuning" / "report.json",
        n_trials=3,
        seed=42,
    )
    assert result.report_path.exists()
    assert result.best_params is not None
    assert result.best_score is not None
