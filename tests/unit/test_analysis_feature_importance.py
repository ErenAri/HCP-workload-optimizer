"""Tests for feature importance analysis."""
from __future__ import annotations

from pathlib import Path

import pytest

from hpcopt.analysis.feature_importance import (
    build_importance_report,
    compute_permutation_importance,
)


def _train_model(tmp_path: Path) -> tuple[Path, Path]:
    """Train a model and return (model_dir, dataset_path)."""
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail", out_dir=tmp_path, n_jobs=100, seed=7, params={"alpha": 1.25},
    )
    result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path,
        out_dir=tmp_path / "models",
        model_id="fi_test",
        seed=7,
    )
    return result.model_dir, stress.dataset_path


def test_compute_permutation_importance(tmp_path: Path) -> None:
    model_dir, dataset_path = _train_model(tmp_path)

    import pandas as pd
    from hpcopt.models.runtime_quantile import _prepare_training_frame, _time_split

    df = _prepare_training_frame(pd.read_parquet(dataset_path))
    _, _, test_df = _time_split(df)

    result = compute_permutation_importance(
        model_dir=model_dir,
        test_df=test_df,
        n_repeats=5,
        seed=42,
        quantile_name="p50",
    )
    assert "features" in result
    assert len(result["features"]) > 0
    assert result["quantile"] == "p50"
    assert all("importance_mean" in f for f in result["features"])


def test_compute_permutation_importance_missing_model(tmp_path: Path) -> None:
    import pandas as pd
    empty_df = pd.DataFrame({"requested_cpus": [1], "runtime_actual_sec": [100]})
    with pytest.raises(FileNotFoundError):
        compute_permutation_importance(
            model_dir=tmp_path / "nonexistent",
            test_df=empty_df,
        )


def test_build_importance_report(tmp_path: Path) -> None:
    model_dir, dataset_path = _train_model(tmp_path)
    out_path = tmp_path / "report.json"

    result = build_importance_report(
        model_dir=model_dir,
        dataset_path=dataset_path,
        out_path=out_path,
        n_repeats=3,
    )
    assert result.report_path.exists()
    assert "permutation_importance" in result.payload
    assert len(result.payload["permutation_importance"]) == 3  # p10, p50, p90
