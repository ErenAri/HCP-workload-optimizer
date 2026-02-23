"""Tests for the runtime predictor model cache."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from hpcopt.api.model_cache import (
    get_runtime_predictor,
    is_loaded,
    reset_for_testing,
    warm_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    reset_for_testing()
    yield  # type: ignore[misc]
    reset_for_testing()


def test_cache_miss_returns_none() -> None:
    """When no model dir is configured, get_runtime_predictor returns (None, None)."""
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=None):
        predictor, model_dir = get_runtime_predictor()
    assert predictor is None
    assert model_dir is None


def test_is_loaded_false_by_default() -> None:
    assert is_loaded() is False


def test_warm_cache_returns_false_when_no_model() -> None:
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=None):
        assert warm_cache() is False


def test_warm_cache_returns_true_with_trained_model(tmp_path: Path) -> None:
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail", out_dir=tmp_path, n_jobs=80, seed=1, params={"alpha": 1.25},
    )
    result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path, out_dir=tmp_path / "m", model_id="cache_test", seed=1,
    )
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=result.model_dir):
        assert warm_cache() is True
        assert is_loaded() is True


def test_cache_hit_returns_same_instance(tmp_path: Path) -> None:
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail", out_dir=tmp_path, n_jobs=80, seed=2, params={"alpha": 1.25},
    )
    result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path, out_dir=tmp_path / "m", model_id="hit_test", seed=2,
    )
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=result.model_dir):
        p1, _ = get_runtime_predictor()
        p2, _ = get_runtime_predictor()
        assert p1 is p2  # same cached instance


def test_retry_on_io_error(tmp_path: Path) -> None:
    """OSError during load should retry and eventually return None after exhausting attempts."""
    fake_dir = tmp_path / "fake_model"
    fake_dir.mkdir()

    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=fake_dir), \
         patch("hpcopt.api.model_cache.RuntimeQuantilePredictor", side_effect=OSError("disk read error")):
        predictor, _ = get_runtime_predictor()
    assert predictor is None


def test_concurrent_access(tmp_path: Path) -> None:
    """Multiple threads can safely call get_runtime_predictor simultaneously."""
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    from hpcopt.simulate.stress import generate_stress_scenario

    stress = generate_stress_scenario(
        scenario="heavy_tail", out_dir=tmp_path, n_jobs=80, seed=3, params={"alpha": 1.25},
    )
    result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path, out_dir=tmp_path / "m", model_id="conc_test", seed=3,
    )

    results: list[object] = []
    errors: list[Exception] = []

    def _get() -> None:
        try:
            p, _ = get_runtime_predictor()
            results.append(p)
        except Exception as e:
            errors.append(e)

    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=result.model_dir):
        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(errors) == 0
    assert len(results) == 10
    # All should return the same cached instance
    assert all(r is results[0] for r in results)
