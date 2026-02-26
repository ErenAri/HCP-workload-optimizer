"""Tests for ensemble runtime predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from hpcopt.models.ensemble import EnsemblePredictor
from hpcopt.models.runtime_quantile import MIN_PREDICTION_SEC


class _DummyPredictor:
    def __init__(self, prediction: dict[str, float]):
        self._prediction = prediction

    def predict_one(self, features: dict[str, float]) -> dict[str, float]:
        del features
        return dict(self._prediction)


class _FakeRuntimePredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def predict_one(self, features: dict[str, float]) -> dict[str, float]:
        del features
        return {"p10": 10.0, "p50": 20.0, "p90": 30.0}


def _write_metrics(model_dir: Path, p10: float, p50: float, p90: float) -> None:
    payload = {
        "quantiles": {
            "p10": {"pinball_loss": p10},
            "p50": {"pinball_loss": p50},
            "p90": {"pinball_loss": p90},
        }
    }
    (model_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ensemble_init_validates_predictor_count() -> None:
    with pytest.raises(ValueError, match="at least 2 predictors"):
        EnsemblePredictor(predictors=[_DummyPredictor({"p10": 1.0, "p50": 2.0, "p90": 3.0})])  # type: ignore[list-item]


def test_ensemble_init_normalizes_weights_and_validates_length() -> None:
    p1 = _DummyPredictor({"p10": 1.0, "p50": 2.0, "p90": 3.0})
    p2 = _DummyPredictor({"p10": 2.0, "p50": 3.0, "p90": 4.0})

    with pytest.raises(ValueError, match="weights length must match"):
        EnsemblePredictor(predictors=[p1, p2], weights=[1.0])  # type: ignore[list-item]

    ensemble = EnsemblePredictor(predictors=[p1, p2], weights=[1.0, 3.0], model_names=["a", "b"])  # type: ignore[list-item]
    assert ensemble.weights == pytest.approx([0.25, 0.75])
    assert ensemble.summary == {
        "type": "ensemble",
        "n_models": 2,
        "members": [{"name": "a", "weight": 0.25}, {"name": "b", "weight": 0.75}],
    }


def test_from_model_dirs_auto_weights_with_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hpcopt.models.ensemble.RuntimeQuantilePredictor", _FakeRuntimePredictor)

    good_a = tmp_path / "model_a"
    good_b = tmp_path / "model_b"
    good_a.mkdir()
    good_b.mkdir()
    _write_metrics(good_a, p10=1.0, p50=1.0, p90=1.0)  # avg 1.0
    _write_metrics(good_b, p10=2.0, p50=2.0, p90=2.0)  # avg 2.0

    ensemble = EnsemblePredictor.from_model_dirs([good_a, good_b], auto_weight=True)
    assert ensemble.model_names == ["model_a", "model_b"]
    assert ensemble.weights == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_from_model_dirs_skips_missing_and_requires_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("hpcopt.models.ensemble.RuntimeQuantilePredictor", _FakeRuntimePredictor)

    missing = tmp_path / "missing_dir"
    only_valid = tmp_path / "single_model"
    only_valid.mkdir()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="Need at least 2 valid model dirs, found 1"):
            EnsemblePredictor.from_model_dirs([missing, only_valid], auto_weight=True)
    assert "Model dir not found, skipping" in caplog.text


def test_predict_one_applies_floor_and_monotonic_ordering() -> None:
    p1 = _DummyPredictor({"p10": -5.0, "p50": 2.0, "p90": 1.0})
    p2 = _DummyPredictor({"p10": 0.0, "p50": 5.0, "p90": 3.0})
    ensemble = EnsemblePredictor(predictors=[p1, p2])  # type: ignore[list-item]

    pred = ensemble.predict_one({"requested_cpus": 4})
    assert pred["p10"] == pytest.approx(MIN_PREDICTION_SEC)
    assert pred["p50"] == pytest.approx(2.0)
    assert pred["p90"] == pytest.approx(3.5)


def test_predict_batch_preserves_index() -> None:
    p1 = _DummyPredictor({"p10": 10.0, "p50": 20.0, "p90": 30.0})
    p2 = _DummyPredictor({"p10": 10.0, "p50": 20.0, "p90": 30.0})
    ensemble = EnsemblePredictor(predictors=[p1, p2], model_names=["left", "right"])  # type: ignore[list-item]

    features_df = pd.DataFrame(
        [{"requested_cpus": 2}, {"requested_cpus": 4}],
        index=[101, 202],
    )
    out = ensemble.predict_batch(features_df)

    assert list(out.columns) == ["p10", "p50", "p90"]
    assert out.index.tolist() == [101, 202]
    assert out.iloc[0].to_dict() == {"p10": 10.0, "p50": 20.0, "p90": 30.0}
