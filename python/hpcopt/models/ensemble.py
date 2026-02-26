"""Ensemble runtime predictor combining multiple model backends.

Supports weighted averaging of predictions from different backends
(sklearn, lightgbm) trained on the same data. Automatically selects
the best model per quantile based on validation metrics, or uses
user-specified weights.

Usage:
    from hpcopt.models.ensemble import EnsemblePredictor

    ensemble = EnsemblePredictor.from_model_dirs([
        Path("outputs/models/ctc_sp2_sklearn"),
        Path("outputs/models/ctc_sp2_lightgbm"),
    ])
    prediction = ensemble.predict_one(features)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpcopt.models.runtime_quantile import (
    FEATURE_COLUMNS,
    MIN_PREDICTION_SEC,
    QUANTILES,
    RuntimeQuantilePredictor,
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble of multiple RuntimeQuantilePredictor models.

    Combines predictions using configurable weights. Defaults to
    inverse-pinball-loss weighting (better models get more weight).
    """

    def __init__(
        self,
        predictors: list[RuntimeQuantilePredictor],
        weights: list[float] | None = None,
        model_names: list[str] | None = None,
    ):
        if len(predictors) < 2:
            raise ValueError("Ensemble requires at least 2 predictors")

        self.predictors = predictors
        self.model_names = model_names or [
            f"model_{i}" for i in range(len(predictors))
        ]

        if weights is not None:
            if len(weights) != len(predictors):
                raise ValueError("weights length must match predictors length")
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(predictors)] * len(predictors)

    @classmethod
    def from_model_dirs(
        cls,
        model_dirs: list[Path],
        auto_weight: bool = True,
    ) -> "EnsemblePredictor":
        """Create ensemble from list of model directories.

        If auto_weight=True, weights are set based on inverse pinball
        loss from each model's metrics.json (better models = higher weight).
        """
        predictors = []
        names = []
        pinball_scores: list[float] = []

        for d in model_dirs:
            d = Path(d)
            if not d.exists():
                logger.warning("Model dir not found, skipping: %s", d)
                continue

            predictors.append(RuntimeQuantilePredictor(d))
            names.append(d.name)

            metrics_path = d / "metrics.json"
            if metrics_path.exists() and auto_weight:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                # Use average pinball loss across quantiles as quality score
                qm = metrics.get("quantiles", {})
                avg_pinball = np.mean([
                    qm[q]["pinball_loss"] for q in QUANTILES if q in qm
                ])
                pinball_scores.append(float(avg_pinball))

        if len(predictors) < 2:
            raise ValueError(
                f"Need at least 2 valid model dirs, found {len(predictors)}"
            )

        weights = None
        if auto_weight and len(pinball_scores) == len(predictors):
            # Inverse pinball loss: lower loss -> higher weight
            inv = [1.0 / max(s, 1e-6) for s in pinball_scores]
            total = sum(inv)
            weights = [w / total for w in inv]
            for name, w, pb in zip(names, weights, pinball_scores):
                logger.info(
                    "Ensemble member %s: pinball=%.2f, weight=%.3f",
                    name, pb, w,
                )

        return cls(predictors=predictors, weights=weights, model_names=names)

    def predict_one(self, features: dict[str, Any]) -> dict[str, float]:
        """Weighted ensemble prediction for a single job."""
        weighted_preds: dict[str, float] = {q: 0.0 for q in QUANTILES}

        for predictor, weight in zip(self.predictors, self.weights):
            pred = predictor.predict_one(features)
            for q in QUANTILES:
                weighted_preds[q] += weight * pred[q]

        # Floor and enforce monotonic quantiles
        vals = [max(MIN_PREDICTION_SEC, weighted_preds[q]) for q in QUANTILES]
        ordered = sorted(vals)
        return dict(zip(QUANTILES.keys(), ordered))

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict for a batch of jobs, returning DataFrame with p10/p50/p90 columns."""
        results = []
        for _, row in features_df.iterrows():
            pred = self.predict_one(row.to_dict())
            results.append(pred)
        return pd.DataFrame(results, index=features_df.index)

    @property
    def summary(self) -> dict[str, Any]:
        """Summary of ensemble configuration."""
        return {
            "type": "ensemble",
            "n_models": len(self.predictors),
            "members": [
                {"name": name, "weight": round(w, 4)}
                for name, w in zip(self.model_names, self.weights)
            ],
        }
