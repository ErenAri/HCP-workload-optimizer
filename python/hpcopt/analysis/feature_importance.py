"""Feature importance analysis: permutation importance + optional SHAP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from hpcopt.models.runtime_quantile import (
    FEATURE_COLUMNS,
    QUANTILES,
    TARGET_COLUMN,
    _prepare_training_frame,
    _time_split,
)
from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureImportanceResult:
    report_path: Path
    payload: dict[str, Any]


def compute_permutation_importance(
    model_dir: Path,
    test_df: pd.DataFrame,
    n_repeats: int = 10,
    seed: int = 42,
    quantile_name: str = "p50",
) -> dict[str, Any]:
    """Compute permutation importance for a trained quantile model."""
    model_path = model_dir / f"{quantile_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)
    X = test_df[FEATURE_COLUMNS]
    y = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )

    ranked: list[dict[str, Any]] = []
    for i, col in enumerate(FEATURE_COLUMNS):
        ranked.append(
            {
                "feature": col,
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
                "importance_ci_lower": float(
                    result.importances_mean[i] - 1.96 * result.importances_std[i]
                ),
                "importance_ci_upper": float(
                    result.importances_mean[i] + 1.96 * result.importances_std[i]
                ),
            }
        )
    ranked.sort(key=lambda x: float(x["importance_mean"]), reverse=True)
    return {
        "quantile": quantile_name,
        "n_repeats": n_repeats,
        "seed": seed,
        "n_samples": len(y),
        "features": ranked,
    }


def _try_shap_importance(
    model_dir: Path,
    test_df: pd.DataFrame,
    quantile_name: str = "p50",
    max_samples: int = 500,
) -> dict[str, Any] | None:
    """Attempt SHAP TreeExplainer; graceful fallback if shap not installed."""
    try:
        import shap
    except ImportError:
        logger.info("shap not installed; skipping SHAP feature importance")
        return None

    model_path = model_dir / f"{quantile_name}.joblib"
    if not model_path.exists():
        return None

    pipeline = joblib.load(model_path)
    X = test_df[FEATURE_COLUMNS]
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    # Transform features through the preprocessor
    preprocessor = pipeline.named_steps["preprocess"]
    X_transformed = preprocessor.transform(X)

    try:
        regressor = pipeline.named_steps["regressor"]
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)

        # Get feature names from the preprocessor
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        features: list[dict[str, Any]] = []
        for i, name in enumerate(feature_names):
            features.append(
                {
                    "feature": name,
                    "mean_abs_shap": float(mean_abs_shap[i]),
                }
            )
        features.sort(key=lambda x: float(x["mean_abs_shap"]), reverse=True)

        return {
            "quantile": quantile_name,
            "method": "shap_tree_explainer",
            "n_samples": len(X),
            "features": features,
        }
    except (ValueError, OSError) as exc:
        logger.warning("SHAP analysis failed: %s", exc)
        return None


def build_importance_report(
    model_dir: Path,
    dataset_path: Path,
    out_path: Path,
    n_repeats: int = 10,
    seed: int = 42,
) -> FeatureImportanceResult:
    """Build feature importance report with permutation + optional SHAP."""
    ensure_dir(out_path.parent)

    trace_df = pd.read_parquet(dataset_path)
    df = _prepare_training_frame(trace_df)
    _, _, test_df = _time_split(df)

    permutation_results: dict[str, Any] = {}
    shap_results: dict[str, Any] = {}

    for quantile_name in QUANTILES:
        perm = compute_permutation_importance(
            model_dir=model_dir,
            test_df=test_df,
            n_repeats=n_repeats,
            seed=seed,
            quantile_name=quantile_name,
        )
        permutation_results[quantile_name] = perm

        shap_result = _try_shap_importance(
            model_dir=model_dir,
            test_df=test_df,
            quantile_name=quantile_name,
        )
        if shap_result is not None:
            shap_results[quantile_name] = shap_result

    payload = {
        "model_dir": str(model_dir),
        "dataset_path": str(dataset_path),
        "test_samples": len(test_df),
        "permutation_importance": permutation_results,
        "shap_importance": shap_results if shap_results else None,
    }

    write_json(out_path, payload)
    return FeatureImportanceResult(report_path=out_path, payload=payload)
