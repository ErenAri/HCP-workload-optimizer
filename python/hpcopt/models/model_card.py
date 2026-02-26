"""Model card generation for trained HPC workload prediction models.

Generates structured model cards documenting dataset characteristics, features,
performance metrics, fairness/bias evaluation, known limitations, and intended use.
Output: ``model_card.json`` alongside existing model artifacts.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.utils.io import write_json

logger = logging.getLogger(__name__)


def generate_model_card(
    model_dir: Path,
    dataset_path: Path,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
    group_columns: list[str] | None = None,
) -> Path:
    """Generate a model card JSON file for a trained model.

    Parameters
    ----------
    model_dir:
        Directory containing trained model artifacts.
    dataset_path:
        Path to the training dataset (parquet).
    metrics:
        Training metrics dict (from metrics.json).
    metadata:
        Model metadata dict (from metadata.json).
    feature_columns:
        List of feature column names used by the model.
    target_column:
        Name of the target column.
    group_columns:
        Columns to evaluate for fairness (e.g. user_id, group_id).

    Returns
    -------
    Path
        Path to the generated model_card.json file.
    """
    if group_columns is None:
        group_columns = ["user_id", "group_id"]

    card: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "model_id": metadata.get("model_id", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
    }

    # --- Model details ---
    card["model_details"] = {
        "description": "Gradient boosting quantile regression model for HPC job runtime prediction.",
        "intended_use": "Predict p10/p50/p90 runtime quantiles for HPC job scheduling and resource allocation.",
        "out_of_scope_use": "Not suitable for real-time safety-critical scheduling without human oversight.",
        "version": metadata.get("model_id", "unknown"),
        "training_date": metadata.get("trained_at_utc"),
        "framework": "scikit-learn GradientBoostingRegressor",
        "features": feature_columns,
        "target": target_column,
    }

    # --- Dataset characteristics ---
    dataset_info: dict[str, Any] = {
        "path": str(dataset_path),
        "rows_total": metrics.get("rows_total"),
        "rows_train": metrics.get("rows_train"),
        "rows_test": metrics.get("rows_test"),
    }

    try:
        df = pd.read_parquet(dataset_path)
        dataset_info["column_count"] = len(df.columns)
        for col in feature_columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any():
                    dataset_info[f"{col}_stats"] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "null_pct": float(series.isna().mean()),
                    }
    except (OSError, ValueError) as exc:
        logger.warning("Could not load dataset for model card: %s", exc)

    card["dataset"] = dataset_info

    # --- Performance metrics ---
    card["performance"] = {
        "quantile_metrics": metrics.get("quantiles", {}),
        "interval_coverage_p10_p90": metrics.get("interval_coverage_p10_p90"),
        "naive_baselines": metrics.get("naive_baselines", {}),
        "p50_lift_vs_naive": metrics.get("p50_lift_vs_naive", {}),
    }

    # --- Fairness / bias evaluation ---
    fairness: dict[str, Any] = {}
    try:
        df = pd.read_parquet(dataset_path)
        for group_col in group_columns:
            if group_col not in df.columns:
                continue
            groups = df[group_col].dropna()
            if groups.empty:
                continue

            group_stats: dict[str, Any] = {
                "unique_values": int(groups.nunique()),
                "top_5_counts": groups.value_counts().head(5).to_dict(),
            }

            # Error distribution by group (using target column stats)
            grouped = df.groupby(group_col)[target_column].agg(["mean", "std", "count"])
            grouped = grouped[grouped["count"] >= 10]  # min 10 samples
            if not grouped.empty:
                group_stats["runtime_mean_by_group"] = {
                    "min": float(grouped["mean"].min()),
                    "max": float(grouped["mean"].max()),
                    "std_across_groups": float(grouped["mean"].std()),
                }

            fairness[group_col] = group_stats
    except (OSError, ValueError) as exc:
        logger.warning("Could not compute fairness metrics: %s", exc)

    card["fairness_evaluation"] = fairness

    # --- Known limitations ---
    card["limitations"] = [
        "Predictions may be less accurate for job types not well-represented in training data.",
        "Model assumes workload patterns are relatively stable over time "
        "(may degrade under significant workload shifts).",
        "Categorical features (user_id, queue_id) may not generalize to unseen values.",
        "Runtime floor of 1 second is enforced; sub-second predictions are not supported.",
    ]

    card["ethical_considerations"] = [
        "Model uses user_id and group_id as features, which may encode demographic information.",
        "Recommend periodic fairness audits via drift detection to identify systematic prediction bias.",
    ]

    # --- Write outputs ---
    card_path = model_dir / "model_card.json"
    write_json(card_path, card)
    logger.info("Model card written to %s", card_path)

    return card_path
