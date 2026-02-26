from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hpcopt.models.runtime_quantile import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    QUANTILES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (overridden by configs/models/drift_thresholds.yaml)
# ---------------------------------------------------------------------------

DEFAULT_PSI_THRESHOLD = 0.20  # > 0.20 indicates significant distributional shift
DEFAULT_METRIC_DEGRADATION_FACTOR = 1.50  # 50% worse than baseline is a flag
DEFAULT_N_BINS = 10

_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "psi_threshold": DEFAULT_PSI_THRESHOLD,
    "metric_degradation_factor": DEFAULT_METRIC_DEGRADATION_FACTOR,
    "n_bins": DEFAULT_N_BINS,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FeaturePSI:
    """PSI result for a single numeric feature."""

    feature: str
    psi_value: float
    threshold: float
    flagged: bool
    n_bins: int
    train_bin_counts: list[int]
    eval_bin_counts: list[int]


@dataclass
class MetricDegradation:
    """Metric comparison between baseline and evaluation."""

    quantile_name: str
    alpha: float
    baseline_pinball: float | None
    eval_pinball: float
    degradation_ratio: float | None
    flagged: bool


@dataclass
class DriftReport:
    """Full drift assessment for a model evaluated on new data."""

    model_dir: str
    eval_dataset_path: str
    feature_psi: list[FeaturePSI] = field(default_factory=list)
    metric_degradations: list[MetricDegradation] = field(default_factory=list)
    overall_drift_detected: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "eval_dataset_path": self.eval_dataset_path,
            "feature_psi": [
                {
                    "feature": f.feature,
                    "psi_value": f.psi_value,
                    "threshold": f.threshold,
                    "flagged": f.flagged,
                    "n_bins": f.n_bins,
                }
                for f in self.feature_psi
            ],
            "metric_degradations": [
                {
                    "quantile_name": m.quantile_name,
                    "alpha": m.alpha,
                    "baseline_pinball": m.baseline_pinball,
                    "eval_pinball": m.eval_pinball,
                    "degradation_ratio": m.degradation_ratio,
                    "flagged": m.flagged,
                }
                for m in self.metric_degradations
            ],
            "overall_drift_detected": self.overall_drift_detected,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_thresholds(config_path: Path | None) -> dict[str, Any]:
    """Load drift thresholds from YAML config, falling back to defaults."""
    if config_path is None:
        config_path = Path("configs/models/drift_thresholds.yaml")
    if not config_path.exists():
        logger.info("Drift threshold config not found at %s; using defaults.", config_path)
        return dict(_DEFAULT_THRESHOLDS)

    try:
        import yaml  # optional dependency

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("Drift config is not a mapping; using defaults.")
            return dict(_DEFAULT_THRESHOLDS)
        merged = dict(_DEFAULT_THRESHOLDS)
        merged.update(raw)
        return merged
    except ImportError:
        logger.warning("PyYAML not installed; using default drift thresholds.")
        return dict(_DEFAULT_THRESHOLDS)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.warning("Failed to load drift config %s: %s; using defaults.", config_path, exc)
        return dict(_DEFAULT_THRESHOLDS)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    delta = y_true - y_pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def _compute_psi(
    train_values: np.ndarray,
    eval_values: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[float, list[int], list[int]]:
    """Compute Population Stability Index between two distributions.

    Bins are derived from the *training* distribution quantiles so that the
    baseline has roughly uniform counts per bin.
    """
    eps = 1e-6

    # Build bin edges from training distribution quantiles.
    quantile_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(train_values, quantile_edges)
    # Ensure unique edges (collapse duplicates to avoid zero-width bins).
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # Degenerate case -- single unique value in training data.
        return 0.0, [len(train_values)], [len(eval_values)]

    train_counts, _ = np.histogram(train_values, bins=bin_edges)
    eval_counts, _ = np.histogram(eval_values, bins=bin_edges)

    # Normalise to proportions.
    train_prop = train_counts.astype(float) / max(train_counts.sum(), 1)
    eval_prop = eval_counts.astype(float) / max(eval_counts.sum(), 1)

    # Add epsilon to avoid log(0).
    train_prop = np.clip(train_prop, eps, None)
    eval_prop = np.clip(eval_prop, eps, None)

    psi = float(np.sum((eval_prop - train_prop) * np.log(eval_prop / train_prop)))
    return psi, train_counts.tolist(), eval_counts.tolist()


def _prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preparation to align eval data with model expectations."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Derive time-of-day features if missing.
    if "submit_hour" not in df.columns or df["submit_hour"].isna().all():
        if "submit_ts" in df.columns:
            ts = pd.to_datetime(pd.to_numeric(df["submit_ts"], errors="coerce"), unit="s", utc=True)
            df["submit_hour"] = ts.dt.hour
            df["submit_dow"] = ts.dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_drift_report(
    model_dir: str | Path,
    eval_dataset_path: str | Path,
    baseline_metrics_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> DriftReport:
    """Evaluate a trained model on new data and compute drift indicators.

    Parameters
    ----------
    model_dir:
        Directory containing ``p10.joblib``, ``p50.joblib``, ``p90.joblib``
        and optionally ``metrics.json`` (used as baseline if
        *baseline_metrics_path* is ``None``).
    eval_dataset_path:
        Path to a parquet file with the same canonical schema produced by
        the ingestion pipeline.
    baseline_metrics_path:
        Optional path to a ``metrics.json`` from the original training run.
        Falls back to ``<model_dir>/metrics.json``.
    config_path:
        Optional path to a YAML file with drift threshold overrides.

    Returns
    -------
    DriftReport
        Dataclass containing per-feature PSI, metric degradation flags, and
        an ``overall_drift_detected`` boolean.
    """
    model_dir = Path(model_dir)
    eval_dataset_path = Path(eval_dataset_path)
    thresholds = _load_thresholds(Path(config_path) if config_path else None)

    psi_threshold = float(thresholds.get("psi_threshold", DEFAULT_PSI_THRESHOLD))
    degrad_factor = float(thresholds.get("metric_degradation_factor", DEFAULT_METRIC_DEGRADATION_FACTOR))
    n_bins = int(thresholds.get("n_bins", DEFAULT_N_BINS))

    # ------------------------------------------------------------------
    # Load evaluation data
    # ------------------------------------------------------------------
    eval_df = pd.read_parquet(eval_dataset_path)
    eval_df = _prepare_eval_frame(eval_df)

    if eval_df.empty:
        raise ValueError("Evaluation dataset is empty.")

    # ------------------------------------------------------------------
    # Load baseline metrics (from training run)
    # ------------------------------------------------------------------
    baseline_metrics: dict[str, Any] | None = None
    if baseline_metrics_path is not None:
        bpath = Path(baseline_metrics_path)
    else:
        bpath = model_dir / "metrics.json"

    if bpath.exists():
        try:
            baseline_metrics = json.loads(bpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load baseline metrics from %s: %s", bpath, exc)

    # ------------------------------------------------------------------
    # Load training data distribution (from dataset referenced in metadata)
    # ------------------------------------------------------------------
    train_df: pd.DataFrame | None = None
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            ds_path = Path(str(meta.get("dataset_path", "")))
            if ds_path.exists():
                train_df = pd.read_parquet(ds_path)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Could not load training dataset for PSI: %s", exc)

    # ------------------------------------------------------------------
    # PSI computation for numeric features
    # ------------------------------------------------------------------
    feature_psi_results: list[FeaturePSI] = []
    if train_df is not None:
        for feat in NUMERIC_FEATURES:
            if feat not in train_df.columns or feat not in eval_df.columns:
                continue
            train_vals = pd.to_numeric(train_df[feat], errors="coerce").dropna().to_numpy()
            eval_vals = pd.to_numeric(eval_df[feat], errors="coerce").dropna().to_numpy()
            if len(train_vals) < n_bins or len(eval_vals) < n_bins:
                logger.info(
                    "Skipping PSI for %s: insufficient data (train=%d, eval=%d).",
                    feat,
                    len(train_vals),
                    len(eval_vals),
                )
                continue

            psi_value, train_counts, eval_counts = _compute_psi(train_vals, eval_vals, n_bins=n_bins)
            feature_psi_results.append(
                FeaturePSI(
                    feature=feat,
                    psi_value=psi_value,
                    threshold=psi_threshold,
                    flagged=psi_value > psi_threshold,
                    n_bins=n_bins,
                    train_bin_counts=train_counts,
                    eval_bin_counts=eval_counts,
                )
            )

    # ------------------------------------------------------------------
    # Metric degradation -- pinball loss per quantile
    # ------------------------------------------------------------------
    metric_degradations: list[MetricDegradation] = []
    y_eval = pd.to_numeric(eval_df.get(TARGET_COLUMN), errors="coerce").dropna().to_numpy()
    x_eval = eval_df[FEATURE_COLUMNS]

    for q_name, alpha in QUANTILES.items():
        model_path = model_dir / f"{q_name}.joblib"
        if not model_path.exists():
            logger.warning("Model artifact missing: %s", model_path)
            continue

        pipeline = joblib.load(model_path)
        try:
            preds = np.maximum(pipeline.predict(x_eval), 1.0)
        except (ValueError, OSError) as exc:
            logger.warning("Prediction failed for %s: %s", q_name, exc)
            continue

        eval_pinball = _pinball_loss(y_eval, preds, alpha)

        baseline_pinball: float | None = None
        if baseline_metrics and "quantiles" in baseline_metrics:
            q_block = baseline_metrics["quantiles"].get(q_name)
            if q_block and "pinball_loss" in q_block:
                baseline_pinball = float(q_block["pinball_loss"])

        if baseline_pinball is not None and baseline_pinball > 0:
            degradation_ratio = eval_pinball / baseline_pinball
        else:
            degradation_ratio = None

        flagged = degradation_ratio is not None and degradation_ratio > degrad_factor

        metric_degradations.append(
            MetricDegradation(
                quantile_name=q_name,
                alpha=alpha,
                baseline_pinball=baseline_pinball,
                eval_pinball=eval_pinball,
                degradation_ratio=degradation_ratio,
                flagged=flagged,
            )
        )

    # ------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------
    any_psi_flag = any(f.flagged for f in feature_psi_results)
    any_metric_flag = any(m.flagged for m in metric_degradations)
    overall_drift = any_psi_flag or any_metric_flag

    summary: dict[str, Any] = {
        "psi_features_flagged": [f.feature for f in feature_psi_results if f.flagged],
        "metrics_flagged": [m.quantile_name for m in metric_degradations if m.flagged],
        "psi_threshold_used": psi_threshold,
        "degradation_factor_used": degrad_factor,
        "eval_row_count": len(eval_df),
        "train_row_count": len(train_df) if train_df is not None else None,
    }

    report = DriftReport(
        model_dir=str(model_dir),
        eval_dataset_path=str(eval_dataset_path),
        feature_psi=feature_psi_results,
        metric_degradations=metric_degradations,
        overall_drift_detected=overall_drift,
        summary=summary,
    )
    logger.info(
        "Drift report: overall_drift_detected=%s, psi_flagged=%s, metric_flagged=%s",
        overall_drift,
        summary["psi_features_flagged"],
        summary["metrics_flagged"],
    )
    return report
