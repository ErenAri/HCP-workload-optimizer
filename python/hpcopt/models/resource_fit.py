"""Resource-fit ML model: fragmentation risk classifier + optimal node size regressor."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)

# LightGBM backend auto-detection (mirrors runtime_quantile.py)
try:
    with warnings.catch_warnings():
        # LightGBM optionally imports matplotlib/pyparsing paths that emit
        # interpreter deprecation warnings on Python 3.11+.
        warnings.filterwarnings("ignore", message=r"module 'sre_constants' is deprecated", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)
        import lightgbm as lgb  # noqa: F401

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

DEFAULT_BACKEND = "lightgbm" if _HAS_LIGHTGBM else "sklearn"

RESOURCE_FEATURES = [
    "requested_cpus",
    "requested_mem",
    "runtime_requested_sec",
    "queue_id",
    "partition_id",
    "user_id",
    "submit_hour",
    "submit_dow",
]

NUMERIC_RESOURCE_FEATURES = ["requested_cpus", "requested_mem", "runtime_requested_sec"]
CATEGORICAL_RESOURCE_FEATURES = ["queue_id", "partition_id", "user_id", "submit_hour", "submit_dow"]


@dataclass(frozen=True)
class ResourceFitTrainResult:
    model_dir: Path
    metrics_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ResourceFitPrediction:
    recommended_node_cpus: int
    fragmentation_risk: str  # low, medium, high
    confidence: float
    fallback_used: bool


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_RESOURCE_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_RESOURCE_FEATURES,
            ),
        ]
    )


def _prepare_resource_frame(trace_df: pd.DataFrame, node_sizes: list[int]) -> pd.DataFrame:
    """Prepare training data for resource-fit model."""
    required = {"requested_cpus", "allocated_cpus", "runtime_actual_sec"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"Resource-fit training missing columns: {sorted(missing)}")

    df = trace_df.copy()
    for col in RESOURCE_FEATURES:
        if col not in df.columns:
            df[col] = None

    df["requested_cpus"] = pd.to_numeric(df["requested_cpus"], errors="coerce").clip(lower=1)
    df["allocated_cpus"] = pd.to_numeric(df["allocated_cpus"], errors="coerce").clip(lower=1)
    df = df.dropna(subset=["requested_cpus", "allocated_cpus"])

    if "submit_ts" in df.columns:
        submit_dt = pd.to_datetime(df["submit_ts"], unit="s", utc=True, errors="coerce")
        df["submit_hour"] = submit_dt.dt.hour.fillna(0).astype(int)
        df["submit_dow"] = submit_dt.dt.dayofweek.fillna(0).astype(int)

    # Compute optimal node and fragmentation labels
    sorted_sizes = sorted(node_sizes)
    optimal_nodes = []
    frag_labels = []
    for _, row in df.iterrows():
        req = int(row["requested_cpus"])
        best = sorted_sizes[-1]
        for size in sorted_sizes:
            if size >= req:
                best = size
                break
        optimal_nodes.append(best)
        waste = max(best - req, 0) / best if best > 0 else 0
        if waste <= 0.15:
            frag_labels.append(0)  # low
        elif waste <= 0.35:
            frag_labels.append(1)  # medium
        else:
            frag_labels.append(2)  # high

    df["optimal_node_cpus"] = optimal_nodes
    df["frag_label"] = frag_labels
    return df


def train_resource_fit_model(
    dataset_path: Path,
    out_dir: Path,
    model_id: str,
    node_sizes: list[int] | None = None,
    seed: int = 42,
    backend: str | None = None,
) -> ResourceFitTrainResult:
    """Train resource-fit classifier and regressor."""
    ensure_dir(out_dir)
    model_dir = out_dir / model_id
    ensure_dir(model_dir)

    if node_sizes is None:
        node_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    resolved_backend = backend or DEFAULT_BACKEND
    logger.info("ResourceFit training with backend=%s (lightgbm_available=%s)", resolved_backend, _HAS_LIGHTGBM)

    trace_df = pd.read_parquet(dataset_path)
    df = _prepare_resource_frame(trace_df, node_sizes)

    n = len(df)
    train_end = int(n * 0.8)
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]

    x_train = train_df[RESOURCE_FEATURES]
    x_test = test_df[RESOURCE_FEATURES]

    # Build estimators based on backend.
    def _make_classifier() -> Any:
        if resolved_backend == "lightgbm":
            if not _HAS_LIGHTGBM:
                raise ImportError(
                    "LightGBM backend requested but lightgbm is not installed. "
                    "Install with: pip install 'hpc-workload-optimizer[lightgbm]'"
                )
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                num_leaves=31, subsample=0.8, random_state=seed, verbose=-1,
            )
        return GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=seed, subsample=0.8)

    def _make_regressor() -> Any:
        if resolved_backend == "lightgbm":
            if not _HAS_LIGHTGBM:
                raise ImportError(
                    "LightGBM backend requested but lightgbm is not installed. "
                    "Install with: pip install 'hpc-workload-optimizer[lightgbm]'"
                )
            return lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                num_leaves=31, subsample=0.8, random_state=seed, verbose=-1,
            )
        return GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=seed, subsample=0.8)

    # Train fragmentation classifier
    y_frag_train = train_df["frag_label"].to_numpy(dtype=int)
    y_frag_test = test_df["frag_label"].to_numpy(dtype=int)

    frag_pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor()),
            ("classifier", _make_classifier()),
        ]
    )
    frag_pipeline.fit(x_train, y_frag_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        frag_accuracy = float(np.mean(frag_pipeline.predict(x_test) == y_frag_test))

    # Train node size regressor
    y_node_train = train_df["optimal_node_cpus"].to_numpy(dtype=float)
    y_node_test = test_df["optimal_node_cpus"].to_numpy(dtype=float)

    node_pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor()),
            ("regressor", _make_regressor()),
        ]
    )
    node_pipeline.fit(x_train, y_node_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        node_pred = node_pipeline.predict(x_test)
    node_mae = float(np.mean(np.abs(y_node_test - node_pred)))

    # Save models
    frag_path = model_dir / "fragmentation_classifier.joblib"
    node_path = model_dir / "node_size_regressor.joblib"
    joblib.dump(frag_pipeline, frag_path)
    joblib.dump(node_pipeline, node_path)

    metrics = {
        "model_id": model_id,
        "backend": resolved_backend,
        "fragmentation_accuracy": frag_accuracy,
        "node_size_mae": node_mae,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "node_sizes": node_sizes,
        "seed": seed,
    }
    metadata = {
        "model_id": model_id,
        "model_type": "resource_fit",
        "backend": resolved_backend,
        "features": RESOURCE_FEATURES,
        "node_sizes": node_sizes,
        "seed": seed,
    }

    metrics_path = model_dir / "metrics.json"
    metadata_path = model_dir / "metadata.json"
    write_json(metrics_path, metrics)
    write_json(metadata_path, metadata)

    return ResourceFitTrainResult(
        model_dir=model_dir,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )


class ResourceFitPredictor:
    """Predict optimal node size and fragmentation risk."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        frag_path = model_dir / "fragmentation_classifier.joblib"
        node_path = model_dir / "node_size_regressor.joblib"
        if not frag_path.exists() or not node_path.exists():
            raise FileNotFoundError(f"Resource-fit model artifacts missing in {model_dir}")
        self.frag_classifier = joblib.load(frag_path)
        self.node_regressor = joblib.load(node_path)

    def predict(self, features: dict[str, Any]) -> ResourceFitPrediction:
        row = {key: features.get(key) for key in RESOURCE_FEATURES}
        frame = pd.DataFrame([row], columns=RESOURCE_FEATURES)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            frag_pred = int(self.frag_classifier.predict(frame)[0])
            frag_proba = self.frag_classifier.predict_proba(frame)[0]
            node_pred = float(self.node_regressor.predict(frame)[0])
        confidence = float(max(frag_proba))
        recommended = max(1, int(round(node_pred)))

        frag_labels = {0: "low", 1: "medium", 2: "high"}
        return ResourceFitPrediction(
            recommended_node_cpus=recommended,
            fragmentation_risk=frag_labels.get(frag_pred, "medium"),
            confidence=confidence,
            fallback_used=False,
        )
