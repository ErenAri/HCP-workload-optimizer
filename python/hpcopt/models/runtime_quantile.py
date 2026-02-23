from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)

# Time-based split ratios for train/valid/test partitioning
TIME_SPLIT_TRAIN_RATIO = 0.7
TIME_SPLIT_VALID_END_RATIO = 0.85
# Minimum floor for quantile predictions (seconds)
MIN_PREDICTION_SEC = 1.0

TARGET_COLUMN = "runtime_actual_sec"
FEATURE_COLUMNS = [
    "requested_cpus",
    "runtime_requested_sec",
    "requested_mem",
    "queue_id",
    "partition_id",
    "user_id",
    "group_id",
    "submit_hour",
    "submit_dow",
]

NUMERIC_FEATURES = ["requested_cpus", "runtime_requested_sec", "requested_mem"]
CATEGORICAL_FEATURES = [
    "queue_id",
    "partition_id",
    "user_id",
    "group_id",
    "submit_hour",
    "submit_dow",
]
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


@dataclass
class RuntimeTrainResult:
    model_dir: Path
    metrics_path: Path
    metadata_path: Path


def _prepare_training_frame(trace_df: pd.DataFrame) -> pd.DataFrame:
    required = {"submit_ts", "runtime_actual_sec", "requested_cpus"}
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(f"training dataset missing required columns: {sorted(missing)}")

    df = trace_df.copy()
    for col in ["runtime_requested_sec", "requested_mem", "queue_id", "partition_id", "user_id", "group_id"]:
        if col not in df.columns:
            df[col] = None

    df["submit_ts"] = pd.to_numeric(df["submit_ts"], errors="coerce")
    df["runtime_actual_sec"] = pd.to_numeric(df["runtime_actual_sec"], errors="coerce")
    df["requested_cpus"] = pd.to_numeric(df["requested_cpus"], errors="coerce")
    df["runtime_requested_sec"] = pd.to_numeric(df["runtime_requested_sec"], errors="coerce")
    df["requested_mem"] = pd.to_numeric(df["requested_mem"], errors="coerce")

    df = df.dropna(subset=["submit_ts", "runtime_actual_sec", "requested_cpus"])
    if df.empty:
        raise ValueError("training dataset is empty after null filtering.")

    df["submit_ts"] = df["submit_ts"].astype(int)
    df["runtime_actual_sec"] = df["runtime_actual_sec"].clip(lower=1).astype(float)
    df["requested_cpus"] = df["requested_cpus"].clip(lower=1).astype(float)
    df["runtime_requested_sec"] = df["runtime_requested_sec"].where(df["runtime_requested_sec"] > 0)
    df["requested_mem"] = df["requested_mem"].where(df["requested_mem"] > 0)
    if df["requested_mem"].notna().sum() == 0:
        df["requested_mem"] = 0.0

    submit_dt = pd.to_datetime(df["submit_ts"], unit="s", utc=True)
    df["submit_hour"] = submit_dt.dt.hour
    df["submit_dow"] = submit_dt.dt.dayofweek

    df = df.sort_values(["submit_ts", "requested_cpus"]).reset_index(drop=True)
    return df


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    delta = y_true - y_pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _build_pipeline(
    alpha: float,
    seed: int,
    hyperparams: dict[str, Any] | None = None,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    hp = hyperparams or {}
    estimator = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        random_state=seed,
        n_estimators=int(hp.get("n_estimators", 120)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        max_depth=int(hp.get("max_depth", 3)),
        subsample=float(hp.get("subsample", 0.8)),
        min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
        min_samples_split=int(hp.get("min_samples_split", 20)),
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", estimator)])


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 30:
        # Keep splits valid even for tiny fixtures.
        train_end = max(int(n * 0.8), 1)
        valid_end = max(int(n * 0.9), train_end + 1)
    else:
        train_end = int(n * TIME_SPLIT_TRAIN_RATIO)
        valid_end = int(n * TIME_SPLIT_VALID_END_RATIO)
    valid_end = min(valid_end, n)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]
    if valid.empty:
        valid = train.copy()
    if test.empty:
        test = valid.copy()
    return train, valid, test


def train_runtime_quantile_models(
    dataset_path: Path,
    out_dir: Path,
    model_id: str,
    seed: int = 42,
    hyperparams: dict[str, Any] | None = None,
) -> RuntimeTrainResult:
    ensure_dir(out_dir)
    model_dir = out_dir / model_id
    ensure_dir(model_dir)

    trace_df = pd.read_parquet(dataset_path)
    df = _prepare_training_frame(trace_df)
    train_df, valid_df, test_df = _time_split(df)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    metrics: dict[str, Any] = {
        "model_id": model_id,
        "dataset_path": str(dataset_path),
        "seed": int(seed),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "rows_test": int(len(test_df)),
        "quantiles": {},
    }

    trained_models: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for quantile_name, quantile in QUANTILES.items():
        pipeline = _build_pipeline(alpha=quantile, seed=seed, hyperparams=hyperparams)
        pipeline.fit(x_train, y_train)
        pred_test = pipeline.predict(x_test)
        pred_test = np.maximum(pred_test, MIN_PREDICTION_SEC)

        pinball = _pinball_loss(y_test, pred_test, alpha=quantile)
        mae = float(np.mean(np.abs(y_test - pred_test)))

        trained_models[quantile_name] = pipeline
        test_predictions[quantile_name] = pred_test
        metrics["quantiles"][quantile_name] = {
            "alpha": quantile,
            "pinball_loss": float(pinball),
            "mae": float(mae),
        }

        model_path = model_dir / f"{quantile_name}.joblib"
        joblib.dump(pipeline, model_path)

    p10 = test_predictions["p10"]
    p90 = test_predictions["p90"]
    coverage = np.mean((y_test >= p10) & (y_test <= p90))
    metrics["interval_coverage_p10_p90"] = float(coverage)

    # Naive comparators for policy-credible lift reporting.
    global_median = float(np.median(y_train))
    global_mean = float(np.mean(y_train))

    train_users = pd.to_numeric(train_df["user_id"], errors="coerce").fillna(-1).astype(int)
    test_users = pd.to_numeric(test_df["user_id"], errors="coerce").fillna(-1).astype(int)
    user_runtime_map = (
        pd.DataFrame({"user_id": train_users, "runtime_actual_sec": y_train})
        .groupby("user_id", as_index=True)["runtime_actual_sec"]
        .median()
        .to_dict()
    )

    baseline_preds: dict[str, np.ndarray] = {
        "global_median": np.full(shape=y_test.shape, fill_value=global_median, dtype=float),
        "global_mean": np.full(shape=y_test.shape, fill_value=global_mean, dtype=float),
        "user_history_median": np.array(
            [float(user_runtime_map.get(int(uid), global_median)) for uid in test_users],
            dtype=float,
        ),
    }

    metrics["naive_baselines"] = {}
    for baseline_name, pred in baseline_preds.items():
        metrics["naive_baselines"][baseline_name] = {
            "mae": float(np.mean(np.abs(y_test - pred))),
            "mape": _safe_mape(y_test, pred),
            "pinball_loss_p50": _pinball_loss(y_test, pred, alpha=0.5),
        }

    p50_mae = float(metrics["quantiles"]["p50"]["mae"])
    p50_pinball = float(metrics["quantiles"]["p50"]["pinball_loss"])
    metrics["p50_lift_vs_naive"] = {
        baseline_name: {
            "mae_improvement": float(payload["mae"] - p50_mae),
            "pinball_p50_improvement": float(payload["pinball_loss_p50"] - p50_pinball),
        }
        for baseline_name, payload in metrics["naive_baselines"].items()
    }
    metrics["timestamp_utc"] = dt.datetime.now(tz=dt.UTC).isoformat()

    model_hashes: dict[str, str] = {}
    for quantile_name in QUANTILES:
        artifact = model_dir / f"{quantile_name}.joblib"
        if artifact.exists():
            digest = hashlib.sha256()
            with artifact.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_hashes[artifact.name] = digest.hexdigest()

    metadata = {
        "model_id": model_id,
        "model_type": "gradient_boosting_quantile",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "quantiles": QUANTILES,
        "seed": int(seed),
        "trained_at_utc": metrics["timestamp_utc"],
        "dataset_path": str(dataset_path),
        "model_hashes": model_hashes,
    }

    metrics_path = model_dir / "metrics.json"
    metadata_path = model_dir / "metadata.json"
    write_json(metrics_path, metrics)
    write_json(metadata_path, metadata)

    latest_pointer = out_dir / "runtime_latest.json"
    write_json(
        latest_pointer,
        {
            "model_id": model_id,
            "model_dir": str(model_dir),
            "updated_at_utc": metrics["timestamp_utc"],
        },
    )

    # Generate model card
    try:
        from hpcopt.models.model_card import generate_model_card
        generate_model_card(
            model_dir=model_dir,
            dataset_path=dataset_path,
            metrics=metrics,
            metadata=metadata,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )
    except (ImportError, OSError) as exc:
        logger.warning("Could not generate model card: %s", exc)

    return RuntimeTrainResult(model_dir=model_dir, metrics_path=metrics_path, metadata_path=metadata_path)


def resolve_runtime_model_dir(explicit_model_dir: Path | None = None) -> Path | None:
    if explicit_model_dir is not None and explicit_model_dir.exists():
        return explicit_model_dir

    env_model_dir = os.getenv("HPCOPT_RUNTIME_MODEL_DIR")
    if env_model_dir:
        candidate = Path(env_model_dir)
        if candidate.exists():
            return candidate

    latest_pointer = Path("outputs/models/runtime_latest.json")
    if latest_pointer.exists():
        try:
            latest = json.loads(latest_pointer.read_text(encoding="utf-8"))
            candidate = Path(str(latest["model_dir"]))
            if candidate.exists():
                return candidate
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not resolve runtime model from latest pointer: %s", exc)
            return None
    return None


def _verify_model_hash(model_path: Path, expected_hashes: dict[str, str] | None) -> None:
    """Verify model file hash against metadata before loading."""
    if expected_hashes is None:
        return
    expected = expected_hashes.get(model_path.name)
    if expected is None:
        return
    digest = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Model hash mismatch for {model_path.name}: "
            f"expected={expected}, actual={actual}"
        )


def _load_model_hashes(model_dir: Path) -> dict[str, str] | None:
    """Load expected model hashes from metadata.json if available."""
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hashes = metadata.get("model_hashes") if isinstance(metadata, dict) else None
        if not isinstance(hashes, dict):
            return None
        out: dict[str, str] = {}
        for key, value in hashes.items():
            if isinstance(key, str) and isinstance(value, str):
                out[key] = value
        return out or None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load model metadata for hash verification: %s", exc)
        return None


class RuntimeQuantilePredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models: dict[str, Pipeline] = {}
        expected_hashes = _load_model_hashes(model_dir)
        for quantile_name in QUANTILES:
            model_path = model_dir / f"{quantile_name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"missing model artifact: {model_path}")
            _verify_model_hash(model_path, expected_hashes)
            self.models[quantile_name] = joblib.load(model_path)

    def predict_one(self, features: dict[str, Any]) -> dict[str, float]:
        row = {key: features.get(key) for key in FEATURE_COLUMNS}
        if row.get("submit_hour") is None or row.get("submit_dow") is None:
            submit_ts = features.get("submit_ts")
            if submit_ts is not None:
                ts = pd.to_datetime(int(submit_ts), unit="s", utc=True)
            else:
                ts = pd.Timestamp.now(tz="UTC")
            row["submit_hour"] = int(ts.hour)
            row["submit_dow"] = int(ts.dayofweek)
        frame = pd.DataFrame({k: [v] for k, v in row.items()})

        p10 = float(max(MIN_PREDICTION_SEC, self.models["p10"].predict(frame)[0]))
        p50 = float(max(MIN_PREDICTION_SEC, self.models["p50"].predict(frame)[0]))
        p90 = float(max(MIN_PREDICTION_SEC, self.models["p90"].predict(frame)[0]))

        # Enforce monotonic quantiles for control safety.
        ordered = sorted([p10, p50, p90])
        return {"p10": ordered[0], "p50": ordered[1], "p90": ordered[2]}
