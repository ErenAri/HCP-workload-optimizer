"""Hyperparameter tuning for quantile models: Optuna or random search with chronological CV."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hpcopt.models.runtime_quantile import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    _pinball_loss,
    _prepare_training_frame,
)
from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass
class HyperParams:
    """Hyperparameters for gradient boosting quantile regression."""

    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    min_samples_leaf: int = 10
    min_samples_split: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HyperParams:
        return cls(
            n_estimators=int(d.get("n_estimators", 120)),
            learning_rate=float(d.get("learning_rate", 0.05)),
            max_depth=int(d.get("max_depth", 3)),
            subsample=float(d.get("subsample", 0.8)),
            min_samples_leaf=int(d.get("min_samples_leaf", 10)),
            min_samples_split=int(d.get("min_samples_split", 20)),
        )


@dataclass(frozen=True)
class TuningResult:
    report_path: Path
    best_params: HyperParams
    best_score: float
    payload: dict[str, Any]


# Default hyperparameter ranges for search.
N_ESTIMATORS_CHOICES = [50, 80, 100, 120, 150, 200, 300]
LEARNING_RATE_CHOICES = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]
MAX_DEPTH_CHOICES = [2, 3, 4, 5, 6]
SUBSAMPLE_CHOICES = [0.6, 0.7, 0.8, 0.9, 1.0]
MIN_SAMPLES_LEAF_CHOICES = [5, 10, 20, 30]
MIN_SAMPLES_SPLIT_CHOICES = [10, 20, 30, 50]
SEARCH_SPACE = {
    "n_estimators": N_ESTIMATORS_CHOICES,
    "learning_rate": LEARNING_RATE_CHOICES,
    "max_depth": MAX_DEPTH_CHOICES,
    "subsample": SUBSAMPLE_CHOICES,
    "min_samples_leaf": MIN_SAMPLES_LEAF_CHOICES,
    "min_samples_split": MIN_SAMPLES_SPLIT_CHOICES,
}


def _build_pipeline_with_params(alpha: float, seed: int, params: HyperParams) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    estimator = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        random_state=seed,
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        subsample=params.subsample,
        min_samples_leaf=params.min_samples_leaf,
        min_samples_split=params.min_samples_split,
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", estimator)])


def _chronological_cv_score(
    df: pd.DataFrame,
    params: HyperParams,
    alpha: float,
    seed: int,
    n_folds: int = 3,
) -> float:
    """Evaluate params using chronological cross-validation. Returns mean pinball loss."""
    n = len(df)
    fold_size = n // (n_folds + 1)
    if fold_size < 10:
        raise ValueError("Too few rows for chronological CV")

    scores = []
    for fold_idx in range(n_folds):
        train_end = fold_size * (fold_idx + 1)
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        if val_end <= val_start:
            continue

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]

        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
        x_val = val_df[FEATURE_COLUMNS]
        y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)

        pipeline = _build_pipeline_with_params(alpha=alpha, seed=seed, params=params)
        pipeline.fit(x_train, y_train)
        pred = np.maximum(pipeline.predict(x_val), 1.0)
        loss = _pinball_loss(y_val, pred, alpha=alpha)
        scores.append(loss)

    return float(np.mean(scores)) if scores else float("inf")


def _random_search(
    df: pd.DataFrame,
    alpha: float,
    seed: int,
    n_trials: int = 20,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    """Random search over hyperparameter space."""
    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []

    for trial_idx in range(n_trials):
        params = HyperParams(
            n_estimators=rng.choice(N_ESTIMATORS_CHOICES),
            learning_rate=rng.choice(LEARNING_RATE_CHOICES),
            max_depth=rng.choice(MAX_DEPTH_CHOICES),
            subsample=rng.choice(SUBSAMPLE_CHOICES),
            min_samples_leaf=rng.choice(MIN_SAMPLES_LEAF_CHOICES),
            min_samples_split=rng.choice(MIN_SAMPLES_SPLIT_CHOICES),
        )
        try:
            score = _chronological_cv_score(
                df=df, params=params, alpha=alpha, seed=seed, n_folds=n_folds
            )
            trials.append(
                {
                    "trial": trial_idx,
                    "params": params.to_dict(),
                    "score": score,
                    "status": "ok",
                }
            )
            logger.info("Trial %d: score=%.6f params=%s", trial_idx, score, params.to_dict())
        except Exception as exc:
            trials.append(
                {
                    "trial": trial_idx,
                    "params": params.to_dict(),
                    "score": float("inf"),
                    "status": "error",
                    "error": str(exc),
                }
            )
    return trials


def _optuna_search(
    df: pd.DataFrame,
    alpha: float,
    seed: int,
    n_trials: int = 20,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    """Optuna-based search if available."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.info("optuna not installed; falling back to random search")
        return _random_search(df, alpha, seed, n_trials, n_folds)

    trials_log: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = HyperParams(
            n_estimators=trial.suggest_categorical("n_estimators", N_ESTIMATORS_CHOICES),
            learning_rate=trial.suggest_categorical(
                "learning_rate",
                LEARNING_RATE_CHOICES,
            ),
            max_depth=trial.suggest_categorical("max_depth", MAX_DEPTH_CHOICES),
            subsample=trial.suggest_categorical("subsample", SUBSAMPLE_CHOICES),
            min_samples_leaf=trial.suggest_categorical(
                "min_samples_leaf",
                MIN_SAMPLES_LEAF_CHOICES,
            ),
            min_samples_split=trial.suggest_categorical(
                "min_samples_split",
                MIN_SAMPLES_SPLIT_CHOICES,
            ),
        )
        score = _chronological_cv_score(
            df=df, params=params, alpha=alpha, seed=seed, n_folds=n_folds
        )
        trials_log.append(
            {
                "trial": trial.number,
                "params": params.to_dict(),
                "score": score,
                "status": "ok",
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return trials_log


def tune_quantile_hyperparams(
    dataset_path: Path,
    quantile: float = 0.5,
    seed: int = 42,
    n_trials: int = 20,
    n_folds: int = 3,
    use_optuna: bool = True,
) -> tuple[HyperParams, list[dict[str, Any]]]:
    """Tune hyperparameters for a quantile model."""
    trace_df = pd.read_parquet(dataset_path)
    df = _prepare_training_frame(trace_df)

    if use_optuna:
        trials = _optuna_search(df, alpha=quantile, seed=seed, n_trials=n_trials, n_folds=n_folds)
    else:
        trials = _random_search(df, alpha=quantile, seed=seed, n_trials=n_trials, n_folds=n_folds)

    ok_trials = [t for t in trials if t["status"] == "ok"]
    if not ok_trials:
        logger.warning("No successful trials; returning default params")
        return HyperParams(), trials

    best = min(ok_trials, key=lambda t: t["score"])
    best_params = HyperParams.from_dict(best["params"])
    return best_params, trials


def build_tuning_report(
    dataset_path: Path,
    out_path: Path,
    quantile: float = 0.5,
    seed: int = 42,
    n_trials: int = 20,
    n_folds: int = 3,
    use_optuna: bool = True,
) -> TuningResult:
    """Run tuning and produce structured report."""
    ensure_dir(out_path.parent)

    best_params, trials = tune_quantile_hyperparams(
        dataset_path=dataset_path,
        quantile=quantile,
        seed=seed,
        n_trials=n_trials,
        n_folds=n_folds,
        use_optuna=use_optuna,
    )

    ok_trials = [t for t in trials if t["status"] == "ok"]
    best_score = min((t["score"] for t in ok_trials), default=float("inf"))

    payload = {
        "dataset_path": str(dataset_path),
        "quantile": quantile,
        "seed": seed,
        "n_trials_requested": n_trials,
        "n_trials_completed": len(ok_trials),
        "n_folds": n_folds,
        "best_params": best_params.to_dict(),
        "best_score": best_score,
        "trials": trials,
        "default_params": HyperParams().to_dict(),
    }

    write_json(out_path, payload)
    return TuningResult(
        report_path=out_path,
        best_params=best_params,
        best_score=best_score,
        payload=payload,
    )
