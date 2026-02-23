"""Tests for the resource-fit model: training and prediction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hpcopt.models.resource_fit import (
    ResourceFitPredictor,
    train_resource_fit_model,
)


def _make_resource_dataset(tmp_path: Path, n_rows: int = 200) -> Path:
    """Create a minimal parquet dataset for resource-fit training.

    Uses CPU values that span low/medium/high fragmentation labels
    (waste ratio relative to best-fit node).  Values like 3, 5, 6, 10
    intentionally don't match standard node sizes to create waste.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    # Include non-power-of-2 values to ensure multiple fragmentation classes
    cpu_choices = [1, 2, 3, 4, 5, 6, 8, 10, 16, 24, 32]
    df = pd.DataFrame({
        "job_id": range(n_rows),
        "submit_ts": rng.integers(1_700_000_000, 1_700_100_000, size=n_rows),
        "requested_cpus": rng.choice(cpu_choices, size=n_rows),
        "allocated_cpus": rng.choice(cpu_choices, size=n_rows),
        "runtime_actual_sec": rng.integers(60, 7200, size=n_rows),
        "runtime_requested_sec": rng.integers(120, 14400, size=n_rows),
        "requested_mem": rng.integers(100, 8000, size=n_rows),
        "queue_id": rng.integers(0, 3, size=n_rows),
        "partition_id": rng.integers(0, 2, size=n_rows),
        "user_id": rng.integers(0, 10, size=n_rows),
    })
    path = tmp_path / "resource_trace.parquet"
    df.to_parquet(path, index=False)
    return path


def test_train_resource_fit_model(tmp_path: Path) -> None:
    dataset_path = _make_resource_dataset(tmp_path)
    result = train_resource_fit_model(
        dataset_path=dataset_path,
        out_dir=tmp_path / "models",
        model_id="rf_test",
    )
    assert result.model_dir.exists()
    assert result.metrics_path.exists()
    assert result.metadata_path.exists()
    assert (result.model_dir / "fragmentation_classifier.joblib").exists()
    assert (result.model_dir / "node_size_regressor.joblib").exists()


def test_resource_fit_prediction(tmp_path: Path) -> None:
    dataset_path = _make_resource_dataset(tmp_path)
    result = train_resource_fit_model(
        dataset_path=dataset_path,
        out_dir=tmp_path / "models",
        model_id="rf_pred",
    )
    predictor = ResourceFitPredictor(result.model_dir)
    prediction = predictor.predict({
        "requested_cpus": 8,
        "requested_mem": 4000,
        "runtime_requested_sec": 3600,
        "queue_id": 1,
        "partition_id": 0,
        "user_id": 5,
        "submit_hour": 10,
        "submit_dow": 2,
    })
    assert prediction.recommended_node_cpus >= 1
    assert prediction.fragmentation_risk in {"low", "medium", "high"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.fallback_used is False


def test_resource_fit_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ResourceFitPredictor(tmp_path / "nonexistent")


def test_fragmentation_classification(tmp_path: Path) -> None:
    """Test that fragmentation labels are assigned correctly."""
    dataset_path = _make_resource_dataset(tmp_path, n_rows=300)
    result = train_resource_fit_model(
        dataset_path=dataset_path,
        out_dir=tmp_path / "models",
        model_id="rf_frag",
        node_sizes=[4, 8, 16, 32, 64],
    )
    import json
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "fragmentation_accuracy" in metrics
    assert metrics["fragmentation_accuracy"] > 0.0
    assert metrics["node_size_mae"] >= 0.0
