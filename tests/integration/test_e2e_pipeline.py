"""End-to-end pipeline test: ingest -> features -> train -> predict via API."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_full_pipeline_ingest_to_prediction(tmp_path: Path, monkeypatch) -> None:
    """Run the complete pipeline from SWF ingestion through API prediction.

    Steps:
      1. Ingest a small SWF trace into canonical parquet.
      2. Build the feature dataset with chronological folds.
      3. Train runtime quantile models on that dataset.
      4. Point the API at the trained model and POST a predict request.
      5. Assert the response uses the real model (not fallback).
    """
    from hpcopt.ingest.swf import ingest_swf
    from hpcopt.features.pipeline import build_feature_dataset
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models

    # --- Step 1: Ingest SWF trace ---
    trace_fixture = Path(__file__).resolve().parent.parent / "fixtures" / "sample_trace.swf"
    if not trace_fixture.exists():
        pytest.skip(f"Missing fixture: {trace_fixture}")

    curated_dir = tmp_path / "curated"
    report_dir = tmp_path / "reports"
    ingest_result = ingest_swf(
        input_path=trace_fixture,
        out_dir=curated_dir,
        dataset_id="e2e_test",
        report_dir=report_dir,
    )
    assert ingest_result.dataset_path.exists()
    assert ingest_result.row_count > 0

    # --- Step 2: Build feature dataset ---
    feature_dir = tmp_path / "features"
    feature_report_dir = tmp_path / "feature_reports"
    feature_result = build_feature_dataset(
        dataset_path=ingest_result.dataset_path,
        out_dir=feature_dir,
        report_dir=feature_report_dir,
        dataset_id="e2e_test",
        n_folds=2,
    )
    assert feature_result.feature_dataset_path.exists()
    assert feature_result.row_count > 0

    # --- Step 3: Train runtime quantile models ---
    model_dir = tmp_path / "models"
    train_result = train_runtime_quantile_models(
        dataset_path=feature_result.feature_dataset_path,
        out_dir=model_dir,
        model_id="e2e_test_model",
        seed=42,
    )
    assert train_result.model_dir.exists()
    assert train_result.metrics_path.exists()

    # --- Step 4: Point API at trained model and predict ---
    monkeypatch.setenv("HPCOPT_RUNTIME_MODEL_DIR", str(train_result.model_dir))

    import hpcopt.api.app as api_module
    api_module._RUNTIME_PREDICTOR_CACHE["model_dir"] = None
    api_module._RUNTIME_PREDICTOR_CACHE["predictor"] = None

    client = TestClient(api_module.app)
    response = client.post(
        "/v1/runtime/predict",
        json={
            "requested_runtime_sec": 1200,
            "requested_cpus": 8,
            "queue_id": 1,
            "partition_id": 1,
            "user_id": 10,
            "group_id": 1,
            "runtime_guard_k": 0.5,
        },
    )

    # --- Step 5: Validate response ---
    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is False
    assert payload["runtime_p50_sec"] > 0
    assert payload["runtime_p90_sec"] >= payload["runtime_p50_sec"]
    assert payload["runtime_guard_sec"] >= payload["runtime_p50_sec"]
    assert payload["predictor_version"].startswith("runtime-quantile:")


@pytest.mark.integration
def test_pipeline_with_stress_data(tmp_path: Path, monkeypatch) -> None:
    """Verify the pipeline works with synthetic stress data (no fixture needed)."""
    from hpcopt.simulate.stress import generate_stress_scenario
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models

    # Generate stress data
    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path / "stress",
        n_jobs=150,
        seed=99,
        params={"alpha": 1.25},
    )
    assert stress.dataset_path.exists()

    # Train on stress data
    train_result = train_runtime_quantile_models(
        dataset_path=stress.dataset_path,
        out_dir=tmp_path / "models",
        model_id="stress_e2e",
        seed=7,
    )

    # Predict via API
    monkeypatch.setenv("HPCOPT_RUNTIME_MODEL_DIR", str(train_result.model_dir))

    import hpcopt.api.app as api_module
    api_module._RUNTIME_PREDICTOR_CACHE["model_dir"] = None
    api_module._RUNTIME_PREDICTOR_CACHE["predictor"] = None

    client = TestClient(api_module.app)
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 16, "runtime_guard_k": 0.75},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is False
    assert payload["runtime_p90_sec"] >= payload["runtime_p50_sec"]
