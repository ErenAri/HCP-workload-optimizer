from fastapi.testclient import TestClient

from hpcopt.api.app import app
from hpcopt.api.model_cache import reset_for_testing as reset_model_cache
from hpcopt.models.runtime_quantile import train_runtime_quantile_models
from hpcopt.simulate.stress import generate_stress_scenario


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "hpcopt-api"
    assert "X-Trace-ID" in response.headers


def test_runtime_predict_endpoint() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={
            "requested_runtime_sec": 1200,
            "requested_cpus": 8,
            "queue_depth_jobs": 100,
            "runtime_guard_k": 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_p50_sec"] > 0
    assert payload["runtime_p90_sec"] >= payload["runtime_p50_sec"]
    assert payload["runtime_guard_sec"] >= payload["runtime_p50_sec"]
    assert "X-Trace-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == response.headers["X-Trace-ID"]
    assert "X-Model-Version" in response.headers
    assert response.headers["X-Fallback-Used"] in {"true", "false"}


def test_resource_fit_endpoint() -> None:
    response = client.post(
        "/v1/resource-fit/predict",
        json={
            "requested_cpus": 10,
            "candidate_node_cpus": [8, 16, 32],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation"]["recommended_node_cpus"] == 16


def test_system_status_endpoint() -> None:
    response = client.get("/v1/system/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "hpcopt-api"
    assert "uptime_seconds" in payload
    assert isinstance(payload["shutdown_requested"], bool)


def test_runtime_predict_uses_trained_model_when_available(tmp_path, monkeypatch) -> None:
    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=120,
        seed=12,
        params={"alpha": 1.25},
    )
    train = train_runtime_quantile_models(
        dataset_path=stress.dataset_path,
        out_dir=tmp_path / "models",
        model_id="api_runtime_model",
        seed=3,
    )
    monkeypatch.setenv("HPCOPT_RUNTIME_MODEL_DIR", str(train.model_dir))
    reset_model_cache()

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
    assert response.status_code == 200
    payload = response.json()
    assert payload["fallback_used"] is False
    assert payload["predictor_version"].startswith("runtime-quantile:")
