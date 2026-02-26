from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from hpcopt.api.app import app
from hpcopt.api.model_cache import reset_for_testing as reset_model_cache
from hpcopt.api.rate_limit import reset_for_testing as reset_rate_limit
from hpcopt.models.runtime_quantile import train_runtime_quantile_models
from hpcopt.simulate.stress import generate_stress_scenario

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limits() -> None:
    reset_rate_limit()
    yield  # type: ignore[misc]
    reset_rate_limit()


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


def test_admin_log_level_valid() -> None:
    response = client.post("/v1/admin/log-level", json={"level": "DEBUG"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["level"] == "DEBUG"


def test_admin_log_level_invalid() -> None:
    response = client.post("/v1/admin/log-level", json={"level": "INVALID_LEVEL"})
    assert response.status_code == 400


def test_admin_rbac_non_admin_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-admin API key should be rejected for admin paths when keys are configured."""
    monkeypatch.setenv("HPCOPT_API_KEYS", "u,admin-a")
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    invalidate_api_keys_cache()

    response = client.post(
        "/v1/admin/log-level",
        json={"level": "DEBUG"},
        headers={"X-API-Key": "u"},
    )
    assert response.status_code == 403

    # Admin key should work
    response = client.post(
        "/v1/admin/log-level",
        json={"level": "DEBUG"},
        headers={"X-API-Key": "admin-a"},
    )
    assert response.status_code == 200

    # Cleanup
    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    invalidate_api_keys_cache()


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_ready_endpoint() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_extra_fields_rejected() -> None:
    """Unknown fields in request body should be rejected (extra='forbid')."""
    response = client.post(
        "/v1/runtime/predict",
        json={
            "requested_cpus": 4,
            "unknown_field": "should_fail",
        },
    )
    assert response.status_code == 422


def test_input_bounds_enforced() -> None:
    """Extreme values beyond bounds should be rejected."""
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 200_000},  # exceeds le=100_000
    )
    assert response.status_code == 422

    response = client.post(
        "/v1/resource-fit/predict",
        json={
            "requested_cpus": 4,
            "candidate_node_cpus": list(range(1, 1002)),  # exceeds max_length=1000
        },
    )
    assert response.status_code == 422


def test_recommendation_endpoint_not_found() -> None:
    response = client.get("/v1/recommendations/nonexistent_run")
    assert response.status_code == 404


def test_recommendation_endpoint_path_traversal() -> None:
    response = client.get("/v1/recommendations/../../etc/passwd")
    assert response.status_code in {400, 404}  # 404 if URL path collapses


def test_recommendation_endpoint_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    rec_dir = tmp_path / "recommendations"
    rec_dir.mkdir(parents=True)
    (rec_dir / "test_run.json").write_text(json.dumps({"run_id": "test_run", "recommendations": []}), encoding="utf-8")
    monkeypatch.setenv("HPCOPT_ARTIFACTS_DIR", str(tmp_path))

    response = client.get("/v1/recommendations/test_run")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "test_run"

    monkeypatch.delenv("HPCOPT_ARTIFACTS_DIR", raising=False)


def test_rfc7807_error_format() -> None:
    """Error responses should follow RFC 7807 Problem Details format."""
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": -1},  # violates ge=1
    )
    assert response.status_code == 422
    payload = response.json()
    assert "type" in payload
    assert "title" in payload
    assert "status" in payload
    assert "detail" in payload
    assert "instance" in payload
