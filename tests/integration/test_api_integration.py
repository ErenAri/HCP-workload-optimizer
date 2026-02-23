"""Integration tests for the API: full flow, auth, rate limiting, health."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from hpcopt.api.app import app


@pytest.fixture
def client() -> TestClient:
    from hpcopt.utils.secrets import invalidate_api_keys_cache
    invalidate_api_keys_cache()
    yield TestClient(app)
    invalidate_api_keys_cache()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ready_endpoint(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")


def test_runtime_predict_fallback(client: TestClient) -> None:
    payload = {
        "requested_cpus": 4,
        "requested_runtime_sec": 3600,
    }
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=None):
        response = client.post("/v1/runtime/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_used"] is True
    assert data["runtime_p50_sec"] > 0
    assert data["runtime_p90_sec"] >= data["runtime_p50_sec"]
    assert response.headers["X-Fallback-Used"] == "true"
    assert "X-Trace-ID" in response.headers


def test_runtime_predict_response_headers(client: TestClient) -> None:
    payload = {
        "requested_cpus": 8,
        "requested_runtime_sec": 1200,
    }
    response = client.post("/v1/runtime/predict", json=payload)
    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == response.headers["X-Trace-ID"]
    assert "X-Model-Version" in response.headers
    assert response.headers["X-Fallback-Used"] in ("true", "false")


def test_resource_fit_predict(client: TestClient) -> None:
    payload = {
        "requested_cpus": 12,
        "candidate_node_cpus": [8, 16, 32, 64],
    }
    response = client.post("/v1/resource-fit/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fragmentation_risk" in data
    assert data["fragmentation_risk"] in ("low", "medium", "high")


def test_auth_enforcement() -> None:
    """Test that auth middleware blocks requests when HPCOPT_API_KEYS is set."""
    with patch.dict(os.environ, {"HPCOPT_API_KEYS": "a,b"}):
        client = TestClient(app)

        # Health should be exempt
        response = client.get("/health")
        assert response.status_code == 200

        # Predict without key should fail
        payload = {"requested_cpus": 4}
        response = client.post(
            "/v1/runtime/predict",
            json=payload,
        )
        assert response.status_code == 401
        body = response.json()
        assert body["title"] == "UNAUTHORIZED"
        assert "instance" in body  # trace_id in RFC 7807
        assert "X-Trace-ID" in response.headers


def test_runtime_predict_validation(client: TestClient) -> None:
    """Test that invalid input returns 422."""
    payload = {"requested_cpus": -1}
    response = client.post("/v1/runtime/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["title"] == "VALIDATION_ERROR"
    assert "instance" in body  # trace_id in RFC 7807
