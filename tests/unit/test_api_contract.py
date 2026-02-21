from __future__ import annotations

import hpcopt.api.app as api_module
from fastapi.testclient import TestClient

from hpcopt.api.app import app


client = TestClient(app)


def test_rate_limit_error_contract(monkeypatch) -> None:
    original_limit = api_module._RATE_LIMIT
    api_module._RATE_BUCKETS.clear()
    monkeypatch.setattr(api_module, "_RATE_LIMIT", 1)
    try:
        first = client.post(
            "/v1/runtime/predict",
            headers={"X-API-Key": "rate-test-key"},
            json={"requested_cpus": 2, "requested_runtime_sec": 60},
        )
        assert first.status_code == 200

        second = client.post(
            "/v1/runtime/predict",
            headers={"X-API-Key": "rate-test-key"},
            json={"requested_cpus": 2, "requested_runtime_sec": 60},
        )
        assert second.status_code == 429
        payload = second.json()
        assert payload["error"]["code"] == "RATE_LIMITED"
        assert payload["error"]["message"] == "Rate limit exceeded"
        assert "trace_id" in payload["error"]
        assert "Retry-After" in second.headers
        assert second.headers["X-Trace-ID"] == second.headers["X-Correlation-ID"]
    finally:
        monkeypatch.setattr(api_module, "_RATE_LIMIT", original_limit)
        api_module._RATE_BUCKETS.clear()


def test_validation_error_headers() -> None:
    response = client.post("/v1/runtime/predict", json={"requested_cpus": 0})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"] == "Request validation failed"
    assert response.headers["X-Trace-ID"] == response.headers["X-Correlation-ID"]

