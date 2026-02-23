from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from hpcopt.api.app import app
from hpcopt.api.rate_limit import (
    reset_for_testing,
    restore_limits_for_testing,
    set_limits_for_testing,
)


client = TestClient(app)


def test_rate_limit_error_contract() -> None:
    reset_for_testing()
    old_limit, old_per_endpoint = set_limits_for_testing(global_limit=1, per_endpoint={})
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
        restore_limits_for_testing(old_limit, old_per_endpoint)
        reset_for_testing()


def test_request_timeout_returns_504(monkeypatch) -> None:
    """Requests exceeding the timeout return 504 GATEWAY_TIMEOUT."""
    import hpcopt.api.app as app_module

    monkeypatch.setattr(app_module, "_REQUEST_TIMEOUT_SEC", 0.001)

    async def _slow_handler(*args, **kwargs):
        await asyncio.sleep(5)

    with patch("hpcopt.api.app.get_runtime_predictor", side_effect=_slow_handler):
        response = client.post(
            "/v1/runtime/predict",
            json={"requested_cpus": 2, "requested_runtime_sec": 60},
        )
    assert response.status_code == 504
    payload = response.json()
    assert payload["error"]["code"] == "GATEWAY_TIMEOUT"
    assert payload["error"]["message"] == "Request timed out"
    assert "trace_id" in payload["error"]
    assert "X-Trace-ID" in response.headers


def test_validation_error_headers() -> None:
    response = client.post("/v1/runtime/predict", json={"requested_cpus": 0})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"] == "Request validation failed"
    assert response.headers["X-Trace-ID"] == response.headers["X-Correlation-ID"]
