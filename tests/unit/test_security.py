"""Security-focused tests: auth edge cases, input validation, admin RBAC."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from hpcopt.api.app import app
from hpcopt.api.auth import check_admin_auth
from hpcopt.api.rate_limit import reset_for_testing as reset_rate_limit

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_rate_limit()
    yield  # type: ignore[misc]
    reset_rate_limit()


# --- Auth edge cases ---


def test_auth_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCOPT_API_KEYS", "a")
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    invalidate_api_keys_cache()

    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4},
        headers={"X-API-Key": ""},
    )
    assert response.status_code == 401

    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    invalidate_api_keys_cache()


def test_auth_very_long_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCOPT_API_KEYS", "a")
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    invalidate_api_keys_cache()

    long_key = "x" * 10_000
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4},
        headers={"X-API-Key": long_key},
    )
    assert response.status_code == 401

    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    invalidate_api_keys_cache()


def test_auth_key_with_null_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCOPT_API_KEYS", "a")
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    invalidate_api_keys_cache()

    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4},
        headers={"X-API-Key": "key\x00injection"},
    )
    assert response.status_code == 401

    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    invalidate_api_keys_cache()


# --- Admin RBAC ---


def test_admin_auth_non_admin_path() -> None:
    """Non-admin paths should always pass admin check."""
    assert check_admin_auth("/v1/runtime/predict", "user-key") is True


def test_admin_auth_no_keys_configured() -> None:
    """When no keys are configured, admin access is unrestricted."""
    assert check_admin_auth("/v1/admin/log-level", "") is True


def test_admin_auth_prefix_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCOPT_API_KEYS", "admin-a,u")
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    invalidate_api_keys_cache()

    assert check_admin_auth("/v1/admin/log-level", "admin-a") is True
    assert check_admin_auth("/v1/admin/log-level", "u") is False

    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    invalidate_api_keys_cache()


# --- Input validation bounds ---


def test_requested_cpus_max_bound() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 100_001},  # > 100_000
    )
    assert response.status_code == 422


def test_candidate_node_cpus_max_length() -> None:
    response = client.post(
        "/v1/resource-fit/predict",
        json={"requested_cpus": 4, "candidate_node_cpus": list(range(1, 1002))},
    )
    assert response.status_code == 422


def test_queue_depth_max_bound() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4, "queue_depth_jobs": 1_000_001},
    )
    assert response.status_code == 422


def test_runtime_requested_sec_max_bound() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4, "requested_runtime_sec": 31_536_001},
    )
    assert response.status_code == 422


def test_extra_fields_rejected_runtime() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4, "evil_field": "injected"},
    )
    assert response.status_code == 422


def test_extra_fields_rejected_resource_fit() -> None:
    response = client.post(
        "/v1/resource-fit/predict",
        json={"requested_cpus": 4, "candidate_node_cpus": [8, 16], "extra": True},
    )
    assert response.status_code == 422


# --- Body size limit ---


def test_oversized_body_rejected() -> None:
    """Request body exceeding 1MB should be rejected."""
    large_payload = '{"requested_cpus": 4, "data": "' + "x" * (1024 * 1024) + '"}'
    response = client.post(
        "/v1/runtime/predict",
        content=large_payload.encode(),
        headers={"Content-Type": "application/json", "Content-Length": str(len(large_payload))},
    )
    # Should be either 413 (body limit) or 422 (extra field)
    assert response.status_code in {413, 422}
