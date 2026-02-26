"""Tests for the API deprecation sunset mechanism.

Verifies that deprecated endpoints return proper headers when configured
in configs/api/deprecation.yaml.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from hpcopt.api.deprecation import set_entries_for_testing


def test_deprecation_headers_not_present_when_empty_config() -> None:
    """With no deprecated endpoints, no Sunset/Deprecation headers should appear."""
    from hpcopt.api.app import app

    old = set_entries_for_testing([])

    client = TestClient(app)
    response = client.get("/health")

    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers

    set_entries_for_testing(old)


def test_deprecation_headers_present_when_configured() -> None:
    """When an endpoint is in the deprecation config, Sunset + Deprecation headers appear."""
    from hpcopt.api.app import app

    old = set_entries_for_testing(
        [
            {
                "path_prefix": "/v1/runtime/predict",
                "deprecated_at": "2026-06-01",
                "sunset_at": "2026-12-01",
                "docs_url": "https://docs.hpcopt.dev/api/migration/v1-to-v2",
            }
        ]
    )

    client = TestClient(app)
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4},
    )

    assert response.headers.get("Deprecation") == "2026-06-01"
    assert response.headers.get("Sunset") == "2026-12-01"
    assert "successor-version" in response.headers.get("Link", "")

    set_entries_for_testing(old)


def test_no_deprecation_for_non_matching_endpoint() -> None:
    """Endpoints not in the deprecation config should not have sunset headers."""
    from hpcopt.api.app import app

    old = set_entries_for_testing(
        [
            {
                "path_prefix": "/v1/resource-fit/predict",
                "deprecated_at": "2026-06-01",
                "sunset_at": "2026-12-01",
            }
        ]
    )

    client = TestClient(app)
    response = client.get("/health")

    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers

    set_entries_for_testing(old)
