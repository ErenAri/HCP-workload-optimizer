"""Tests for the API deprecation sunset mechanism.

Verifies that deprecated endpoints return proper headers when configured
in configs/api/deprecation.yaml.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_deprecation_headers_not_present_when_empty_config() -> None:
    """With no deprecated endpoints, no Sunset/Deprecation headers should appear."""
    from hpcopt.api import app as app_module

    # Temporarily clear any cached config
    original = app_module._DEPRECATION_ENTRIES
    app_module._DEPRECATION_ENTRIES = []

    client = TestClient(app_module.app)
    response = client.get("/health")

    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers

    app_module._DEPRECATION_ENTRIES = original


def test_deprecation_headers_present_when_configured() -> None:
    """When an endpoint is in the deprecation config, Sunset + Deprecation headers appear."""
    from hpcopt.api import app as app_module

    original = app_module._DEPRECATION_ENTRIES
    app_module._DEPRECATION_ENTRIES = [
        {
            "path_prefix": "/v1/runtime/predict",
            "deprecated_at": "2026-06-01",
            "sunset_at": "2026-12-01",
            "docs_url": "https://docs.hpcopt.dev/api/migration/v1-to-v2",
        }
    ]

    client = TestClient(app_module.app)
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 4},
    )

    assert response.headers.get("Deprecation") == "2026-06-01"
    assert response.headers.get("Sunset") == "2026-12-01"
    assert "successor-version" in response.headers.get("Link", "")

    app_module._DEPRECATION_ENTRIES = original


def test_no_deprecation_for_non_matching_endpoint() -> None:
    """Endpoints not in the deprecation config should not have sunset headers."""
    from hpcopt.api import app as app_module

    original = app_module._DEPRECATION_ENTRIES
    app_module._DEPRECATION_ENTRIES = [
        {
            "path_prefix": "/v1/resource-fit/predict",
            "deprecated_at": "2026-06-01",
            "sunset_at": "2026-12-01",
        }
    ]

    client = TestClient(app_module.app)
    response = client.get("/health")

    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers

    app_module._DEPRECATION_ENTRIES = original
