"""Tests for the Prometheus metrics module."""
from __future__ import annotations

from unittest.mock import patch

from hpcopt.api.metrics import (
    get_metrics_response,
    is_available,
    record_auth_failure,
    record_cache_hit,
    record_fallback,
    record_model_load_duration,
    record_rate_limit_rejection,
    record_request_metrics,
    set_model_loaded,
    set_model_staleness,
)


def test_is_available() -> None:
    """Should return True when prometheus_client is installed."""
    result = is_available()
    assert isinstance(result, bool)


def test_record_request_metrics_no_error() -> None:
    """Recording metrics should not raise."""
    record_request_metrics("GET", "/health", 200, 0.05)


def test_record_fallback_no_error() -> None:
    record_fallback()


def test_record_rate_limit_rejection_no_error() -> None:
    record_rate_limit_rejection()


def test_record_auth_failure_no_error() -> None:
    record_auth_failure()


def test_record_cache_hit_no_error() -> None:
    record_cache_hit("model")


def test_record_model_load_duration_no_error() -> None:
    record_model_load_duration(0.5)


def test_set_model_loaded_no_error() -> None:
    set_model_loaded(True)
    set_model_loaded(False)


def test_set_model_staleness_no_error() -> None:
    set_model_staleness(3600.0)


def test_get_metrics_response_returns_string() -> None:
    response = get_metrics_response()
    assert isinstance(response, str)


def test_noop_when_prometheus_not_available() -> None:
    """All metric functions should be no-ops when prometheus_client is missing."""
    with patch("hpcopt.api.metrics._PROMETHEUS_AVAILABLE", False):
        # Reset metrics so _ensure_metrics returns False
        import hpcopt.api.metrics as m
        old_total = m._requests_total
        m._requests_total = None
        try:
            record_request_metrics("GET", "/test", 200, 0.01)
            record_fallback()
            record_rate_limit_rejection()
            record_auth_failure()
            record_cache_hit()
            record_model_load_duration(0.1)
            set_model_loaded(True)
            set_model_staleness(100.0)
            assert get_metrics_response() == ""
        finally:
            m._requests_total = old_total
