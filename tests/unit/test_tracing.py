"""Tests for the optional OpenTelemetry tracing module."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_init_tracing_no_otel() -> None:
    """When opentelemetry is not installed, init_tracing should no-op gracefully."""
    from hpcopt.api.tracing import init_tracing

    fake_app = MagicMock()
    # Should not raise even without opentelemetry packages
    init_tracing(fake_app)
