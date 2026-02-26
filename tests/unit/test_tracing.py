"""Tests for the optional OpenTelemetry tracing module."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _install_fake_otel(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    tracker: dict[str, object] = {"instrumented": False, "provider": None, "span_processors": 0}

    trace_mod = ModuleType("opentelemetry.trace")

    def set_tracer_provider(provider: object) -> None:
        tracker["provider"] = provider

    trace_mod.set_tracer_provider = set_tracer_provider  # type: ignore[attr-defined]

    class _FastAPIInstrumentor:
        @staticmethod
        def instrument_app(app: object) -> None:
            del app
            tracker["instrumented"] = True

    instr_mod = ModuleType("opentelemetry.instrumentation.fastapi")
    instr_mod.FastAPIInstrumentor = _FastAPIInstrumentor  # type: ignore[attr-defined]

    class _Resource:
        @staticmethod
        def create(payload: dict[str, str]) -> dict[str, str]:
            return payload

    resources_mod = ModuleType("opentelemetry.sdk.resources")
    resources_mod.Resource = _Resource  # type: ignore[attr-defined]

    class _TracerProvider:
        def __init__(self, resource: object):
            self.resource = resource

        def add_span_processor(self, processor: object) -> None:
            del processor
            tracker["span_processors"] = int(tracker["span_processors"]) + 1

    trace_sdk_mod = ModuleType("opentelemetry.sdk.trace")
    trace_sdk_mod.TracerProvider = _TracerProvider  # type: ignore[attr-defined]

    otlp_mod = ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

    class _OTLPSpanExporter:
        def __init__(self, endpoint: str):
            self.endpoint = endpoint

    otlp_mod.OTLPSpanExporter = _OTLPSpanExporter  # type: ignore[attr-defined]

    export_mod = ModuleType("opentelemetry.sdk.trace.export")

    class _BatchSpanProcessor:
        def __init__(self, exporter: object):
            self.exporter = exporter

    export_mod.BatchSpanProcessor = _BatchSpanProcessor  # type: ignore[attr-defined]

    root = ModuleType("opentelemetry")
    root.trace = trace_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "opentelemetry", root)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.trace", trace_mod)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.instrumentation.fastapi", instr_mod)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.resources", resources_mod)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.trace", trace_sdk_mod)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", otlp_mod)
    monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.trace.export", export_mod)

    return tracker


def test_init_tracing_no_otel() -> None:
    """When opentelemetry is not installed, init_tracing should no-op gracefully."""
    from hpcopt.api.tracing import init_tracing

    fake_app = MagicMock()
    # Should not raise even without opentelemetry packages
    init_tracing(fake_app)


def test_init_tracing_import_error_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpcopt.api.tracing import init_tracing

    original_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name.startswith("opentelemetry"):
            raise ImportError("missing otel")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    init_tracing(MagicMock())


def test_init_tracing_with_otel_console_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpcopt.api.tracing import init_tracing

    tracker = _install_fake_otel(monkeypatch)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "hpcopt-api-test")

    app = MagicMock()
    init_tracing(app)

    assert tracker["provider"] is not None
    assert tracker["instrumented"] is True
    assert tracker["span_processors"] == 0


def test_init_tracing_with_otlp_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpcopt.api.tracing import init_tracing

    tracker = _install_fake_otel(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")

    app = MagicMock()
    init_tracing(app)

    assert tracker["provider"] is not None
    assert tracker["instrumented"] is True
    assert tracker["span_processors"] == 1


def test_init_tracing_otlp_import_error_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from hpcopt.api import tracing

    tracker = _install_fake_otel(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delitem(__import__("sys").modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    monkeypatch.delitem(__import__("sys").modules, "opentelemetry.sdk.trace.export")

    app = MagicMock()
    with caplog.at_level("WARNING"):
        tracing.init_tracing(app)

    assert "OTLP exporter requested" in caplog.text
    assert tracker["instrumented"] is True
    assert tracker["span_processors"] == 0


def test_init_tracing_instrumentation_failure_is_handled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from hpcopt.api import tracing

    tracker = _install_fake_otel(monkeypatch)
    instr_mod = __import__("sys").modules["opentelemetry.instrumentation.fastapi"]

    class _BrokenInstrumentor:
        @staticmethod
        def instrument_app(app: object) -> None:
            del app
            raise RuntimeError("boom")

    setattr(instr_mod, "FastAPIInstrumentor", _BrokenInstrumentor)

    with caplog.at_level("WARNING"):
        tracing.init_tracing(MagicMock())

    assert "Failed to instrument FastAPI with OpenTelemetry" in caplog.text
    assert tracker["provider"] is not None
