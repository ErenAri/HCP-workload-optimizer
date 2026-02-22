"""Optional OpenTelemetry instrumentation for the hpcopt API.

No-ops gracefully when ``opentelemetry`` packages are not installed.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def init_tracing(app: object) -> None:
    """Instrument *app* with OpenTelemetry if the SDK is available.

    Configuration is driven by standard OTEL env vars:
      - ``OTEL_SERVICE_NAME`` (default: ``hpcopt-api``)
      - ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default: none / console exporter)
    """
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        logger.debug("OpenTelemetry packages not installed; tracing disabled")
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "hpcopt-api")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Try OTLP exporter if endpoint is configured, otherwise console
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OpenTelemetry OTLP exporter configured: %s", otlp_endpoint)
        except ImportError:
            logger.warning("OTLP exporter requested but opentelemetry-exporter-otlp not installed")

    trace.set_tracer_provider(provider)

    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry FastAPI instrumentation enabled (service=%s)", service_name)
    except Exception:
        logger.warning("Failed to instrument FastAPI with OpenTelemetry", exc_info=True)
