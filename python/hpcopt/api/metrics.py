"""Prometheus metrics instrumentation for the hpcopt API.

All metric objects are created lazily so that the module can be imported
even when ``prometheus_client`` is not installed -- the feature simply
becomes a no-op.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Graceful import – metrics are entirely optional.
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric singletons (created once, reused for the process lifetime).
# ---------------------------------------------------------------------------

_requests_total: Optional["Counter"] = None
_fallback_total: Optional["Counter"] = None
_request_duration: Optional["Histogram"] = None
_model_loaded: Optional["Gauge"] = None
_model_staleness: Optional["Gauge"] = None


def _ensure_metrics() -> bool:
    """Lazily initialise metric objects.  Returns *True* when metrics are
    available and ready to use.
    """
    global _requests_total, _fallback_total, _request_duration
    global _model_loaded, _model_staleness

    if not _PROMETHEUS_AVAILABLE:
        return False

    if _requests_total is not None:
        # Already initialised.
        return True

    _requests_total = Counter(
        "hpcopt_requests_total",
        "Total HTTP requests handled by hpcopt",
        labelnames=["method", "endpoint", "status"],
    )

    _fallback_total = Counter(
        "hpcopt_fallback_total",
        "Number of times the fallback prediction path was used",
    )

    _request_duration = Histogram(
        "hpcopt_request_duration_seconds",
        "Histogram of request durations in seconds",
        labelnames=["endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _model_loaded = Gauge(
        "hpcopt_model_loaded",
        "Whether a prediction model is currently loaded (0 or 1)",
    )

    _model_staleness = Gauge(
        "hpcopt_model_staleness_seconds",
        "Seconds since the active model was last refreshed",
    )

    return True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_request_metrics(
    method: str,
    endpoint: str,
    status: int,
    duration: float,
) -> None:
    """Record metrics for a single HTTP request.

    Parameters
    ----------
    method:
        HTTP method (GET, POST, …).
    endpoint:
        Request path or route name.
    status:
        HTTP status code returned to the client.
    duration:
        Wall-clock time of the request in **seconds**.
    """
    if not _ensure_metrics():
        return
    assert _requests_total is not None
    assert _request_duration is not None
    _requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    _request_duration.labels(endpoint=endpoint).observe(duration)


def record_fallback() -> None:
    """Increment the fallback counter."""
    if not _ensure_metrics():
        return
    assert _fallback_total is not None
    _fallback_total.inc()


def set_model_loaded(loaded: bool) -> None:
    """Set the model-loaded gauge to 1 (loaded) or 0 (not loaded)."""
    if not _ensure_metrics():
        return
    assert _model_loaded is not None
    _model_loaded.set(1 if loaded else 0)


def set_model_staleness(seconds: float) -> None:
    """Update the model staleness gauge."""
    if not _ensure_metrics():
        return
    assert _model_staleness is not None
    _model_staleness.set(seconds)


def get_metrics_response() -> str:
    """Return all registered metrics in Prometheus text exposition format.

    Returns an empty string when ``prometheus_client`` is not installed.
    """
    if not _ensure_metrics():
        return ""
    payload = generate_latest()
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


def is_available() -> bool:
    """Return *True* when Prometheus metrics are functional."""
    return _PROMETHEUS_AVAILABLE
