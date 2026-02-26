"""Prometheus metrics exporter for HPC Workload Optimizer.

Exposes prediction accuracy, simulation metrics, and system health
as Prometheus-compatible metrics for Grafana dashboards.

Usage:
    # As standalone: python -m hpcopt.integrations.metrics_exporter
    # In FastAPI: app.mount("/metrics", metrics_app)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Simple Prometheus-compatible metrics registry.

    No external dependency required — generates Prometheus text format
    directly. Compatible with Grafana + Prometheus stack.
    """

    def __init__(self) -> None:
        self._gauges: dict[str, tuple[float, str, dict[str, str]]] = {}
        self._counters: dict[str, tuple[float, str, dict[str, str]]] = {}
        self._histograms: dict[str, tuple[list[float], str]] = {}

    def gauge(self, name: str, value: float, help_text: str = "", labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._label_key(name, labels)
        self._gauges[key] = (value, help_text, labels or {})

    def counter(self, name: str, value: float, help_text: str = "", labels: dict[str, str] | None = None) -> None:
        """Set a counter value."""
        key = self._label_key(name, labels)
        self._counters[key] = (value, help_text, labels or {})

    def observe_histogram(self, name: str, value: float, help_text: str = "") -> None:
        """Record a histogram observation."""
        if name not in self._histograms:
            self._histograms[name] = ([], help_text)
        self._histograms[name][0].append(value)

    def _label_key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        seen_help: set[str] = set()

        # Gauges
        for key, (value, help_text, labels) in sorted(self._gauges.items()):
            base_name = key.split("{")[0]
            if base_name not in seen_help and help_text:
                lines.append(f"# HELP {base_name} {help_text}")
                lines.append(f"# TYPE {base_name} gauge")
                seen_help.add(base_name)
            lines.append(f"{key} {value}")

        # Counters
        for key, (value, help_text, labels) in sorted(self._counters.items()):
            base_name = key.split("{")[0]
            if base_name not in seen_help and help_text:
                lines.append(f"# HELP {base_name} {help_text}")
                lines.append(f"# TYPE {base_name} counter")
                seen_help.add(base_name)
            lines.append(f"{key} {value}")

        lines.append("")
        return "\n".join(lines)


class HPCOptMetricsExporter:
    """Exports HPC Workload Optimizer metrics for Prometheus/Grafana."""

    def __init__(self, feedback_store: Path | None = None):
        self.registry = MetricsRegistry()
        self.feedback_store = feedback_store
        self._start_time = time.time()

    def collect(self) -> str:
        """Collect all metrics and render as Prometheus text format."""
        self.registry = MetricsRegistry()  # Reset

        # System uptime
        self.registry.gauge(
            "hpcopt_uptime_seconds",
            time.time() - self._start_time,
            "Time since exporter started",
        )

        # Feedback loop metrics
        if self.feedback_store:
            self._collect_feedback_metrics()

        return self.registry.render()

    def _collect_feedback_metrics(self) -> None:
        """Collect prediction feedback metrics."""
        try:
            from hpcopt.integrations.feedback import FeedbackTracker

            assert self.feedback_store is not None  # narrowed by caller
            tracker = FeedbackTracker(store_path=self.feedback_store)
            report = tracker.generate_report()

            self.registry.gauge(
                "hpcopt_predictions_total",
                report.total_predictions,
                "Total prediction-vs-actual records",
            )
            self.registry.gauge(
                "hpcopt_interval_coverage",
                report.interval_coverage,
                "Fraction of actuals within predicted [p10, p90] interval",
            )
            self.registry.gauge(
                "hpcopt_prediction_mae_seconds",
                report.mean_absolute_error,
                "Mean absolute error of p50 predictions in seconds",
            )
            self.registry.gauge(
                "hpcopt_prediction_mape",
                report.mape,
                "Mean absolute percentage error of predictions",
            )
            self.registry.gauge(
                "hpcopt_prediction_p95_error_seconds",
                report.p95_error,
                "95th percentile prediction error in seconds",
            )
            self.registry.gauge(
                "hpcopt_drift_detected",
                1.0 if report.drift_detected else 0.0,
                "Whether model drift has been detected (1=yes, 0=no)",
            )
            self.registry.gauge(
                "hpcopt_overestimate_rate",
                report.overestimate_rate,
                "Fraction of predictions that overestimate runtime",
            )

        except Exception as exc:
            logger.warning("Failed to collect feedback metrics: %s", exc)

    def update_simulation_metrics(
        self,
        trace: str,
        policy: str,
        p95_bsld: float,
        utilization: float,
        elapsed_sec: float,
    ) -> None:
        """Update metrics from a simulation run."""
        labels = {"trace": trace, "policy": policy}
        self.registry.gauge(
            "hpcopt_simulation_p95_bsld",
            p95_bsld,
            "p95 Bounded Slowdown from simulation",
            labels,
        )
        self.registry.gauge(
            "hpcopt_simulation_utilization",
            utilization,
            "CPU utilization from simulation",
            labels,
        )
        self.registry.gauge(
            "hpcopt_simulation_elapsed_seconds",
            elapsed_sec,
            "Simulation wall clock time in seconds",
            labels,
        )


# ── FastAPI integration ─────────────────────────────────────────


def create_metrics_app(feedback_store: Path | None = None) -> Any:
    """Create a FastAPI sub-app that serves /metrics endpoint."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse
    except ImportError:
        logger.warning("FastAPI not available for metrics endpoint")
        return None

    app = FastAPI(title="HPC Workload Optimizer Metrics")
    exporter = HPCOptMetricsExporter(feedback_store=feedback_store)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return exporter.collect()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "exporter": "hpcopt_metrics"}

    return app
