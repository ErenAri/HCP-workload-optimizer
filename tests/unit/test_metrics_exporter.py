"""Tests for the Prometheus metrics exporter."""
from __future__ import annotations

from pathlib import Path

from hpcopt.integrations.metrics_exporter import (
    HPCOptMetricsExporter,
    MetricsRegistry,
    create_metrics_app,
)


def test_metrics_registry_gauge() -> None:
    """Gauge values are stored and rendered."""
    reg = MetricsRegistry()
    reg.gauge("test_metric", 42.0, "A test metric")
    output = reg.render()
    assert "test_metric 42.0" in output
    assert "# HELP test_metric A test metric" in output
    assert "# TYPE test_metric gauge" in output


def test_metrics_registry_counter() -> None:
    """Counter values are stored and rendered."""
    reg = MetricsRegistry()
    reg.counter("request_count", 100.0, "Total requests")
    output = reg.render()
    assert "request_count 100.0" in output
    assert "# TYPE request_count counter" in output


def test_metrics_registry_labels() -> None:
    """Labels are rendered correctly."""
    reg = MetricsRegistry()
    reg.gauge("sim_bsld", 3.14, "BSLD", {"trace": "ctc", "policy": "easy"})
    output = reg.render()
    assert 'sim_bsld{policy="easy",trace="ctc"} 3.14' in output


def test_metrics_registry_histogram() -> None:
    """Histogram observations are recorded."""
    reg = MetricsRegistry()
    reg.observe_histogram("latency", 0.5, "Request latency")
    reg.observe_histogram("latency", 1.0)
    assert len(reg._histograms["latency"][0]) == 2


def test_exporter_collect(tmp_path: Path) -> None:
    """Exporter collects uptime and feedback metrics."""
    # Create some feedback data
    from hpcopt.integrations.feedback import FeedbackTracker

    tracker = FeedbackTracker(store_path=tmp_path)
    for i in range(5):
        tracker.record(
            job_id=i,
            predicted={"p10": 10, "p50": 50, "p90": 100},
            actual_sec=55,
        )

    exporter = HPCOptMetricsExporter(feedback_store=tmp_path)
    output = exporter.collect()

    assert "hpcopt_uptime_seconds" in output
    assert "hpcopt_predictions_total" in output
    assert "hpcopt_interval_coverage" in output
    assert "hpcopt_drift_detected" in output


def test_exporter_no_feedback_store() -> None:
    """Exporter works without feedback store."""
    exporter = HPCOptMetricsExporter(feedback_store=None)
    output = exporter.collect()
    assert "hpcopt_uptime_seconds" in output
    # Should not have feedback metrics
    assert "hpcopt_predictions_total" not in output


def test_exporter_simulation_metrics() -> None:
    """Simulation metrics are added correctly."""
    exporter = HPCOptMetricsExporter()
    exporter.update_simulation_metrics(
        trace="ctc_sp2",
        policy="EASY_BACKFILL",
        p95_bsld=3.94,
        utilization=0.555,
        elapsed_sec=0.035,
    )
    output = exporter.registry.render()
    assert "hpcopt_simulation_p95_bsld" in output
    assert "hpcopt_simulation_utilization" in output


def test_create_metrics_app(tmp_path: Path) -> None:
    """create_metrics_app returns a FastAPI app."""
    app = create_metrics_app(feedback_store=tmp_path)
    assert app is not None
