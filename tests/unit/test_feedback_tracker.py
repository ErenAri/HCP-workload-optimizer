"""Tests for the prediction feedback loop tracker."""
from __future__ import annotations

import json
from pathlib import Path

from hpcopt.integrations.feedback import FeedbackTracker, PredictionRecord


def test_feedback_tracker_record(tmp_path: Path) -> None:
    """Recording a prediction creates a valid PredictionRecord."""
    tracker = FeedbackTracker(store_path=tmp_path)
    rec = tracker.record(
        job_id=1,
        predicted={"p10": 50, "p50": 100, "p90": 200},
        actual_sec=120,
    )
    assert isinstance(rec, PredictionRecord)
    assert rec.job_id == 1
    assert rec.predicted_p10 == 50
    assert rec.predicted_p50 == 100
    assert rec.predicted_p90 == 200
    assert rec.actual_sec == 120
    assert rec.in_interval is True  # 120 in [50, 200]
    assert rec.error_sec == -20  # 100 - 120
    assert rec.abs_pct_error > 0


def test_feedback_tracker_out_of_interval(tmp_path: Path) -> None:
    """Actual outside [p10, p90] marks in_interval=False."""
    tracker = FeedbackTracker(store_path=tmp_path)
    rec = tracker.record(
        job_id=2,
        predicted={"p10": 50, "p50": 100, "p90": 200},
        actual_sec=300,
    )
    assert rec.in_interval is False


def test_feedback_tracker_report_empty(tmp_path: Path) -> None:
    """Report on empty tracker returns zero metrics."""
    tracker = FeedbackTracker(store_path=tmp_path)
    report = tracker.generate_report()
    assert report.total_predictions == 0
    assert report.drift_detected is False
    assert report.drift_severity == "none"


def test_feedback_tracker_report_with_data(tmp_path: Path) -> None:
    """Report computes coverage and error metrics correctly."""
    tracker = FeedbackTracker(store_path=tmp_path)
    for i in range(10):
        tracker.record(
            job_id=i,
            predicted={"p10": 50, "p50": 100, "p90": 200},
            actual_sec=100 + i * 5,
        )
    report = tracker.generate_report()
    assert report.total_predictions == 10
    assert report.interval_coverage == 1.0  # all within [50, 200]
    assert report.drift_detected is False
    assert report.mean_absolute_error > 0


def test_feedback_tracker_drift_detection(tmp_path: Path) -> None:
    """Coverage below threshold triggers drift detection."""
    tracker = FeedbackTracker(
        store_path=tmp_path,
        coverage_warning_threshold=0.70,
        coverage_critical_threshold=0.50,
    )
    # Generate predictions where most are outside interval
    for i in range(10):
        tracker.record(
            job_id=i,
            predicted={"p10": 50, "p50": 100, "p90": 120},
            actual_sec=500,  # way outside [50, 120]
        )
    report = tracker.generate_report()
    assert report.drift_detected is True
    assert report.drift_severity == "severe"


def test_feedback_tracker_persistence(tmp_path: Path) -> None:
    """Records persist to JSONL and reload on new instance."""
    tracker1 = FeedbackTracker(store_path=tmp_path)
    tracker1.record(job_id=1, predicted={"p10": 10, "p50": 50, "p90": 90}, actual_sec=60)
    tracker1.record(job_id=2, predicted={"p10": 20, "p50": 60, "p90": 100}, actual_sec=70)

    # Create new instance — should load records
    tracker2 = FeedbackTracker(store_path=tmp_path)
    assert len(tracker2._records) == 2
    report = tracker2.generate_report()
    assert report.total_predictions == 2


def test_feedback_tracker_to_dict(tmp_path: Path) -> None:
    """to_dict returns JSON-serializable summary."""
    tracker = FeedbackTracker(store_path=tmp_path)
    tracker.record(job_id=1, predicted={"p10": 10, "p50": 50, "p90": 90}, actual_sec=60)
    d = tracker.to_dict()
    assert isinstance(d, dict)
    assert "interval_coverage" in d
    assert "drift_detected" in d
    assert "recommendation" in d
    # Ensure it's JSON-serializable
    json.dumps(d)


def test_feedback_tracker_report_last_n(tmp_path: Path) -> None:
    """generate_report with last_n limits window."""
    tracker = FeedbackTracker(store_path=tmp_path)
    for i in range(20):
        tracker.record(job_id=i, predicted={"p10": 10, "p50": 50, "p90": 100}, actual_sec=50)
    report = tracker.generate_report(last_n=5)
    assert report.total_predictions == 5
