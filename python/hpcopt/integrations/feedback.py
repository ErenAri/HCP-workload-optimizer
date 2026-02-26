"""Prediction feedback loop — tracks prediction accuracy over time.

Compares runtime model predictions against actual job runtimes to
detect model drift, compute accuracy metrics, and trigger retraining
alerts.

Usage:
    from hpcopt.integrations.feedback import FeedbackTracker
    tracker = FeedbackTracker(store_path=Path("outputs/feedback"))
    tracker.record(job_id=123, predicted={"p10": 60, "p50": 120, "p90": 300}, actual_sec=145)
    report = tracker.generate_report()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction-vs-actual record."""
    job_id: int
    predicted_p10: float
    predicted_p50: float
    predicted_p90: float
    actual_sec: float
    timestamp: str
    in_interval: bool  # True if actual is within [p10, p90]
    error_sec: float   # p50 - actual
    abs_pct_error: float  # |p50 - actual| / max(actual, 1)


@dataclass
class FeedbackReport:
    """Summary of prediction accuracy over a time window."""
    total_predictions: int
    interval_coverage: float    # fraction within [p10, p90]
    mean_absolute_error: float  # seconds
    median_absolute_error: float
    mape: float                 # mean absolute percentage error
    p95_error: float            # 95th percentile error
    overestimate_rate: float    # fraction where p50 > actual
    underestimate_rate: float   # fraction where p50 < actual
    drift_detected: bool        # True if coverage drops below threshold
    drift_severity: str         # none, mild, severe
    recommendation: str         # human-readable action recommendation


class FeedbackTracker:
    """Track prediction accuracy and detect model drift."""

    def __init__(
        self,
        store_path: Path,
        coverage_warning_threshold: float = 0.70,
        coverage_critical_threshold: float = 0.50,
        max_records: int = 100_000,
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.coverage_warning = coverage_warning_threshold
        self.coverage_critical = coverage_critical_threshold
        self.max_records = max_records

        self.records_file = self.store_path / "predictions.jsonl"
        self._records: list[PredictionRecord] = []

        # Load existing records
        if self.records_file.exists():
            self._load_records()

    def _load_records(self) -> None:
        """Load records from JSONL file."""
        try:
            with open(self.records_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        self._records.append(PredictionRecord(**d))
            logger.info("Loaded %d feedback records", len(self._records))
        except Exception as exc:
            logger.warning("Failed to load feedback records: %s", exc)

    def _save_record(self, record: PredictionRecord) -> None:
        """Append record to JSONL file."""
        with open(self.records_file, "a") as f:
            f.write(json.dumps({
                "job_id": record.job_id,
                "predicted_p10": record.predicted_p10,
                "predicted_p50": record.predicted_p50,
                "predicted_p90": record.predicted_p90,
                "actual_sec": record.actual_sec,
                "timestamp": record.timestamp,
                "in_interval": record.in_interval,
                "error_sec": record.error_sec,
                "abs_pct_error": record.abs_pct_error,
            }) + "\n")

    def record(
        self,
        job_id: int,
        predicted: dict[str, float],
        actual_sec: float,
    ) -> PredictionRecord:
        """Record a prediction-vs-actual observation."""
        p10 = predicted.get("p10", 0)
        p50 = predicted.get("p50", 0)
        p90 = predicted.get("p90", 0)

        in_interval = p10 <= actual_sec <= p90
        error = p50 - actual_sec
        abs_pct = abs(error) / max(actual_sec, 1.0)

        rec = PredictionRecord(
            job_id=job_id,
            predicted_p10=p10,
            predicted_p50=p50,
            predicted_p90=p90,
            actual_sec=actual_sec,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            in_interval=in_interval,
            error_sec=error,
            abs_pct_error=abs_pct,
        )

        self._records.append(rec)
        self._save_record(rec)

        # Trim if too many records
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        return rec

    def generate_report(self, last_n: int | None = None) -> FeedbackReport:
        """Generate accuracy report over recent predictions."""
        records = self._records[-last_n:] if last_n else self._records

        if not records:
            return FeedbackReport(
                total_predictions=0,
                interval_coverage=0,
                mean_absolute_error=0,
                median_absolute_error=0,
                mape=0,
                p95_error=0,
                overestimate_rate=0,
                underestimate_rate=0,
                drift_detected=False,
                drift_severity="none",
                recommendation="No predictions recorded yet.",
            )

        n = len(records)
        coverage = sum(1 for r in records if r.in_interval) / n
        abs_errors = [abs(r.error_sec) for r in records]
        pct_errors = [r.abs_pct_error for r in records]

        mae = float(np.mean(abs_errors))
        median_ae = float(np.median(abs_errors))
        mape = float(np.mean(pct_errors))
        p95_err = float(np.percentile(abs_errors, 95))

        overest = sum(1 for r in records if r.error_sec > 0) / n
        underest = sum(1 for r in records if r.error_sec < 0) / n

        # Drift detection
        if coverage < self.coverage_critical:
            drift = True
            severity = "severe"
            rec = "RETRAIN IMMEDIATELY: Coverage below critical threshold ({:.0%}).".format(coverage)
        elif coverage < self.coverage_warning:
            drift = True
            severity = "mild"
            rec = "Consider retraining: Coverage dropped to {:.0%} (threshold: {:.0%}).".format(
                coverage, self.coverage_warning
            )
        else:
            drift = False
            severity = "none"
            rec = "Model performing within expected bounds (coverage: {:.0%}).".format(coverage)

        return FeedbackReport(
            total_predictions=n,
            interval_coverage=coverage,
            mean_absolute_error=mae,
            median_absolute_error=median_ae,
            mape=mape,
            p95_error=p95_err,
            overestimate_rate=overest,
            underestimate_rate=underest,
            drift_detected=drift,
            drift_severity=severity,
            recommendation=rec,
        )

    def to_dict(self) -> dict[str, Any]:
        """Export report as JSON-serializable dict."""
        report = self.generate_report()
        return {
            "total_predictions": report.total_predictions,
            "interval_coverage": round(report.interval_coverage, 4),
            "mean_absolute_error_sec": round(report.mean_absolute_error, 1),
            "median_absolute_error_sec": round(report.median_absolute_error, 1),
            "mape": round(report.mape, 4),
            "p95_error_sec": round(report.p95_error, 1),
            "overestimate_rate": round(report.overestimate_rate, 4),
            "underestimate_rate": round(report.underestimate_rate, 4),
            "drift_detected": report.drift_detected,
            "drift_severity": report.drift_severity,
            "recommendation": report.recommendation,
        }
