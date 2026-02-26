# Live Integration

Connect the optimizer to a live Slurm or PBS cluster for real-time prediction and feedback tracking.

## Slurm Integration

```python
from pathlib import Path
from hpcopt.integrations.slurm_connector import SlurmConnector

connector = SlurmConnector(
    model_dir=Path("outputs/models/latest"),
    lookback_hours=24,  # Fetch last 24h of jobs
)

result = connector.sync()
print(f"Jobs ingested: {result.jobs_ingested}")
print(f"Predictions made: {result.recommendations_generated}")
```

### CLI Usage

```bash
python -m hpcopt.integrations.slurm_connector \
    --model-dir outputs/models/latest \
    --lookback-hours 48 \
    --output sync_report.json
```

## PBS Pro Integration

```python
from hpcopt.integrations.pbs_connector import PBSConnector

connector = PBSConnector(
    model_dir=Path("outputs/models/latest"),
)
result = connector.sync()
```

## Prediction Feedback Loop

Track prediction accuracy and detect model drift:

```python
from hpcopt.integrations.feedback import FeedbackTracker

tracker = FeedbackTracker(store_path=Path("outputs/feedback"))

# Record each completed job
tracker.record(
    job_id=12345,
    predicted={"p10": 60, "p50": 120, "p90": 300},
    actual_sec=145,
)

# Check model health
report = tracker.generate_report()
print(f"Coverage: {report.interval_coverage:.0%}")
print(f"Drift: {report.drift_severity}")
print(f"Action: {report.recommendation}")
```

!!! warning "Drift Detection Thresholds"
    - **Warning** (mild): Coverage drops below **70%**
    - **Critical** (severe): Coverage drops below **50%**
    
    When drift is detected, the system recommends model retraining.

## Prometheus Metrics

Expose metrics for Grafana monitoring:

```python
from hpcopt.integrations.metrics_exporter import create_metrics_app

# Mount as FastAPI sub-app
metrics_app = create_metrics_app(
    feedback_store=Path("outputs/feedback")
)
app.mount("/metrics", metrics_app)
```

### Available Metrics

| Metric | Type | Description |
|---|---|---|
| `hpcopt_predictions_total` | gauge | Total predictions made |
| `hpcopt_interval_coverage` | gauge | Fraction within [p10, p90] |
| `hpcopt_prediction_mae_seconds` | gauge | Mean absolute error |
| `hpcopt_drift_detected` | gauge | 1 if drift detected |
| `hpcopt_simulation_p95_bsld` | gauge | Simulation BSLD |

### Grafana Dashboard

1. Add Prometheus datasource pointing to your `/metrics` endpoint
2. Import the dashboard from `docs/grafana/hpcopt_dashboard.json`
3. Set alerting rules on `hpcopt_drift_detected == 1`

## Automated Sync (Cron)

```bash
# crontab -e
*/15 * * * * python -m hpcopt.integrations.slurm_connector \
    --model-dir /opt/hpcopt/models/latest \
    --output /var/log/hpcopt/sync_$(date +\%Y\%m\%d_\%H\%M).json
```
