# ML Runtime Modeling and Uncertainty Handling

## 1. Role of ML in This System

ML is used to improve uncertainty estimates for scheduling decisions; it does not replace scheduler contracts.

Operational use:

- compute `p10`, `p50`, `p90` runtime predictions,
- derive uncertainty-aware backfill guard,
- quantify how often policy decisions are truly prediction-driven versus fallback-driven.

## 2. Training Pipeline

Implementation:
- `python/hpcopt/models/runtime_quantile.py`

Training command:

```bash
hpcopt train runtime --dataset <canonical_trace.parquet> --out outputs/models
```

Model family:
- `GradientBoostingRegressor` in quantile mode for each alpha in `{0.10, 0.50, 0.90}`.

## 3. Feature Contract

Feature vector:

- `requested_cpus`
- `runtime_requested_sec`
- `requested_mem`
- `queue_id`
- `partition_id`
- `user_id`
- `group_id`
- `submit_hour`
- `submit_dow`

Preparation rules:

- required fields are coerced to numeric types where needed,
- missing optional columns are created as null,
- all-null memory columns are converted to a stable numeric placeholder,
- chronological sort is applied before splitting.

## 4. Time-Split Discipline

The pipeline uses deterministic chronological splits:

- standard split target:
  - train: 70%
  - validation: 15%
  - test: 15%
- tiny datasets use protected fallback split sizes.

This avoids leakage from future job behavior into earlier predictions.

## 5. Reported Training Metrics

Per-quantile metrics:

- pinball loss,
- MAE.

Interval metric:

- empirical coverage of `[p10, p90]`.

Artifacts:

- `p10.joblib`, `p50.joblib`, `p90.joblib`,
- `metrics.json`,
- `metadata.json`,
- pointer file `outputs/models/runtime_latest.json`.

## 6. Inference Semantics

Predictor class:
- `RuntimeQuantilePredictor`

Monotonicity enforcement:
- predicted values are sorted to satisfy `p10 <= p50 <= p90` before use.

Model resolution order:

1. explicit path argument,
2. environment variable `HPCOPT_RUNTIME_MODEL_DIR`,
3. `outputs/models/runtime_latest.json`.

## 7. Integration into Policy Logic

In `ML_BACKFILL_P50`, each job receives:

- `runtime_p50_sec`,
- `runtime_p90_sec`,
- `runtime_guard_sec`.

Guard formula:

```text
runtime_guard = p50 + runtime_guard_k * (p90 - p50)
```

Strict uncertainty mode:
- backfill gate uses `p90` instead of `runtime_guard`.

## 8. Fallback Telemetry Contract

Every simulation report includes:

- `prediction_used_count` and rate,
- `requested_fallback_count` and rate,
- `actual_fallback_count` and rate,
- total scheduled jobs.

This telemetry is mandatory for honest attribution of policy improvement.

## 9. API Prediction Endpoints

The API exposes model usage via:

- `POST /v1/runtime/predict`

Behavior:

- uses trained model artifacts when available,
- falls back to deterministic heuristic when model artifacts are absent,
- returns guard value consistent with policy contract.

## 10. Current Limitations

- no online learning or drift adaptation,
- no memory prediction model in active use,
- no GPU efficiency predictor in MVP implementation,
- no direct production scheduler actuation.

These are deliberate deferments to preserve systems-first rigor in the core pipeline.

