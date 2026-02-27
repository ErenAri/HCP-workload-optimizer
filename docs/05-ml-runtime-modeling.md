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
- backend-selectable quantile regressors for each alpha in `{0.10, 0.50, 0.90}`:
  - `GradientBoostingRegressor` (`sklearn` backend),
  - `LGBMRegressor` (`lightgbm` backend, when installed).

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
- `user_overrequest_mean_lookback`
- `user_runtime_median_lookback`
- `queue_congestion_at_submit_jobs`

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

Naive comparator block (for lift claims):

- global median runtime baseline,
- global mean runtime baseline,
- user-history median baseline (train-window only, fallback to global median),
- reported as `naive_baselines` and `p50_lift_vs_naive` in training metrics artifact.

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

Transform semantics:
- training uses `log1p(runtime_actual_sec)` and inference applies `expm1` inversion when metadata flag `log_transform=true` is present.

Model resolution order:

1. explicit path argument,
2. environment variable `HPCOPT_RUNTIME_MODEL_DIR`,
3. `outputs/models/runtime_latest.json`.

## 7. Integration into Policy Logic

In `ML_BACKFILL_P50` and `ML_BACKFILL_P10`, each job receives:

- `runtime_p50_sec`,
- `runtime_p90_sec`,
- `runtime_guard_sec`.

Guard formulas:

```text
ML_BACKFILL_P50: runtime_guard = p50 + runtime_guard_k * (p90 - p50)
ML_BACKFILL_P10: runtime_guard = p10 + runtime_guard_k * (p50 - p10)
```

Estimate semantics:
- `ML_BACKFILL_P50` uses `runtime_estimate_sec = p50`,
- `ML_BACKFILL_P10` uses `runtime_estimate_sec = p10` (more conservative backfill window sizing).

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

## 10. Hyperparameter Tuning

Implementation: `python/hpcopt/models/tuning.py`

Command:
```bash
hpcopt train tune --dataset <dataset.parquet> --quantile 0.5 --n-trials 20 --n-folds 3
```

Tuning approach:
- random search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `min_samples_leaf`,
- backend-aware parameter support (`num_leaves`, `colsample_bytree`, `min_child_samples` when `lightgbm` is selected),
- chronological cross-validation (no future leakage),
- scored by pinball loss for the target quantile,
- outputs best parameters and full trial history.

## 11. Resource-Fit Modeling

Implementation: `python/hpcopt/models/resource_fit.py`

A secondary model for resource allocation optimization:
- **Fragmentation risk classifier**: GradientBoostingClassifier predicting low/medium/high fragmentation risk based on CPU waste ratio.
- **Optimal node size regressor**: GradientBoostingRegressor predicting the best-fit node CPU count from a configurable set of node sizes.

Feature vector: `requested_cpus`, `requested_mem`, `runtime_requested_sec`, `queue_id`, `partition_id`, `user_id`, `submit_hour`, `submit_dow`.

## 12. Feature Importance Analysis

Implementation: `python/hpcopt/analysis/feature_importance.py`

Command:
```bash
hpcopt analysis feature-importance --model-dir <model_dir> --dataset <dataset.parquet>
```

Uses permutation importance to rank features per quantile. Reports importance scores with standard deviations.

## 13. Model Registry

Implementation: `python/hpcopt/models/registry.py`

Append-only JSONL-backed registry with thread-safe operations:
- `register`: create a new model entry (status: `registered`),
- `promote`: set a model as production (demotes any existing production model),
- `archive`: mark a model as archived (ineligible for promotion),
- `get_production`: resolve the current production model.

CLI commands: `hpcopt model list`, `hpcopt model promote`, `hpcopt model archive`.

## 14. Drift Detection

Implementation: `python/hpcopt/models/drift.py`

Command:
```bash
hpcopt model drift-check --eval-dataset <new_data.parquet>
```

Two drift indicators:
- **Feature PSI** (Population Stability Index): per-feature distributional shift detection. Bins derived from training distribution quantiles. Threshold: PSI > 0.20 indicates significant drift.
- **Metric degradation**: per-quantile pinball loss comparison against baseline training metrics. Threshold: 50% worse than baseline is flagged.

Configurable thresholds via `configs/models/drift_thresholds.yaml`.

## 15. Current Limitations

- no online learning (drift detection is batch-mode, not streaming),
- no automatic drift-triggered retraining pipeline,
- no memory prediction model in active use,
- no GPU efficiency predictor,
- no direct production scheduler actuation.

These are deliberate deferments to preserve systems-first rigor in the core pipeline.
