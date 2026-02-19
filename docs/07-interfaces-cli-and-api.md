# Interfaces: CLI and API

## 1. CLI Overview

Primary interface:
- `hpcopt` (Typer-based command surface in `python/hpcopt/cli/main.py`).

Top-level command groups:

- `ingest`
- `profile`
- `features`
- `train`
- `simulate`
- `stress`
- `recommend`
- `report`
- `serve`
- `data`

## 2. Implemented Commands

### Ingestion

```bash
hpcopt ingest swf --input <trace.swf|trace.swf.gz> --out data/curated
```

Outputs:

- canonical parquet,
- dataset metadata file,
- quality report,
- run manifest.

### Trace Profiling

```bash
hpcopt profile trace --dataset <dataset.parquet> --out outputs/reports
```

Outputs:

- trace profile JSON,
- profile manifest.

### Runtime Model Training

```bash
hpcopt train runtime --dataset <dataset.parquet> --out outputs/models
```

Outputs:

- quantile model artifacts (`p10/p50/p90`),
- metrics and metadata,
- training manifest.

### Simulation

```bash
hpcopt simulate run --trace <dataset.parquet> --policy FIFO_STRICT --capacity-cpus 64
hpcopt simulate run --trace <dataset.parquet> --policy EASY_BACKFILL_BASELINE --capacity-cpus 64
hpcopt simulate run --trace <dataset.parquet> --policy ML_BACKFILL_P50 --capacity-cpus 64
```

Key options:

- `--strict-invariants`
- `--runtime-model-dir`
- `--runtime-guard-k`
- `--strict-uncertainty-mode`

Outputs:

- jobs artifact parquet,
- queue artifact parquet,
- simulation report,
- invariant report,
- manifest.

### Baseline Replay Bundle

```bash
hpcopt simulate replay-baselines --trace <dataset.parquet> --capacity-cpus 64
```

Outputs:
- baseline replay report,
- per-policy artifacts,
- replay manifest.

### Fidelity Gate

```bash
hpcopt simulate fidelity-gate --trace <dataset.parquet> --capacity-cpus 64
```

Outputs:
- fidelity report,
- fidelity manifest.

### Batsim Path

```bash
hpcopt simulate batsim-config --trace <dataset.parquet> --policy FIFO_STRICT --run-id batsim_demo
hpcopt simulate batsim-run --config outputs/simulations/batsim_demo_batsim_run_config.json --dry-run
```

Optional live run:

```bash
hpcopt simulate batsim-run \
  --config outputs/simulations/batsim_demo_batsim_run_config.json \
  --use-wsl \
  --no-dry-run
```

Optional post-run behaviors:

- normalize Batsim output into standard simulation artifacts,
- emit candidate fidelity report (if source trace parquet is available).

### Recommendation

```bash
hpcopt recommend generate \
  --baseline-report <baseline_sim_report.json> \
  --candidate-report <candidate_sim_report.json> \
  --fidelity-report <optional_fidelity_report.json>
```

Outputs:

- recommendation report,
- recommendation manifest.

### Report Export

```bash
hpcopt report export --run-id <run_id> --format both
```

Outputs:

- run export JSON,
- run export markdown.

### Reference Suite Lock

```bash
hpcopt data lock-reference-suite --config configs/data/reference_suite.yaml --raw-dir data/raw
```

## 3. Scaffolded Commands (Not Fully Implemented)

- `hpcopt features build` (placeholder)
- `hpcopt stress run` (placeholder orchestration)

These commands are intentionally present as interface stubs pending deeper implementation.

## 4. API Overview

Service entrypoint:

```bash
hpcopt serve api --host 0.0.0.0 --port 8080
```

Implementation:
- `python/hpcopt/api/app.py`

## 5. API Endpoints

### Health

- `GET /health`

Response:
- service status and package version.

### Runtime Prediction

- `POST /v1/runtime/predict`

Behavior:

- uses trained quantile model when available,
- else deterministic heuristic fallback,
- always returns `runtime_p50_sec`, `runtime_p90_sec`, and `runtime_guard_sec`.

### Resource Fit Baseline

- `POST /v1/resource-fit/predict`

Behavior:

- deterministic capacity-fit baseline over provided candidate node CPU sizes,
- returns fragmentation risk category (`low`, `medium`, `high`).

## 6. API Documentation

When API is running:

- OpenAPI UI: `http://localhost:8080/docs`

## 7. Interface Stability Notes

- CLI commands and report schemas are treated as contract-bearing interfaces.
- Artifact keys used by evaluation/recommendation pipelines should be considered stable unless versioned migration is introduced.

