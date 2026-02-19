# MVP Experiment Protocol

## 1. Purpose

This protocol defines a reproducible path from raw SWF traces to recommendation output with explicit fidelity and constraint gates.

## 2. Environment Requirements

Mandatory:

- Python 3.11+
- project installed in editable mode (`pip install -e .`)

Optional but recommended:

- Rust toolchain (`cargo`, `rustc`) for contract binaries and parser utility,
- WSL (Windows) plus Batsim installation for live Batsim execution path.

## 3. Installation

```bash
python -m pip install -e .[dev]
```

## 4. Reference Suite Lock

Before experiments, lock and verify reference hashes:

```bash
hpcopt data lock-reference-suite \
  --config configs/data/reference_suite.yaml \
  --raw-dir data/raw \
  --strict-missing
```

## 5. Ingestion Workflow

Example for one trace:

```bash
hpcopt ingest swf \
  --input data/raw/CTC-SP2-1996-3.1-cln.swf.gz \
  --out data/curated \
  --report-out outputs/reports
```

Repeat for each suite member.

## 6. Profiling Workflow

```bash
hpcopt profile trace \
  --dataset data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --out outputs/reports
```

Review heavy-tail, congestion, and user-skew profile sections before interpreting simulation outcomes.

## 7. Train Runtime Quantile Model

```bash
hpcopt train runtime \
  --dataset data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --out outputs/models \
  --model-id runtime_ctc_v1
```

## 8. Baseline Replay and Candidate Simulation

Baseline bundle:

```bash
hpcopt simulate replay-baselines \
  --trace data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --capacity-cpus 64 \
  --strict-invariants
```

ML candidate:

```bash
hpcopt simulate run \
  --trace data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --policy ML_BACKFILL_P50 \
  --capacity-cpus 64 \
  --runtime-guard-k 0.5 \
  --strict-uncertainty-mode \
  --strict-invariants
```

## 9. Fidelity Gate

Baseline fidelity:

```bash
hpcopt simulate fidelity-gate \
  --trace data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --capacity-cpus 64
```

Only proceed to accepted recommendation claims if fidelity status is `pass`.

## 10. Recommendation Generation

```bash
hpcopt recommend generate \
  --baseline-report <easy_baseline_sim_report.json> \
  --candidate-report <ml_candidate_sim_report.json> \
  --fidelity-report <fidelity_report.json> \
  --out outputs/reports
```

Interpret output:

- `accepted`: candidate passes guardrails and primary KPI criterion,
- `blocked`: guardrail or primary objective failure, with explicit reasons.

## 11. Export Bundle

```bash
hpcopt report export --run-id <run_id> --format both
```

Use this as the packaging unit for external review.

## 12. Optional Batsim Path

Generate config:

```bash
hpcopt simulate batsim-config \
  --trace data/curated/CTC-SP2-1996-3.1-cln.swf.parquet \
  --policy FIFO_STRICT \
  --run-id batsim_ctc
```

Dry run:

```bash
hpcopt simulate batsim-run \
  --config outputs/simulations/batsim_ctc_batsim_run_config.json \
  --dry-run
```

Live run (example on Windows host with WSL):

```bash
hpcopt simulate batsim-run \
  --config outputs/simulations/batsim_ctc_batsim_run_config.json \
  --use-wsl \
  --no-dry-run
```

If enabled and available, this command emits normalized simulation artifacts and candidate fidelity report.

## 13. Recommended Repeatability Controls

- keep `capacity_cpus` fixed when comparing policies,
- keep `runtime_guard_k` explicit in command logs,
- retain all manifests and reports,
- avoid changing fidelity thresholds during comparative runs,
- document any non-default config overrides.

