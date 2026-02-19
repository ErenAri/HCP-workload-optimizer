# HPC Workload Optimizer

Systems-first HPC scheduling research and engineering platform (Python + Rust) focused on reproducible policy evaluation under uncertainty.

## Abstract

HPC Workload Optimizer (HPCOpt) targets a persistent operations problem in shared compute clusters: queue delay and resource waste caused by static scheduling heuristics and uncertain job runtime requests.  
The project does not frame this as a standalone runtime prediction task. Instead, it builds a contract-driven decision and evaluation stack where:

- scheduler behavior is explicitly specified,
- replay is deterministic,
- invariants are executable,
- policy claims are gated by fidelity,
- recommendations are constrained by fairness and starvation bounds.

This design is meant to satisfy advanced systems engineering credibility requirements, not only model accuracy requirements.

## Primary Objective

Construct an advisory control layer that can demonstrate measurable scheduling improvements in simulation while preserving policy safety and reproducibility.

Operational target:

- improve primary queueing objective (`p95 BSLD`),
- maintain or improve utilization,
- avoid fairness/starvation regressions,
- produce auditable artifacts for every claim.

## Why This Project Is Different

Typical scheduling ML demos optimize a single predictive metric. HPCOpt enforces a stronger standard:

- policy-contract-first simulation,
- deterministic transition semantics,
- fidelity checks against observed traces,
- fallback transparency for uncertainty models,
- recommendation acceptance only under hard constraints.

## Implemented Capabilities (Current State)

- SWF ingestion and canonical parquet export with quality reporting.
- Reference-suite trace hash locking and enforcement.
- Trace profiling for heavy-tail, congestion, over-request, and user-skew analysis.
- Runtime quantile modeling (`p10/p50/p90`) with monotonic inference enforcement.
- Deterministic simulation core for:
  - `FIFO_STRICT`
  - `EASY_BACKFILL_BASELINE`
  - `ML_BACKFILL_P50`
- Invariant reporting with strict-fail mode.
- Baseline fidelity gate (aggregate + distribution + queue-correlation checks).
- Recommendation engine with:
  - primary KPI gating,
  - fairness/starvation constraints,
  - failure-mode and no-improvement narratives.
- Batsim integration path:
  - config generation,
  - run invocation (native/WSL),
  - normalization of Batsim outputs into standard simulation artifacts,
  - optional candidate fidelity report generation.
- API scaffold with runtime and resource-fit prediction endpoints.
- Artifact export and immutable run manifest generation.
- Rust utilities for parser stats and scheduler adapter contract parity.

## Deliberately Deferred

- direct production scheduler integration (Slurm/PBS/LSF),
- autonomous policy actuation,
- RL-based scheduler,
- complete feature-pipeline command implementation (`hpcopt features build` currently scaffolded),
- complete stress-run orchestration command (`hpcopt stress run` currently scaffolded).

## Architecture

```text
Raw SWF traces
  -> Canonical ingestion (parquet + quality report)
  -> Trace profiling
  -> Runtime quantile training
  -> Policy replay (native core and Batsim-normalized path)
  -> Fidelity + objective contract evaluation
  -> Recommendation generation + exportable artifacts
```

Language partition:

- Python: orchestration, simulation logic, ML, fidelity, recommendations, CLI/API.
- Rust: SWF parser utility, deterministic runner scaffolding, adapter contract parity binary.

## Repository Map

```text
python/hpcopt/
  cli/ api/ ingest/ profile/ models/ simulate/ recommend/ artifacts/
rust/
  swf-parser/ sim-runner/
configs/
  data/ simulation/
schemas/
  run_manifest + fidelity + invariant + adapter schemas
docs/
  formal technical documentation corpus
claudedocs/
  planning contracts and research appendix
```

## Installation

```bash
python -m pip install -e .[dev]
```

Optional (for Rust tools):

```bash
cargo --version
rustc --version
```

## Quickstart (Minimal End-to-End)

### 1) Ingest a trace

```bash
hpcopt ingest swf \
  --input data/raw/CTC-SP2-1996-3.1-cln.swf.gz \
  --dataset-id ctc_sp2_1996 \
  --out data/curated \
  --report-out outputs/reports
```

### 2) Build trace profile

```bash
hpcopt profile trace \
  --dataset data/curated/ctc_sp2_1996.parquet \
  --out outputs/reports
```

### 3) Train runtime quantile model

```bash
hpcopt train runtime \
  --dataset data/curated/ctc_sp2_1996.parquet \
  --out outputs/models \
  --model-id runtime_ctc_v1
```

### 4) Replay baselines

```bash
hpcopt simulate replay-baselines \
  --trace data/curated/ctc_sp2_1996.parquet \
  --capacity-cpus 64 \
  --strict-invariants
```

### 5) Run ML candidate policy

```bash
hpcopt simulate run \
  --trace data/curated/ctc_sp2_1996.parquet \
  --policy ML_BACKFILL_P50 \
  --capacity-cpus 64 \
  --runtime-guard-k 0.5 \
  --strict-uncertainty-mode \
  --strict-invariants
```

### 6) Execute fidelity gate

```bash
hpcopt simulate fidelity-gate \
  --trace data/curated/ctc_sp2_1996.parquet \
  --capacity-cpus 64
```

### 7) Generate recommendation

```bash
hpcopt recommend generate \
  --baseline-report <easy_baseline_sim_report.json> \
  --candidate-report <ml_candidate_sim_report.json> \
  --fidelity-report <fidelity_report.json> \
  --out outputs/reports
```

### 8) Export run bundle

```bash
hpcopt report export --run-id <run_id> --format both
```

## Batsim Workflow (Simulation Backend Path)

Generate Batsim run config:

```bash
hpcopt simulate batsim-config \
  --trace data/curated/ctc_sp2_1996.parquet \
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

When live run succeeds and normalization is enabled, the command emits:

- normalized jobs and queue parquet artifacts,
- simulation report in standard format,
- invariant report,
- optional candidate fidelity report.

## API

Start service:

```bash
hpcopt serve api --host 0.0.0.0 --port 8080
```

Available endpoints:

- `GET /health`
- `POST /v1/runtime/predict`
- `POST /v1/resource-fit/predict`

OpenAPI docs:
- `http://localhost:8080/docs`

Runtime prediction endpoint automatically uses trained model artifacts when available; otherwise it falls back to deterministic heuristic behavior.

## Reproducibility and Contracts

The project emits immutable manifests and schema-bound artifacts:

- `schemas/run_manifest.schema.json`
- `schemas/invariant_report.schema.json`
- `schemas/fidelity_report.schema.json`
- `schemas/adapter_snapshot.schema.json`
- `schemas/adapter_decision.schema.json`

Each run manifest records:

- command and timestamp,
- input/output hashes,
- package/tool versions,
- policy hash,
- config snapshots,
- environment fingerprint,
- seeds,
- manifest self-hash.

## Reference Suite Lock

Lock or refresh trace hashes:

```bash
hpcopt data lock-reference-suite \
  --config configs/data/reference_suite.yaml \
  --raw-dir data/raw
```

## Testing

```bash
pytest -q
```

Current baseline result in this workspace: `29 passed, 1 skipped`.

## Documentation

Primary docs:

- `docs/README.md`
- `docs/01-project-charter.md`
- `docs/02-system-architecture.md`
- `docs/03-data-model-and-ingestion.md`
- `docs/04-policy-and-simulation-contract.md`
- `docs/05-ml-runtime-modeling.md`
- `docs/06-fidelity-objective-and-recommendation.md`
- `docs/07-interfaces-cli-and-api.md`
- `docs/08-reproducibility-and-artifacts.md`
- `docs/09-experiment-protocol-mvp.md`
- `docs/10-roadmap-and-open-problems.md`

Design and contract history:

- `claudedocs/mvp_design_plan_python_rust_batsim.md`
- `claudedocs/policy_spec_baselines_mvp.md`
- `claudedocs/mvp_backlog_p0_p1_p2.md`
- `claudedocs/systems_first_research_appendix.md`

