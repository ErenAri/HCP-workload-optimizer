# HPC Workload Optimizer

**Systems-first HPC scheduling research and engineering platform** — Python + Rust, focused on reproducible policy evaluation under uncertainty.

## Key Results

| Metric | Value |
|---|---|
| Simulation speedup (Rust) | **16,000–51,000×** vs Python |
| EASY_BACKFILL vs FIFO | **92–99.6%** p95 BSLD reduction |
| RL-optimized vs EASY_BACKFILL | **+25.5%** further improvement |
| Model training speedup (LightGBM) | **50×** vs sklearn |

## Quick Install

```bash
pip install hpc-workload-optimizer
# Optional: GPU-accelerated training
pip install hpc-workload-optimizer[lightgbm]
```

## 5-Minute Quickstart

```bash
# Ingest a standard workload format trace
hpcopt ingest swf --input data/raw/CTC-SP2-1996-3.1-cln.swf.gz \
    --dataset-id ctc_sp2 --out outputs/curated

# Train runtime prediction model
hpcopt train runtime --input outputs/curated/ctc_sp2.parquet \
    --model-id ctc_sp2_model --out outputs/models

# Run simulation (Python)
hpcopt simulate --input outputs/curated/ctc_sp2.parquet \
    --policy EASY_BACKFILL_BASELINE --capacity-cpus 512

# Run simulation (Rust — 16,000× faster)
cd rust && cargo build --release
./target/release/sim-runner --input trace.json \
    --policy EASY_BACKFILL_BASELINE --capacity-cpus 512
```

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ingest     │───▶│   Model      │───▶│  Simulate    │
│ SWF/Slurm/PBS│    │ GBR/LightGBM │    │ Rust/Python  │
└──────────────┘    └──────────────┘    └──────────────┘
                                              │
                    ┌──────────────┐    ┌──────┴───────┐
                    │  Recommend   │◀───│  Evaluate    │
                    │ Pareto/RL    │    │ Fidelity/BSLD│
                    └──────────────┘    └──────────────┘
```

## Documentation Sections

- **[Getting Started](tutorials/quickstart.md)** — Installation, first simulation
- **[Architecture](01-project-charter.md)** — Design documents and contracts
- **[Tutorials](tutorials/rust-sim-runner.md)** — Rust engine, LightGBM, RL search
- **[Operations](tutorials/deployment.md)** — Deployment, monitoring, production
- **[Benchmark Results](dashboard.md)** — Interactive results dashboard
