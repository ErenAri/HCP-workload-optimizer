# Rust Simulation Engine

The Rust sim-runner provides **16,000–51,000× speedup** over the Python simulator for large-scale policy evaluation.

## Building

```bash
cd rust/sim-runner
cargo build --release
```

## Usage

```bash
./target/release/sim-runner \
    --input trace.json \
    --policy EASY_BACKFILL_BASELINE \
    --capacity-cpus 512 \
    --output report.json
```

### Supported Policies

| Policy | Description |
|---|---|
| `FIFO_STRICT` | First-in-first-out, no backfill |
| `EASY_BACKFILL_BASELINE` | EASY backfill with shadow time reservation |

### Input Format

JSON array of job objects:

```json
[
  {
    "job_id": 1,
    "submit_ts": 1000000,
    "runtime_actual_sec": 3600,
    "requested_cpus": 4
  }
]
```

### Output Format

```json
{
  "policy_id": "EASY_BACKFILL_BASELINE",
  "metrics": {
    "total_jobs": 77222,
    "p95_bsld": 3.94,
    "utilization": 0.555,
    "mean_wait_sec": 1883,
    "p95_wait_sec": 13045
  }
}
```

## Python Bridge

```python
from hpcopt.simulate.rust_bridge import run_rust_simulation

result = run_rust_simulation(
    trace_path="trace.json",
    policy="EASY_BACKFILL_BASELINE",
    capacity_cpus=512,
)
print(result["metrics"]["p95_bsld"])
```

The bridge auto-discovers the Rust binary and falls back gracefully.

## Performance

| Trace | Jobs | Rust | Python | Speedup |
|---|---|---|---|---|
| CTC-SP2 | 77,222 | 0.035s | ~30 min | **51,000×** |
| HPC2N | 202,870 | 0.28s | ~60 min | **12,800×** |
| SDSC-SP2 | 54,044 | 0.06s | ~20 min | **20,000×** |
