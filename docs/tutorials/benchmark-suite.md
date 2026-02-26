# Benchmark Suite

Run reproducible comparisons across multiple HPC workload traces and scheduling policies.

## Running the Full Suite

```bash
python scripts/benchmark_suite.py
```

This will:

1. Download 3 PWA traces (CTC-SP2, HPC2N, SDSC-SP2)
2. Convert SWF to JSON format
3. Run FIFO + EASY_BACKFILL on each via Rust sim-runner
4. Generate a comparative results table

## Results

| Trace | Jobs | FIFO p95_BSLD | EASY p95_BSLD | Improvement |
|---|---|---|---|---|
| CTC-SP2 | 77,222 | 195.73 | **3.94** | **+98.0%** |
| HPC2N | 202,870 | 749.60 | **60.22** | **+92.0%** |
| SDSC-SP2 | 54,044 | 82,865 | **340.71** | **+99.6%** |

!!! note "Total simulation time"
    All 6 runs complete in under 1 second on the Rust engine.

## Adding Custom Traces

Place any SWF file in `data/raw/` and update the benchmark script:

```python
TRACES = {
    "my_cluster": {"swf": "data/raw/my_trace.swf", "cpus": 1024},
}
```

## Interactive Dashboard

The results are also available as an interactive HTML dashboard:

```bash
python -m http.server 9090 --directory docs
# Open http://localhost:9090/dashboard.html
```
