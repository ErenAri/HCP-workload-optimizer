"""Full benchmark suite: ingest all SWF traces and run Rust sim-runner.

Produces a comparative results table across all traces and policies.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "benchmark"
RUST_BIN = PROJECT_ROOT / "rust" / "target" / "release" / "sim-runner.exe"
POLICIES = ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"]

TRACES = [
    {
        "name": "CTC-SP2",
        "file": "CTC-SP2-1996-3.1-cln.swf.gz",
        "dataset_id": "ctc_sp2",
        "capacity_cpus": 512,  # Cornell Theory Center SP2: 512 CPUs
    },
    {
        "name": "HPC2N",
        "file": "HPC2N-2002-2.2-cln.swf.gz",
        "dataset_id": "hpc2n",
        "capacity_cpus": 240,  # HPC2N cluster: 240 CPUs
    },
    {
        "name": "SDSC-SP2",
        "file": "SDSC-SP2-1998-4.2-cln.swf.gz",
        "dataset_id": "sdsc_sp2",
        "capacity_cpus": 128,  # SDSC SP2: 128 CPUs
    },
]


@dataclass
class BenchmarkResult:
    trace_name: str
    jobs_total: int
    policy: str
    mean_wait_sec: float
    p95_wait_sec: float
    p95_bsld: float
    utilization: float
    makespan_sec: int
    violations: int
    elapsed_sec: float


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}\n")


def ingest_trace(trace: dict) -> Path:
    """Ingest a SWF trace to parquet using hpcopt CLI."""
    swf_path = RAW_DIR / trace["file"]
    if not swf_path.exists():
        print(f"  SKIP: {swf_path} not found")
        return Path()

    curated_dir = OUT_DIR / "curated"
    report_dir = OUT_DIR / "reports"
    curated_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet = curated_dir / f"{trace['dataset_id']}.parquet"
    if parquet.exists():
        print(f"  Using cached: {parquet}")
        return parquet

    cmd = [
        sys.executable, "-m", "hpcopt.cli.main", "ingest", "swf",
        "--input", str(swf_path),
        "--dataset-id", trace["dataset_id"],
        "--out", str(curated_dir),
        "--report-out", str(report_dir),
    ]
    print(f"  Ingesting {swf_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return Path()

    print(f"  -> {parquet}")
    return parquet


def parquet_to_json(parquet_path: Path) -> Path:
    """Convert parquet trace to JSON for Rust sim-runner."""
    json_path = parquet_path.with_suffix(".json")
    if json_path.exists():
        print(f"  Using cached JSON: {json_path}")
        return json_path

    import pandas as pd

    print(f"  Converting {parquet_path.name} to JSON...")
    df = pd.read_parquet(parquet_path)
    required = ["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"]
    for col in required:
        if col not in df.columns:
            print(f"  ERROR: missing column {col}")
            return Path()

    jobs = df[required].to_dict("records")
    for j in jobs:
        j["job_id"] = int(j["job_id"])
        j["submit_ts"] = int(j["submit_ts"])
        j["runtime_actual_sec"] = int(j["runtime_actual_sec"])
        j["requested_cpus"] = int(j["requested_cpus"])

    with open(json_path, "w") as f:
        json.dump(jobs, f)

    print(f"  -> {json_path} ({len(jobs):,} jobs)")
    return json_path


def run_simulation(
    json_path: Path, policy: str, capacity_cpus: int, output_path: Path
) -> tuple[dict, float]:
    """Run Rust sim-runner and return (report_dict, elapsed_seconds)."""
    cmd = [
        str(RUST_BIN),
        "--input", str(json_path),
        "--policy", policy,
        "--capacity-cpus", str(capacity_cpus),
        "--output", str(output_path),
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return {}, elapsed

    with open(output_path) as f:
        report = json.load(f)

    return report, elapsed


def main() -> None:
    banner("HPC Workload Optimizer - Full Benchmark Suite")

    if not RUST_BIN.exists():
        print(f"ERROR: Rust binary not found at {RUST_BIN}")
        print("Build with: cd rust && cargo build --release")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkResult] = []

    for trace in TRACES:
        banner(f"Trace: {trace['name']} (capacity: {trace['capacity_cpus']} CPUs)")

        parquet = ingest_trace(trace)
        if not parquet.exists():
            continue

        json_path = parquet_to_json(parquet)
        if not json_path.exists():
            continue

        for policy in POLICIES:
            report_path = OUT_DIR / "reports" / f"{trace['dataset_id']}_{policy.lower()}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Simulating {policy}...", end=" ", flush=True)
            report, elapsed = run_simulation(json_path, policy, trace["capacity_cpus"], report_path)
            if not report:
                continue

            m = report["metrics"]
            print(f"done in {elapsed:.3f}s")

            results.append(BenchmarkResult(
                trace_name=trace["name"],
                jobs_total=m["jobs_total"],
                policy=policy,
                mean_wait_sec=m["mean_wait_sec"],
                p95_wait_sec=m["p95_wait_sec"],
                p95_bsld=m["p95_bsld"],
                utilization=m["utilization_mean"],
                makespan_sec=m["makespan_sec"],
                violations=m["invariant_violations"],
                elapsed_sec=elapsed,
            ))

    # ── Print results table ─────────────────────────────────────

    banner("BENCHMARK RESULTS")

    header = f"{'Trace':<12} {'Jobs':>8} {'Policy':<25} {'p95_BSLD':>10} {'Util':>7} {'MeanWait':>12} {'p95Wait':>12} {'Time':>8} {'Viol':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.trace_name:<12} {r.jobs_total:>8,} {r.policy:<25} "
            f"{r.p95_bsld:>10.2f} {r.utilization:>6.1%} "
            f"{r.mean_wait_sec:>12,.0f}s {r.p95_wait_sec:>11,.0f}s "
            f"{r.elapsed_sec:>7.3f}s {r.violations:>6}"
        )

    # ── EASY vs FIFO lift ───────────────────────────────────────

    print()
    banner("EASY_BACKFILL vs FIFO — p95_BSLD Improvement")

    fifo_map = {r.trace_name: r for r in results if r.policy == "FIFO_STRICT"}
    easy_map = {r.trace_name: r for r in results if r.policy == "EASY_BACKFILL_BASELINE"}

    print(f"{'Trace':<12} {'FIFO p95_BSLD':>14} {'EASY p95_BSLD':>14} {'Improvement':>12}")
    print("-" * 55)

    for name in fifo_map:
        if name in easy_map:
            fifo_bsld = fifo_map[name].p95_bsld
            easy_bsld = easy_map[name].p95_bsld
            if fifo_bsld > 0:
                improvement = (1 - easy_bsld / fifo_bsld) * 100
            else:
                improvement = 0
            print(f"{name:<12} {fifo_bsld:>14.2f} {easy_bsld:>14.2f} {improvement:>+11.1f}%")

    # ── Save summary ────────────────────────────────────────────

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rust_binary": str(RUST_BIN),
        "results": [
            {
                "trace": r.trace_name,
                "jobs": r.jobs_total,
                "policy": r.policy,
                "p95_bsld": r.p95_bsld,
                "utilization": r.utilization,
                "mean_wait_sec": r.mean_wait_sec,
                "p95_wait_sec": r.p95_wait_sec,
                "makespan_sec": r.makespan_sec,
                "violations": r.violations,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in results
        ],
    }
    summary_path = OUT_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {summary_path}")
    print(f"  Total traces: {len(TRACES)}, policies: {len(POLICIES)}")
    print(f"  Total simulation runs: {len(results)}")


if __name__ == "__main__":
    main()
