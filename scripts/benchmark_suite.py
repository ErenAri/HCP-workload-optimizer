"""Full benchmark suite: ingest all SWF traces and run Rust sim-runner.

Produces a comparative results table across all traces and policies.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
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
        "capacity_cpus": 512,
    },
    {
        "name": "HPC2N",
        "file": "HPC2N-2002-2.2-cln.swf.gz",
        "dataset_id": "hpc2n",
        "capacity_cpus": 240,
    },
    {
        "name": "SDSC-SP2",
        "file": "SDSC-SP2-1998-4.2-cln.swf.gz",
        "dataset_id": "sdsc_sp2",
        "capacity_cpus": 128,
    },
]


def banner(msg: str) -> None:
    print()
    print("=" * 70)
    print("  " + msg)
    print("=" * 70)


def main() -> None:
    import pandas as pd
    from hpcopt.ingest.swf import ingest_swf

    banner("HPC Workload Optimizer - Full Benchmark Suite")

    if not RUST_BIN.exists():
        print("ERROR: Rust binary not found at " + str(RUST_BIN))
        print("Build with: cd rust && cargo build --release")
        sys.exit(1)

    curated = OUT_DIR / "curated"
    reports = OUT_DIR / "reports"
    curated.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    results = []

    for trace in TRACES:
        name = trace["name"]
        ds_id = trace["dataset_id"]
        cap = trace["capacity_cpus"]
        swf_path = RAW_DIR / trace["file"]

        banner(name + " (" + str(cap) + " CPUs)")

        if not swf_path.exists():
            print("  SKIP: " + str(swf_path) + " not found")
            continue

        # Ingest SWF -> parquet
        pq = curated / (ds_id + ".parquet")
        if not pq.exists():
            print("  Ingesting " + trace["file"] + "...")
            result = ingest_swf(swf_path, curated, ds_id, reports)
            print("  Ingested: " + str(result.row_count) + " jobs")
        else:
            print("  Cached parquet: " + str(pq))

        # parquet -> JSON
        jf = curated / (ds_id + ".json")
        if not jf.exists():
            print("  Converting to JSON...")
            df = pd.read_parquet(pq)
            cols = ["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"]
            jobs = df[cols].to_dict("records")
            for j in jobs:
                for k in j:
                    j[k] = int(j[k])
            with open(jf, "w") as f:
                json.dump(jobs, f)
            print("  JSON: " + str(len(jobs)) + " jobs -> " + str(jf))
        else:
            with open(jf) as f:
                n = len(json.load(f))
            print("  Cached JSON: " + str(n) + " jobs")

        # Simulate each policy
        for pol in POLICIES:
            rpt = reports / (ds_id + "_" + pol.lower() + ".json")
            t0 = time.perf_counter()
            proc = subprocess.run(
                [str(RUST_BIN), "--input", str(jf), "--policy", pol, "--capacity-cpus", str(cap), "--output", str(rpt)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            dt = time.perf_counter() - t0

            if proc.returncode != 0:
                print("  " + pol + ": FAILED - " + proc.stderr[:200])
                continue

            with open(rpt) as f:
                m = json.load(f)["metrics"]

            bsld = m["p95_bsld"]
            util = m["utilization_mean"]
            mw = m["mean_wait_sec"]
            pw = m["p95_wait_sec"]
            viol = m["invariant_violations"]
            njobs = m["jobs_total"]

            print(
                "  "
                + pol
                + ": p95_bsld="
                + format(bsld, ".2f")
                + ", util="
                + format(util, ".1%")
                + ", mean_wait="
                + format(mw, ",.0f")
                + "s"
                + ", time="
                + format(dt, ".3f")
                + "s"
            )

            results.append(
                {
                    "trace": name,
                    "jobs": njobs,
                    "policy": pol,
                    "p95_bsld": bsld,
                    "utilization": util,
                    "mean_wait_sec": mw,
                    "p95_wait_sec": pw,
                    "elapsed_sec": dt,
                    "violations": viol,
                }
            )

    # ── Results table ───────────────────────────────────────────

    banner("BENCHMARK RESULTS")

    hdr = "{:<12} {:>8} {:<25} {:>10} {:>7} {:>10} {:>8}".format(
        "Trace", "Jobs", "Policy", "p95_BSLD", "Util", "MeanWait", "Time"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        print(
            "{:<12} {:>8,} {:<25} {:>10.2f} {:>6.1%} {:>10,.0f}s {:>7.3f}s".format(
                r["trace"],
                r["jobs"],
                r["policy"],
                r["p95_bsld"],
                r["utilization"],
                r["mean_wait_sec"],
                r["elapsed_sec"],
            )
        )

    # ── EASY vs FIFO lift ───────────────────────────────────────

    banner("EASY_BACKFILL vs FIFO - p95_BSLD Improvement")

    fifo = {r["trace"]: r for r in results if r["policy"] == "FIFO_STRICT"}
    easy = {r["trace"]: r for r in results if r["policy"] == "EASY_BACKFILL_BASELINE"}

    print("{:<12} {:>14} {:>14} {:>12}".format("Trace", "FIFO p95_BSLD", "EASY p95_BSLD", "Improvement"))
    print("-" * 55)

    for tname in fifo:
        if tname in easy:
            fb = fifo[tname]["p95_bsld"]
            eb = easy[tname]["p95_bsld"]
            imp = (1 - eb / fb) * 100 if fb > 0 else 0
            print("{:<12} {:>14.2f} {:>14.2f} {:>+11.1f}%".format(tname, fb, eb, imp))

    # ── Save summary ────────────────────────────────────────────

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rust_binary": str(RUST_BIN),
        "results": results,
    }
    summary_path = OUT_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to " + str(summary_path))
    print("Total traces: " + str(len(TRACES)) + ", policies: " + str(len(POLICIES)))
    print("Total simulation runs: " + str(len(results)))


if __name__ == "__main__":
    main()
