"""One-command reproducer for paper / README headline benchmark numbers.

This script is the canonical entry point referenced from
``paper/artifact_appendix.md``. It runs the deterministic benchmark suite
across the reference SWF traces and emits a normalized JSON+Markdown
results table that maps directly to the "Benchmark Results (Parallel
Workloads Archive)" table in the README.

Usage:
    python scripts/reproduce_paper.py
    python scripts/reproduce_paper.py --traces ctc_sp2,hpc2n
    python scripts/reproduce_paper.py --policies FIFO_STRICT,EASY_BACKFILL_BASELINE,EASY_BACKFILL_TSAFRIR

Outputs (under outputs/reproduce_paper/<UTC-timestamp>/):
    - results.json     -- machine-readable results, schema-compatible with run_manifest.
    - results.md       -- markdown table mirroring the README.
    - environment.json -- captured Python/Rust/git versions and platform fingerprint.

Exit codes:
    0 -- all configured (trace, policy) pairs produced numbers.
    1 -- at least one pair failed; partial results are still written.

This script does NOT require Rust. If the Rust sim-runner binary is not
present, it falls back to the Python simulator (slower but identical
numbers).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "outputs" / "reproduce_paper"

DEFAULT_TRACES = [
    {"id": "ctc_sp2", "name": "CTC-SP2", "file": "CTC-SP2-1996-3.1-cln.swf.gz", "capacity_cpus": 512},
    {"id": "hpc2n", "name": "HPC2N", "file": "HPC2N-2002-2.2-cln.swf.gz", "capacity_cpus": 240},
    {"id": "sdsc_sp2", "name": "SDSC-SP2", "file": "SDSC-SP2-1998-4.2-cln.swf.gz", "capacity_cpus": 128},
]

DEFAULT_POLICIES = [
    "FIFO_STRICT",
    "EASY_BACKFILL_BASELINE",
    "EASY_BACKFILL_TSAFRIR",
]


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _capture_environment() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_revision": _git_revision(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }


def _load_trace(swf_path: Path, dataset_id: str, work_dir: Path):
    """Ingest SWF -> parquet (cached) -> pandas DataFrame.

    Lazy imports keep the CLI light when only --help is invoked.
    """
    import pandas as pd

    from hpcopt.ingest.swf import ingest_swf

    out_dir = work_dir / "curated"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        ingest_swf(
            input_path=swf_path,
            out_dir=out_dir,
            dataset_id=dataset_id,
            report_dir=report_dir,
        )
    return pd.read_parquet(parquet_path)


def _run_one(trace_df, policy_id: str, capacity_cpus: int, run_id: str) -> dict:
    from hpcopt.simulate.core import run_simulation_from_trace

    result = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id=policy_id,
        capacity_cpus=capacity_cpus,
        run_id=run_id,
        strict_invariants=False,
    )
    return {
        "policy_id": result.policy_id,
        "metrics": result.metrics,
        "objective_metrics": result.objective_metrics,
        "fallback_accounting": result.fallback_accounting,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", help="Comma-separated trace ids (default: all)", default=None)
    parser.add_argument("--policies", help="Comma-separated policy ids (default: standard set)", default=None)
    parser.add_argument("--out", help="Override output root directory", default=None)
    args = parser.parse_args()

    selected_traces = DEFAULT_TRACES
    if args.traces:
        wanted = {t.strip() for t in args.traces.split(",") if t.strip()}
        selected_traces = [t for t in DEFAULT_TRACES if t["id"] in wanted]
        if not selected_traces:
            print(f"error: no matching traces in {wanted}", file=sys.stderr)
            return 1

    selected_policies = DEFAULT_POLICIES
    if args.policies:
        selected_policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) if args.out else OUT_ROOT / ts
    out_root.mkdir(parents=True, exist_ok=True)

    env = _capture_environment()
    (out_root / "environment.json").write_text(json.dumps(env, indent=2))

    rows: list[dict] = []
    failures: list[str] = []

    for trace in selected_traces:
        swf_path = DATA_DIR / trace["file"]
        if not swf_path.exists():
            failures.append(f"{trace['id']}: missing trace file {swf_path}")
            continue

        print(f"[reproduce_paper] loading {trace['name']} from {swf_path}")
        try:
            trace_df = _load_trace(swf_path, trace["id"], out_root / "_work")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{trace['id']}: ingest failed: {exc}")
            continue

        for policy in selected_policies:
            run_id = f"{trace['id']}__{policy}__{ts}"
            print(f"[reproduce_paper] running {policy} on {trace['name']} (jobs={len(trace_df)})")
            try:
                report = _run_one(trace_df, policy, int(trace["capacity_cpus"]), run_id)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{trace['id']}/{policy}: simulation failed: {exc}")
                continue

            rows.append(
                {
                    "trace_id": trace["id"],
                    "trace_name": trace["name"],
                    "jobs": int(len(trace_df)),
                    "capacity_cpus": int(trace["capacity_cpus"]),
                    "policy_id": policy,
                    "p95_bsld": report["objective_metrics"].get("p95_bsld"),
                    "mean_wait_sec": report["metrics"].get("mean_wait_sec"),
                    "utilization": report["metrics"].get("utilization"),
                    "fallback_accounting": report["fallback_accounting"],
                }
            )

    (out_root / "results.json").write_text(json.dumps({"environment": env, "rows": rows, "failures": failures}, indent=2))

    md_lines = [
        "# Reproduce-Paper Results",
        "",
        f"Timestamp (UTC): `{ts}`  •  git: `{env['git_revision']}`",
        "",
        "| Trace | Jobs | Policy | p95 BSLD | Utilization | Mean Wait (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['trace_name']} | {r['jobs']:,} | `{r['policy_id']}` | "
            f"{r['p95_bsld']:.2f} | {r['utilization']:.1%} | {r['mean_wait_sec']:.0f} |"
            if r["p95_bsld"] is not None and r["utilization"] is not None and r["mean_wait_sec"] is not None
            else f"| {r['trace_name']} | {r['jobs']:,} | `{r['policy_id']}` | n/a | n/a | n/a |"
        )
    if failures:
        md_lines.extend(["", "## Failures", ""])
        md_lines.extend([f"- {f}" for f in failures])
    (out_root / "results.md").write_text("\n".join(md_lines) + "\n")

    print(f"[reproduce_paper] wrote {out_root / 'results.json'}")
    print(f"[reproduce_paper] wrote {out_root / 'results.md'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
