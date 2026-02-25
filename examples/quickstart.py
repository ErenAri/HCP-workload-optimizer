#!/usr/bin/env python3
"""HPCOpt Quickstart — end-to-end demo in ~2 minutes.

Demonstrates the full pipeline:
  1. Ingest a real SWF workload trace
  2. Profile the trace
  3. Build time-safe features
  4. Train runtime quantile models
  5. Simulate policies (FIFO, EASY_BACKFILL, ML_BACKFILL)
  6. Run fidelity gate
  7. Generate recommendation

Prerequisites:
  pip install -e ".[dev]"
  # Ensure data/raw/CTC-SP2-1996-3.1-cln.swf.gz exists

Usage:
  python examples/quickstart.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────

RAW_TRACE = Path("data/raw/CTC-SP2-1996-3.1-cln.swf.gz")
OUT = Path("outputs/quickstart")
CURATED_DIR = OUT / "curated"
REPORTS_DIR = OUT / "reports"
MODELS_DIR = OUT / "models"
SIM_DIR = OUT / "simulations"
REC_DIR = OUT / "recommendations"
DATASET_ID = "ctc_sp2_demo"
CAPACITY_CPUS = 64
SEED = 42


def step(num: int, title: str) -> None:
    """Print a step header."""
    print(f"\n{'='*60}")
    print(f"  Step {num}: {title}")
    print(f"{'='*60}")


def main() -> int:
    if not RAW_TRACE.exists():
        print(f"ERROR: Raw trace not found at {RAW_TRACE}")
        print("Download from: https://www.cs.huji.ac.il/labs/parallel/workload/")
        return 1

    t0 = time.perf_counter()

    # ── Step 1: Ingest ──────────────────────────────────────
    step(1, "Ingest SWF trace")
    from hpcopt.ingest.swf import ingest_swf
    from hpcopt.utils.io import ensure_dir

    for d in [CURATED_DIR, REPORTS_DIR, MODELS_DIR, SIM_DIR, REC_DIR]:
        ensure_dir(d)

    ingest_result = ingest_swf(
        input_path=RAW_TRACE, out_dir=CURATED_DIR,
        dataset_id=DATASET_ID, report_dir=REPORTS_DIR,
    )
    dataset = ingest_result.dataset_path
    print(f"  Ingested {ingest_result.row_count:,} jobs -> {dataset}")

    # ── Step 2: Profile ─────────────────────────────────────
    step(2, "Build trace profile")
    from hpcopt.profile.trace_profile import build_trace_profile
    profile = build_trace_profile(
        dataset_path=dataset, report_dir=REPORTS_DIR, dataset_id=DATASET_ID,
    )
    print(f"  Profile: {profile.row_count:,} rows")
    print(f"  Report: {profile.profile_path}")

    # ── Step 3: Features ────────────────────────────────────
    step(3, "Build time-safe feature dataset")
    from hpcopt.features.pipeline import build_feature_dataset
    features = build_feature_dataset(
        dataset_path=dataset, out_dir=CURATED_DIR,
        report_dir=REPORTS_DIR, dataset_id=DATASET_ID,
        n_folds=3, train_fraction=0.7, val_fraction=0.15,
    )
    print(f"  Feature dataset: {features.feature_dataset_path}")

    # ── Step 4: Train ───────────────────────────────────────
    step(4, "Train runtime quantile models (p10/p50/p90)")
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models
    train_result = train_runtime_quantile_models(
        dataset_path=dataset, out_dir=MODELS_DIR,
        model_id=f"runtime_{DATASET_ID}", seed=SEED,
    )
    with open(train_result.metrics_path) as f:
        metrics = json.load(f)
    print(f"  Model: {train_result.model_dir}")
    print(f"  p50 MAE:  {metrics['quantiles']['p50']['mae']:,.0f} sec")
    print(f"  p50 lift vs global mean: {metrics['p50_lift_vs_naive']['global_mean']['mae_improvement']:,.0f} sec")
    print(f"  Interval coverage: {metrics['interval_coverage_p10_p90']:.1%}")

    # ── Step 5: Simulate ────────────────────────────────────
    step(5, "Replay scheduling policies")
    import pandas as pd
    from hpcopt.simulate.core import run_simulation_from_trace

    trace_df = pd.read_parquet(dataset)

    sim_reports: dict[str, Path] = {}
    for policy in ["FIFO_STRICT", "EASY_BACKFILL_BASELINE"]:
        result = run_simulation_from_trace(
            trace_df=trace_df, policy_id=policy,
            capacity_cpus=CAPACITY_CPUS, run_id=f"qs_{policy.lower()}",
            strict_invariants=True,
        )
        report_path = SIM_DIR / f"{policy.lower()}_report.json"
        report = {
            "policy_id": policy, "run_id": f"qs_{policy.lower()}",
            "metrics": result.metrics, "objective_contract": result.objective_metrics,
            "invariant_report": result.invariant_report,
            "fallback_accounting": result.fallback_accounting,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        sim_reports[policy] = report_path

        bsld = result.metrics.get("p95_bsld", "N/A")
        util = result.metrics.get("utilization_mean", "N/A")
        print(f"  {policy}: p95_BSLD={bsld}, util={util}")

    # ML candidate
    from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor
    predictor = RuntimeQuantilePredictor(train_result.model_dir)
    ml_result = run_simulation_from_trace(
        trace_df=trace_df, policy_id="ML_BACKFILL_P50",
        capacity_cpus=CAPACITY_CPUS, run_id="qs_ml_backfill",
        strict_invariants=True, runtime_predictor=predictor,
        runtime_guard_k=0.5,
    )
    ml_report_path = SIM_DIR / "ml_backfill_p50_report.json"
    ml_report = {
        "policy_id": "ML_BACKFILL_P50", "run_id": "qs_ml_backfill",
        "metrics": ml_result.metrics, "objective_contract": ml_result.objective_metrics,
        "invariant_report": ml_result.invariant_report,
        "fallback_accounting": ml_result.fallback_accounting,
    }
    with open(ml_report_path, "w") as f:
        json.dump(ml_report, f, indent=2, default=str)
    sim_reports["ML_BACKFILL_P50"] = ml_report_path

    bsld = ml_result.metrics.get("p95_bsld", "N/A")
    util = ml_result.metrics.get("utilization_mean", "N/A")
    fb = ml_result.fallback_accounting.get("fallback_ratio", "N/A") if ml_result.fallback_accounting else "N/A"
    print(f"  ML_BACKFILL_P50: p95_BSLD={bsld}, util={util}, fallback={fb}")

    # ── Step 6: Fidelity gate ────────────────────────────────
    step(6, "Run fidelity gate")
    from hpcopt.simulate.fidelity import run_baseline_fidelity_gate
    fidelity = run_baseline_fidelity_gate(
        trace_df=trace_df, capacity_cpus=CAPACITY_CPUS,
    )
    fidelity_path = SIM_DIR / "fidelity_report.json"
    with open(fidelity_path, "w") as f:
        json.dump(fidelity.report, f, indent=2, default=str)
    print(f"  Fidelity: {fidelity.status}")

    # ── Step 7: Recommendation ──────────────────────────────
    step(7, "Generate recommendation")
    from hpcopt.recommend.engine import generate_recommendation_report
    rec_path = REC_DIR / "recommendation_report.json"
    rec = generate_recommendation_report(
        baseline_report_path=sim_reports["EASY_BACKFILL_BASELINE"],
        candidate_report_paths=[ml_report_path],
        out_path=rec_path,
        fidelity_report_path=fidelity_path,
    )
    status = rec.payload.get("status", "unknown")
    print(f"  Decision: {status}")
    if rec.payload.get("selected_recommendation"):
        best = rec.payload["selected_recommendation"]
        print(f"  Winner: {best.get('policy_id')} (score={best['score']['score']:.3f})")
    elif rec.payload.get("no_improvement_narrative"):
        narr = rec.payload["no_improvement_narrative"]
        print(f"  Explanation: {narr.get('summary', 'N/A')}")
        print(f"  Workload regime: {narr.get('workload_regime', 'N/A')}")

    # ── Summary ─────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — full pipeline completed in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Outputs in: {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
