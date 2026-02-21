"""Credibility protocol orchestrator.

Chains the manual pipeline steps from the experiment protocol into a single
automated run: ingest -> profile -> features -> train -> replay-baselines ->
simulate ML -> fidelity-gate -> recommend -> export.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.data.reference_suite import (
    assert_reference_by_filename_and_hash,
    assert_reference_trace_hash_match,
    load_reference_suite,
)
from hpcopt.features.pipeline import build_feature_dataset
from hpcopt.ingest.swf import ingest_swf
from hpcopt.models.runtime_quantile import (
    RuntimeQuantilePredictor,
    resolve_runtime_model_dir,
    train_runtime_quantile_models,
)
from hpcopt.profile.trace_profile import build_trace_profile
from hpcopt.recommend.engine import generate_recommendation_report
from hpcopt.simulate.core import run_simulation_from_trace
from hpcopt.simulate.fidelity import run_baseline_fidelity_gate
from hpcopt.simulate.objective import evaluate_constraint_contract
from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass
class TraceRunResult:
    """Result of a single trace credibility run."""

    trace_id: str
    status: str  # "pass" | "fail" | "error"
    dataset_path: Path | None = None
    profile_path: Path | None = None
    feature_dataset_path: Path | None = None
    model_dir: Path | None = None
    baseline_report_path: Path | None = None
    candidate_report_path: Path | None = None
    fidelity_report_path: Path | None = None
    recommendation_report_path: Path | None = None
    fidelity_status: str | None = None
    recommendation_status: str | None = None
    error_message: str | None = None
    artifact_paths: list[Path] = field(default_factory=list)


@dataclass
class CredibilityRunResult:
    """Aggregate result of the full credibility protocol."""

    status: str  # "pass" | "partial" | "fail"
    per_trace: list[TraceRunResult] = field(default_factory=list)
    output_dir: Path | None = None
    timestamp_utc: str = ""


def run_credibility_protocol(
    trace_path: Path,
    trace_id: str,
    capacity_cpus: int,
    runtime_guard_k: float = 0.5,
    seed: int = 42,
    output_dir: Path = Path("outputs/credibility"),
    reference_suite_config: Path = Path("configs/data/reference_suite.yaml"),
    fidelity_config: Path = Path("configs/simulation/fidelity_gate.yaml"),
    strict_invariants: bool = True,
    strict_uncertainty_mode: bool = False,
) -> TraceRunResult:
    """Execute the full credibility protocol for a single trace."""
    result = TraceRunResult(trace_id=trace_id, status="error")
    run_ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    trace_out = output_dir / trace_id
    curated_dir = trace_out / "curated"
    reports_dir = trace_out / "reports"
    models_dir = trace_out / "models"
    sims_dir = trace_out / "simulations"

    for d in [curated_dir, reports_dir, models_dir, sims_dir]:
        ensure_dir(d)

    try:
        # Step 1: Ingest
        logger.info("[%s] Step 1: Ingesting trace %s", trace_id, trace_path)
        ingest_result = ingest_swf(
            input_path=trace_path,
            out_dir=curated_dir,
            dataset_id=trace_id,
            report_dir=reports_dir,
        )
        result.dataset_path = ingest_result.dataset_path
        result.artifact_paths.append(ingest_result.dataset_path)

        # Reference suite hash check (non-fatal)
        try:
            if reference_suite_config.exists():
                ref_match = assert_reference_trace_hash_match(
                    trace_path=trace_path,
                    config_path=reference_suite_config,
                )
                if ref_match is not None:
                    metadata = json.loads(
                        ingest_result.dataset_metadata_path.read_text(encoding="utf-8")
                    )
                    metadata["reference_suite"] = ref_match
                    write_json(ingest_result.dataset_metadata_path, metadata)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("[%s] Reference suite check: %s", trace_id, exc)

        # Step 2: Profile
        logger.info("[%s] Step 2: Building trace profile", trace_id)
        profile_result = build_trace_profile(
            dataset_path=ingest_result.dataset_path,
            report_dir=reports_dir,
            dataset_id=trace_id,
        )
        result.profile_path = profile_result.profile_path
        result.artifact_paths.append(profile_result.profile_path)

        # Step 3: Features
        logger.info("[%s] Step 3: Building feature dataset", trace_id)
        feature_result = build_feature_dataset(
            dataset_path=ingest_result.dataset_path,
            out_dir=curated_dir,
            report_dir=reports_dir,
            dataset_id=trace_id,
        )
        result.feature_dataset_path = feature_result.feature_dataset_path
        result.artifact_paths.append(feature_result.feature_dataset_path)

        # Step 4: Train
        logger.info("[%s] Step 4: Training runtime quantile models", trace_id)
        model_id = f"runtime_{trace_id}_{run_ts}"
        train_result = train_runtime_quantile_models(
            dataset_path=ingest_result.dataset_path,
            out_dir=models_dir,
            model_id=model_id,
            seed=seed,
        )
        result.model_dir = train_result.model_dir
        result.artifact_paths.append(train_result.metrics_path)

        # Step 5: Replay baselines
        logger.info("[%s] Step 5: Replaying baselines", trace_id)
        trace_df = pd.read_parquet(ingest_result.dataset_path)
        run_id = f"cred_{trace_id}_{run_ts}"

        baseline_policies = ("FIFO_STRICT", "EASY_BACKFILL_BASELINE")
        baseline_reports: dict[str, Path] = {}
        for policy in baseline_policies:
            sim = run_simulation_from_trace(
                trace_df=trace_df,
                policy_id=policy,
                capacity_cpus=capacity_cpus,
                run_id=f"{run_id}_{policy.lower()}",
                strict_invariants=strict_invariants,
            )
            jobs_path = sims_dir / f"{run_id}_{policy.lower()}_jobs.parquet"
            sim_report_path = reports_dir / f"{run_id}_{policy.lower()}_sim_report.json"
            sim.jobs_df.to_parquet(jobs_path, index=False)
            write_json(
                sim_report_path,
                {
                    "run_id": run_id,
                    "policy_id": policy,
                    "status": "ok",
                    "metrics": sim.metrics,
                    "objective_metrics": sim.objective_metrics,
                    "fallback_accounting": sim.fallback_accounting,
                },
            )
            baseline_reports[policy] = sim_report_path
            result.artifact_paths.append(sim_report_path)

        result.baseline_report_path = baseline_reports.get("EASY_BACKFILL_BASELINE")

        # Step 6: Simulate ML candidate
        logger.info("[%s] Step 6: Simulating ML candidate", trace_id)
        predictor = RuntimeQuantilePredictor(train_result.model_dir)
        ml_sim = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id="ML_BACKFILL_P50",
            capacity_cpus=capacity_cpus,
            run_id=f"{run_id}_ml_backfill_p50",
            strict_invariants=strict_invariants,
            runtime_predictor=predictor,
            runtime_guard_k=runtime_guard_k,
            strict_uncertainty_mode=strict_uncertainty_mode,
        )
        ml_jobs_path = sims_dir / f"{run_id}_ml_backfill_p50_jobs.parquet"
        ml_report_path = reports_dir / f"{run_id}_ml_backfill_p50_sim_report.json"
        ml_sim.jobs_df.to_parquet(ml_jobs_path, index=False)
        write_json(
            ml_report_path,
            {
                "run_id": run_id,
                "policy_id": "ML_BACKFILL_P50",
                "status": "ok",
                "metrics": ml_sim.metrics,
                "objective_metrics": ml_sim.objective_metrics,
                "fallback_accounting": ml_sim.fallback_accounting,
                "model_dir": str(train_result.model_dir),
            },
        )
        result.candidate_report_path = ml_report_path
        result.artifact_paths.append(ml_report_path)

        # Step 7: Fidelity gate
        logger.info("[%s] Step 7: Running fidelity gate", trace_id)
        fidelity_report_path = reports_dir / f"{run_id}_fidelity_report.json"
        fidelity_config_resolved = (
            fidelity_config if fidelity_config.exists() else None
        )
        fidelity_result = run_baseline_fidelity_gate(
            trace_df=trace_df,
            capacity_cpus=capacity_cpus,
            out_path=fidelity_report_path,
            run_id=run_id,
            config_path=fidelity_config_resolved,
            strict_invariants=strict_invariants,
        )
        result.fidelity_report_path = fidelity_result.report_path
        result.fidelity_status = fidelity_result.status
        result.artifact_paths.append(fidelity_result.report_path)

        # Step 8: Recommend
        logger.info("[%s] Step 8: Generating recommendation", trace_id)
        recommendation_path = reports_dir / f"{run_id}_recommendation_report.json"
        easy_baseline = baseline_reports.get("EASY_BACKFILL_BASELINE")
        rec_result = generate_recommendation_report(
            baseline_report_path=easy_baseline,
            candidate_report_paths=[ml_report_path],
            out_path=recommendation_path,
            fidelity_report_path=fidelity_result.report_path,
        )
        result.recommendation_report_path = rec_result.report_path
        result.recommendation_status = rec_result.payload.get("status", "unknown")
        result.artifact_paths.append(rec_result.report_path)

        # Step 9: Build manifest
        logger.info("[%s] Step 9: Writing credibility manifest", trace_id)
        manifest = build_manifest(
            command="hpcopt credibility run",
            inputs=[trace_path],
            outputs=result.artifact_paths,
            params={
                "trace_id": trace_id,
                "capacity_cpus": capacity_cpus,
                "runtime_guard_k": runtime_guard_k,
                "seed": seed,
                "strict_invariants": strict_invariants,
            },
            config_paths=[
                p
                for p in [reference_suite_config, fidelity_config]
                if p.exists()
            ],
            seeds=[seed],
        )
        manifest_path = reports_dir / f"{run_id}_credibility_manifest.json"
        write_manifest(manifest_path, manifest)
        result.artifact_paths.append(manifest_path)

        # Determine overall status
        result.status = "pass" if result.fidelity_status == "pass" else "fail"
        logger.info(
            "[%s] Credibility protocol complete: status=%s, fidelity=%s, recommendation=%s",
            trace_id,
            result.status,
            result.fidelity_status,
            result.recommendation_status,
        )

    except Exception as exc:
        result.status = "error"
        result.error_message = str(exc)
        logger.error("[%s] Credibility protocol failed: %s", trace_id, exc)

    return result


def run_suite_credibility(
    reference_suite_config: Path = Path("configs/data/reference_suite.yaml"),
    sweep_config_path: Path | None = None,
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("outputs/credibility"),
    fidelity_config: Path = Path("configs/simulation/fidelity_gate.yaml"),
    strict_invariants: bool = True,
) -> CredibilityRunResult:
    """Iterate over all reference-suite members and run the credibility protocol."""
    suite = load_reference_suite(reference_suite_config)
    ensure_dir(output_dir)

    # Load sweep config for per-trace overrides
    sweep: dict[str, Any] = {}
    if sweep_config_path is not None and sweep_config_path.exists():
        sweep = yaml.safe_load(sweep_config_path.read_text(encoding="utf-8")) or {}

    defaults = sweep.get("defaults", {})
    default_capacity = int(defaults.get("capacity_cpus", 64))
    default_guard_k = float(defaults.get("runtime_guard_k", 0.5))
    default_seed = int(defaults.get("seed", 42))
    trace_overrides: dict[str, dict[str, Any]] = sweep.get("traces", {})

    timestamp = dt.datetime.now(tz=dt.UTC).isoformat()
    aggregate = CredibilityRunResult(
        status="fail",
        output_dir=output_dir,
        timestamp_utc=timestamp,
    )

    for trace in suite.traces:
        trace_file = raw_dir / trace.filename
        if not trace_file.exists():
            logger.warning(
                "Skipping trace %s: file not found at %s", trace.trace_id, trace_file
            )
            aggregate.per_trace.append(
                TraceRunResult(
                    trace_id=trace.trace_id,
                    status="error",
                    error_message=f"File not found: {trace_file}",
                )
            )
            continue

        overrides = trace_overrides.get(trace.trace_id, {})
        capacity = int(overrides.get("capacity_cpus", default_capacity))
        guard_k = float(overrides.get("runtime_guard_k", default_guard_k))
        seed = int(overrides.get("seed", default_seed))

        trace_result = run_credibility_protocol(
            trace_path=trace_file,
            trace_id=trace.trace_id,
            capacity_cpus=capacity,
            runtime_guard_k=guard_k,
            seed=seed,
            output_dir=output_dir,
            reference_suite_config=reference_suite_config,
            fidelity_config=fidelity_config,
            strict_invariants=strict_invariants,
        )
        aggregate.per_trace.append(trace_result)

    # Determine aggregate status
    statuses = [r.status for r in aggregate.per_trace]
    if all(s == "pass" for s in statuses):
        aggregate.status = "pass"
    elif any(s == "pass" for s in statuses):
        aggregate.status = "partial"
    else:
        aggregate.status = "fail"

    # Write aggregate summary
    summary_path = output_dir / "credibility_suite_summary.json"
    summary = {
        "timestamp_utc": timestamp,
        "suite_id": suite.suite_id,
        "status": aggregate.status,
        "traces": [
            {
                "trace_id": r.trace_id,
                "status": r.status,
                "fidelity_status": r.fidelity_status,
                "recommendation_status": r.recommendation_status,
                "error_message": r.error_message,
            }
            for r in aggregate.per_trace
        ],
    }
    write_json(summary_path, summary)

    logger.info("Suite credibility complete: status=%s", aggregate.status)
    return aggregate
