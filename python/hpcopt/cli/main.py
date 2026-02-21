from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import typer
import yaml

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.artifacts.benchmark import run_benchmark_suite
from hpcopt.artifacts.report_export import export_run_report
from hpcopt.data.reference_suite import (
    assert_reference_by_filename_and_hash,
    assert_reference_trace_hash_match,
    lock_reference_suite_hashes,
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
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace
from hpcopt.simulate.fidelity import run_baseline_fidelity_gate, run_candidate_fidelity_report
from hpcopt.simulate.objective import evaluate_constraint_contract
from hpcopt.simulate.batsim import (
    SUPPORTED_EDC_MODES,
    build_batsim_run_config,
    invoke_batsim_run,
    normalize_batsim_run_outputs,
)
from hpcopt.simulate.stress import generate_stress_scenario
from hpcopt.utils.io import ensure_dir, write_json

app = typer.Typer(help="HPC Workload Optimizer CLI")

ingest_app = typer.Typer(help="Ingestion commands")
profile_app = typer.Typer(help="Trace profiling commands")
features_app = typer.Typer(help="Feature pipeline commands")
train_app = typer.Typer(help="Training commands")
simulate_app = typer.Typer(help="Simulation commands")
stress_app = typer.Typer(help="Stress scenario commands")
recommend_app = typer.Typer(help="Recommendation commands")
report_app = typer.Typer(help="Reporting commands")
serve_app = typer.Typer(help="Service commands")
data_app = typer.Typer(help="Dataset contract commands")
credibility_app = typer.Typer(help="Credibility protocol commands")
analysis_app = typer.Typer(help="Analysis commands")
model_app = typer.Typer(help="Model management commands")
artifacts_app = typer.Typer(help="Artifact management commands")

app.add_typer(ingest_app, name="ingest")
app.add_typer(profile_app, name="profile")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(simulate_app, name="simulate")
app.add_typer(stress_app, name="stress")
app.add_typer(recommend_app, name="recommend")
app.add_typer(report_app, name="report")
app.add_typer(serve_app, name="serve")
app.add_typer(data_app, name="data")
app.add_typer(credibility_app, name="credibility")
app.add_typer(analysis_app, name="analysis")
app.add_typer(model_app, name="model")
app.add_typer(artifacts_app, name="artifacts")


# ──────────────────────── Ingest ────────────────────────

@ingest_app.command("swf")
def ingest_swf_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input SWF or SWF.GZ path"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
        help="Reference suite config for hash contract checks",
    ),
) -> None:
    ds_id = dataset_id or input.stem.replace(".swf", "")
    result = ingest_swf(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)

    reference_suite_match = assert_reference_trace_hash_match(
        trace_path=input,
        config_path=reference_suite_config,
    )
    if reference_suite_match is not None:
        metadata = json.loads(result.dataset_metadata_path.read_text(encoding="utf-8"))
        metadata["reference_suite"] = reference_suite_match
        write_json(result.dataset_metadata_path, metadata)

    manifest = build_manifest(
        command="hpcopt ingest swf",
        inputs=[input, reference_suite_config],
        outputs=[result.dataset_path, result.quality_report_path, result.dataset_metadata_path],
        params={
            "dataset_id": ds_id,
            "out_dir": str(out),
            "report_out": str(report_out),
            "reference_suite_match": reference_suite_match,
        },
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{ds_id}_run_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Dataset metadata: {result.dataset_metadata_path}")
    typer.echo(f"Quality report: {result.quality_report_path}")
    typer.echo(f"Run manifest: {manifest_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("slurm")
def ingest_slurm_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="sacct --parsable2 output file"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    from hpcopt.ingest.slurm import ingest_slurm
    ds_id = dataset_id or input.stem
    result = ingest_slurm(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("pbs")
def ingest_pbs_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="PBS/Torque accounting log"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    from hpcopt.ingest.pbs import ingest_pbs
    ds_id = dataset_id or input.stem
    result = ingest_pbs(input_path=input, out_dir=out, dataset_id=ds_id, report_dir=report_out)
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Rows: {result.row_count}")


@ingest_app.command("shadow-start")
def ingest_shadow_start_cmd(
    source_type: str = typer.Option("slurm", help="slurm|pbs"),
    source_path: Path = typer.Option(..., help="sacct output or accounting log path"),
    out: Path = typer.Option(Path("data/curated"), help="Output directory"),
    interval_sec: int = typer.Option(300, min=10, help="Polling interval in seconds"),
    watermark_path: Path = typer.Option(Path("outputs/shadow_watermark.json"), help="Watermark file"),
) -> None:
    from hpcopt.ingest.shadow import ShadowIngestionDaemon
    daemon = ShadowIngestionDaemon(
        out_dir=out,
        report_dir=out / "reports",
        watermark_path=watermark_path,
    )
    daemon.start(
        interval_sec=interval_sec,
        source_type=source_type,
        source_path=source_path,
        blocking=True,
    )


# ──────────────────────── Profile ────────────────────────

@profile_app.command("trace")
def profile_trace_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    out: Path = typer.Option(Path("outputs/reports"), help="Profile report output directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
) -> None:
    ds_id = dataset_id or dataset.stem
    result = build_trace_profile(dataset_path=dataset, report_dir=out, dataset_id=ds_id)
    manifest = build_manifest(
        command="hpcopt profile trace",
        inputs=[dataset],
        outputs=[result.profile_path],
        params={"dataset_id": ds_id, "report_out": str(out)},
        seeds=[],
    )
    manifest_path = out / f"{ds_id}_profile_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Profile report: {result.profile_path}")
    typer.echo(f"Profile manifest: {manifest_path}")
    typer.echo(f"Rows: {result.row_count}")


# ──────────────────────── Features ────────────────────────

@features_app.command("build")
def features_build_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    out: Path = typer.Option(Path("data/curated"), help="Feature dataset output directory"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Feature report output directory"),
    dataset_id: str | None = typer.Option(None, help="Dataset ID override"),
    n_folds: int = typer.Option(3, min=1, max=20, help="Number of chronological folds"),
    train_fraction: float = typer.Option(0.70, min=0.1, max=0.95, help="Train fraction per fold"),
    val_fraction: float = typer.Option(0.15, min=0.01, max=0.49, help="Validation fraction per fold"),
) -> None:
    if train_fraction + val_fraction >= 1.0:
        raise typer.BadParameter("train_fraction + val_fraction must be < 1.0")
    ds_id = dataset_id or dataset.stem
    result = build_feature_dataset(
        dataset_path=dataset, out_dir=out, report_dir=report_out, dataset_id=ds_id,
        n_folds=n_folds, train_fraction=train_fraction, val_fraction=val_fraction,
    )
    manifest = build_manifest(
        command="hpcopt features build",
        inputs=[dataset],
        outputs=[result.feature_dataset_path, result.split_manifest_path, result.feature_report_path],
        params={"dataset_id": ds_id, "n_folds": n_folds, "train_fraction": train_fraction, "val_fraction": val_fraction},
        seeds=[],
    )
    manifest_path = report_out / f"{ds_id}_features_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Feature dataset: {result.feature_dataset_path}")
    typer.echo(f"Features manifest: {manifest_path}")


# ──────────────────────── Train ────────────────────────

@train_app.command("runtime")
def train_runtime_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    out: Path = typer.Option(Path("outputs/models"), help="Model output directory"),
    model_id: str | None = typer.Option(None, help="Model id override"),
    seed: int = typer.Option(42, help="Training seed"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Run manifest output directory"),
    hyperparams_config: Path | None = typer.Option(None, help="Optional hyperparams YAML config"),
) -> None:
    resolved_model_id = model_id or f"runtime_quantile_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    hp = None
    if hyperparams_config is not None and hyperparams_config.exists():
        hp = yaml.safe_load(hyperparams_config.read_text(encoding="utf-8"))
    result = train_runtime_quantile_models(
        dataset_path=dataset, out_dir=out, model_id=resolved_model_id, seed=seed, hyperparams=hp,
    )
    manifest = build_manifest(
        command="hpcopt train runtime",
        inputs=[dataset],
        outputs=[result.metrics_path, result.metadata_path],
        params={"model_id": resolved_model_id, "seed": seed, "hyperparams": hp},
        seeds=[seed],
    )
    manifest_path = report_out / f"{resolved_model_id}_train_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Model dir: {result.model_dir}")
    typer.echo(f"Train manifest: {manifest_path}")


@train_app.command("tune")
def train_tune_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Training dataset"),
    out: Path = typer.Option(Path("outputs/reports"), help="Tuning report output directory"),
    quantile: float = typer.Option(0.5, help="Target quantile"),
    seed: int = typer.Option(42, help="Tuning seed"),
    n_trials: int = typer.Option(20, min=1, help="Number of search trials"),
    n_folds: int = typer.Option(3, min=1, help="Chronological CV folds"),
) -> None:
    from hpcopt.models.tuning import build_tuning_report
    ensure_dir(out)
    report_path = out / f"tuning_q{quantile:.2f}_report.json"
    result = build_tuning_report(
        dataset_path=dataset, out_path=report_path,
        quantile=quantile, seed=seed, n_trials=n_trials, n_folds=n_folds,
    )
    typer.echo(f"Best params: {result.best_params.to_dict()}")
    typer.echo(f"Best score: {result.best_score:.6f}")
    typer.echo(f"Tuning report: {result.report_path}")


@train_app.command("resource-fit")
def train_resource_fit_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Training dataset"),
    out: Path = typer.Option(Path("outputs/models"), help="Model output directory"),
    model_id: str | None = typer.Option(None, help="Model id override"),
    seed: int = typer.Option(42, help="Training seed"),
) -> None:
    from hpcopt.models.resource_fit import train_resource_fit_model
    resolved_id = model_id or f"resource_fit_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    result = train_resource_fit_model(dataset_path=dataset, out_dir=out, model_id=resolved_id, seed=seed)
    typer.echo(f"Model dir: {result.model_dir}")
    typer.echo(f"Metrics: {result.metrics_path}")


# ──────────────────────── Simulate ────────────────────────

@simulate_app.command("run")
def simulate_run_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    policy: str = typer.Option("FIFO_STRICT", help="FIFO_STRICT|EASY_BACKFILL_BASELINE|ML_BACKFILL_P50"),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    out: Path = typer.Option(Path("outputs/simulations"), help="Simulation artifact output directory"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    run_id: str | None = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(False, help="Fail on first invariant violation"),
    runtime_model_dir: Path | None = typer.Option(None, help="Runtime quantile model directory"),
    runtime_guard_k: float = typer.Option(0.5, min=0.0, max=2.0, help="ML backfill runtime guard coefficient"),
    strict_uncertainty_mode: bool = typer.Option(False, help="ML policy strict mode"),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"), exists=True, readable=True,
        help="Reference suite config for trace hash checks",
    ),
) -> None:
    if policy not in SUPPORTED_POLICIES:
        raise typer.BadParameter(f"Unsupported policy '{policy}'. Supported: {sorted(SUPPORTED_POLICIES)}")
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"sim_{policy.lower()}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)

    runtime_predictor = None
    resolved_model_dir = None
    if policy == "ML_BACKFILL_P50":
        resolved_model_dir = resolve_runtime_model_dir(runtime_model_dir)
        if resolved_model_dir is not None:
            runtime_predictor = RuntimeQuantilePredictor(resolved_model_dir)

    result = run_simulation_from_trace(
        trace_df=trace_df, policy_id=policy, capacity_cpus=capacity_cpus,
        run_id=resolved_run_id, strict_invariants=strict_invariants,
        runtime_predictor=runtime_predictor, runtime_guard_k=runtime_guard_k,
        strict_uncertainty_mode=strict_uncertainty_mode,
    )

    jobs_path = out / f"{resolved_run_id}_{policy.lower()}_jobs.parquet"
    queue_path = out / f"{resolved_run_id}_{policy.lower()}_queue.parquet"
    sim_report_path = report_out / f"{resolved_run_id}_{policy.lower()}_sim_report.json"
    invariant_path = report_out / f"{resolved_run_id}_{policy.lower()}_invariants.json"

    result.jobs_df.to_parquet(jobs_path, index=False)
    result.queue_series_df.to_parquet(queue_path, index=False)
    source_reference_match = None
    trace_metadata_path = trace.with_suffix(".metadata.json")
    if trace_metadata_path.exists():
        trace_meta = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        source_reference_match = assert_reference_by_filename_and_hash(
            filename=str(trace_meta.get("source_trace_filename", "")),
            sha256_observed=trace_meta.get("source_trace_sha256"),
            config_path=reference_suite_config,
        )
    write_json(sim_report_path, {
        "run_id": resolved_run_id, "policy_id": policy, "status": "ok",
        "metrics": result.metrics, "objective_metrics": result.objective_metrics,
        "fallback_accounting": result.fallback_accounting,
        "model_dir": str(resolved_model_dir) if resolved_model_dir is not None else None,
        "source_trace_reference_suite": source_reference_match,
        "jobs_artifact": str(jobs_path), "queue_artifact": str(queue_path),
    })
    write_json(invariant_path, result.invariant_report)

    manifest = build_manifest(
        command="hpcopt simulate run", inputs=[trace, reference_suite_config],
        outputs=[jobs_path, queue_path, sim_report_path, invariant_path],
        params={
            "run_id": resolved_run_id, "policy_id": policy, "capacity_cpus": capacity_cpus,
            "strict_invariants": strict_invariants,
            "runtime_model_dir": str(resolved_model_dir) if resolved_model_dir else None,
            "runtime_guard_k": runtime_guard_k,
        },
        config_paths=[reference_suite_config], seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_{policy.lower()}_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Simulation report: {sim_report_path}")
    typer.echo(f"Manifest: {manifest_path}")


@simulate_app.command("fidelity-gate")
def simulate_fidelity_gate_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    config: Path = typer.Option(
        Path("configs/simulation/fidelity_gate.yaml"), exists=True, readable=True,
        help="Fidelity gate threshold config",
    ),
    out: Path = typer.Option(Path("outputs/reports"), help="Fidelity report output directory"),
    run_id: str | None = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(True, help="Fail if invariants fail"),
) -> None:
    ensure_dir(out)
    resolved_run_id = run_id or f"fidelity_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    report_path = out / f"{resolved_run_id}_fidelity_report.json"
    result = run_baseline_fidelity_gate(
        trace_df=trace_df, capacity_cpus=capacity_cpus, out_path=report_path,
        run_id=resolved_run_id, config_path=config, strict_invariants=strict_invariants,
    )
    manifest = build_manifest(
        command="hpcopt simulate fidelity-gate", inputs=[trace, config],
        outputs=[result.report_path],
        params={"run_id": resolved_run_id, "capacity_cpus": capacity_cpus},
        config_paths=[config], seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_fidelity_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Fidelity status: {result.status}")
    typer.echo(f"Fidelity report: {result.report_path}")


@simulate_app.command("batsim-config")
def simulate_batsim_config_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    policy: str = typer.Option("FIFO_STRICT"),
    out: Path = typer.Option(Path("outputs/simulations")),
    run_id: str | None = typer.Option(None),
    platform_path: Path | None = typer.Option(None),
    workload_path: Path | None = typer.Option(None),
    capacity_cpus: int | None = typer.Option(None, min=1),
    edc_mode: str = typer.Option("library_file"),
    edc_library_path: str | None = typer.Option(None),
    edc_socket_endpoint: str | None = typer.Option(None),
    edc_init_json: str = typer.Option("{}"),
    export_prefix: Path | None = typer.Option(None),
    use_wsl_defaults: bool = typer.Option(True),
    wsl_distro: str = typer.Option("Ubuntu"),
    report_out: Path = typer.Option(Path("outputs/reports")),
) -> None:
    if edc_mode not in SUPPORTED_EDC_MODES:
        raise typer.BadParameter(f"Unsupported edc_mode '{edc_mode}'")
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"batsim_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    config = build_batsim_run_config(
        run_id=resolved_run_id, trace_dataset=trace, policy_id=policy,
        out_dir=out, platform_path=platform_path, workload_path=workload_path,
        capacity_cpus=capacity_cpus, edc_mode=edc_mode, edc_library_path=edc_library_path,
        edc_socket_endpoint=edc_socket_endpoint, edc_init_json=edc_init_json,
        export_prefix=export_prefix, use_wsl_defaults=use_wsl_defaults, wsl_distro=wsl_distro,
    )
    typer.echo(f"Batsim config: {config.config_path}")


@simulate_app.command("batsim-run")
def simulate_batsim_run_cmd(
    config: Path = typer.Option(..., exists=True, readable=True),
    batsim_bin: str = typer.Option("batsim"),
    dry_run: bool = typer.Option(True),
    use_wsl: bool = typer.Option(False),
    wsl_distro: str = typer.Option("Ubuntu"),
    wsl_load_nix_profile: bool = typer.Option(True),
    normalize_to_sim_report: bool = typer.Option(True),
    simulation_out: Path = typer.Option(Path("outputs/simulations")),
    emit_fidelity_report: bool = typer.Option(True),
    fidelity_config: Path | None = typer.Option(Path("configs/simulation/fidelity_gate.yaml")),
    out: Path = typer.Option(Path("outputs/reports")),
) -> None:
    ensure_dir(out)
    ensure_dir(simulation_out)
    result = invoke_batsim_run(
        config_path=config, batsim_bin=batsim_bin, dry_run=dry_run,
        use_wsl=use_wsl, wsl_distro=wsl_distro, wsl_load_nix_profile=wsl_load_nix_profile,
    )
    report_payload: dict[str, object] = {
        "config_path": str(config), "status": result.status, "reason": result.reason,
        "returncode": result.returncode, "command": result.command,
    }
    report_path = out / f"{config.stem}_batsim_run_report.json"
    write_json(report_path, report_payload)
    typer.echo(f"Batsim run status: {result.status}")
    typer.echo(f"Run report: {report_path}")


@simulate_app.command("replay-baselines")
def simulate_replay_baselines_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    capacity_cpus: int = typer.Option(64, min=1),
    out: Path = typer.Option(Path("outputs/simulations")),
    report_out: Path = typer.Option(Path("outputs/reports")),
    run_id: str | None = typer.Option(None),
    strict_invariants: bool = typer.Option(True),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"), exists=True, readable=True,
    ),
) -> None:
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"baseline_replay_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    policies = ("FIFO_STRICT", "EASY_BACKFILL_BASELINE")
    combined: dict[str, dict[str, object]] = {}
    outputs: list[Path] = []

    for policy in policies:
        sim = run_simulation_from_trace(
            trace_df=trace_df, policy_id=policy, capacity_cpus=capacity_cpus,
            run_id=f"{resolved_run_id}_{policy.lower()}", strict_invariants=strict_invariants,
        )
        jobs_path = out / f"{resolved_run_id}_{policy.lower()}_jobs.parquet"
        queue_path = out / f"{resolved_run_id}_{policy.lower()}_queue.parquet"
        inv_path = report_out / f"{resolved_run_id}_{policy.lower()}_invariants.json"
        sim.jobs_df.to_parquet(jobs_path, index=False)
        sim.queue_series_df.to_parquet(queue_path, index=False)
        write_json(inv_path, sim.invariant_report)
        outputs.extend([jobs_path, queue_path, inv_path])
        combined[policy] = {
            "metrics": sim.metrics, "objective_metrics": sim.objective_metrics,
            "fallback_accounting": sim.fallback_accounting,
        }

    summary_path = report_out / f"{resolved_run_id}_baseline_replay_report.json"
    write_json(summary_path, {"run_id": resolved_run_id, "trace": str(trace), "capacity_cpus": capacity_cpus, "policies": combined})
    manifest = build_manifest(
        command="hpcopt simulate replay-baselines", inputs=[trace, reference_suite_config],
        outputs=outputs + [summary_path], params={"run_id": resolved_run_id, "capacity_cpus": capacity_cpus},
        config_paths=[reference_suite_config], seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_baseline_replay_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Baseline replay report: {summary_path}")


# ──────────────────────── Stress ────────────────────────

@stress_app.command("gen")
def stress_gen_cmd(
    scenario: str = typer.Option(..., help="heavy_tail|low_congestion|user_skew|burst_shock"),
    out: Path = typer.Option(Path("data/curated")),
    n_jobs: int = typer.Option(5000, min=100),
    seed: int = typer.Option(42),
    alpha: float = typer.Option(1.2),
    target_util: float = typer.Option(0.35),
    top_user_share: float = typer.Option(0.65),
    burst_factor: int = typer.Option(4),
    burst_duration_sec: int = typer.Option(1800),
) -> None:
    params = {"alpha": alpha, "target_util": target_util, "top_user_share": top_user_share,
              "burst_factor": burst_factor, "burst_duration_sec": burst_duration_sec}
    result = generate_stress_scenario(scenario=scenario, out_dir=out, n_jobs=n_jobs, seed=seed, params=params)
    typer.echo(f"Stress dataset: {result.dataset_path}")


@stress_app.command("run")
def stress_run_cmd(
    scenario: str = typer.Option(...),
    policy: Path = typer.Option(..., exists=True, readable=True),
    model: str = typer.Option(...),
    dataset: Path | None = typer.Option(None),
    capacity_cpus: int = typer.Option(64, min=1),
    baseline_policy: str = typer.Option("EASY_BACKFILL_BASELINE"),
    out: Path = typer.Option(Path("outputs/simulations")),
    report_out: Path = typer.Option(Path("outputs/reports")),
    run_id: str | None = typer.Option(None),
    strict_invariants: bool = typer.Option(True),
    runtime_model_dir: Path | None = typer.Option(None),
) -> None:
    if baseline_policy not in SUPPORTED_POLICIES:
        raise typer.BadParameter(f"Unsupported baseline policy '{baseline_policy}'")
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_dataset = dataset or (Path("data/curated") / f"stress_{scenario}.parquet")
    if not resolved_dataset.exists():
        raise typer.BadParameter(f"Stress dataset does not exist: {resolved_dataset}")

    cfg = yaml.safe_load(policy.read_text(encoding="utf-8"))
    policy_id = str(cfg.get("policy_id", "ML_BACKFILL_P50"))
    runtime_guard_k = float(cfg.get("runtime_guard_k", 0.5))
    resolved_run_id = run_id or f"stress_{scenario}_{policy_id.lower()}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(resolved_dataset)

    runtime_predictor = None
    resolved_model_dir = None
    if policy_id == "ML_BACKFILL_P50":
        resolved_model_dir = resolve_runtime_model_dir(runtime_model_dir)
        if resolved_model_dir is not None:
            runtime_predictor = RuntimeQuantilePredictor(resolved_model_dir)

    baseline_sim = run_simulation_from_trace(
        trace_df=trace_df, policy_id=baseline_policy, capacity_cpus=capacity_cpus,
        run_id=f"{resolved_run_id}_{baseline_policy.lower()}", strict_invariants=strict_invariants,
    )
    candidate_sim = run_simulation_from_trace(
        trace_df=trace_df, policy_id=policy_id, capacity_cpus=capacity_cpus,
        run_id=f"{resolved_run_id}_{policy_id.lower()}", strict_invariants=strict_invariants,
        runtime_predictor=runtime_predictor, runtime_guard_k=runtime_guard_k,
    )
    fairness_cfg = cfg.get("fairness", {})
    constraints = evaluate_constraint_contract(
        candidate=candidate_sim.objective_metrics, baseline=baseline_sim.objective_metrics,
        starvation_rate_max=float(fairness_cfg.get("starvation_rate_max", 0.02)),
        fairness_dev_delta_max=float(fairness_cfg.get("fairness_dev_delta_max", 0.05)),
        jain_delta_max=float(fairness_cfg.get("jain_delta_max", 0.03)),
    )
    stress_status = "pass" if constraints["constraints_passed"] else "fail"

    # Compute degrade signatures: compare candidate vs baseline objective metrics
    degrade_signatures: dict[str, object] = {}
    for key in candidate_sim.objective_metrics:
        cval = candidate_sim.objective_metrics.get(key)
        bval = baseline_sim.objective_metrics.get(key)
        if isinstance(cval, (int, float)) and isinstance(bval, (int, float)) and bval != 0:
            delta = cval - bval
            ratio = delta / abs(bval)
            degrade_signatures[key] = {"candidate": cval, "baseline": bval, "delta": delta, "ratio": round(ratio, 6)}

    stress_report_path = report_out / f"{resolved_run_id}_stress_report.json"
    write_json(stress_report_path, {
        "run_id": resolved_run_id, "scenario": scenario, "status": stress_status,
        "constraints": constraints, "candidate_policy_id": policy_id,
        "baseline_policy_id": baseline_policy, "degrade_signatures": degrade_signatures,
    })

    manifest = build_manifest(
        command="hpcopt stress run",
        inputs=[resolved_dataset, policy],
        outputs=[stress_report_path],
        params={
            "run_id": resolved_run_id, "scenario": scenario, "policy_id": policy_id,
            "baseline_policy": baseline_policy, "capacity_cpus": capacity_cpus,
        },
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_stress_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Stress status: {stress_status}")
    typer.echo(f"Stress report: {stress_report_path}")
    typer.echo(f"Stress manifest: {manifest_path}")


# ──────────────────────── Recommend ────────────────────────

@recommend_app.command("generate")
def recommend_generate_cmd(
    baseline_report: Path = typer.Option(..., exists=True, readable=True),
    candidate_report: list[Path] = typer.Option(..., exists=True, readable=True),
    fidelity_report: Path | None = typer.Option(None, exists=True, readable=True),
    out: Path = typer.Option(Path("outputs/reports")),
    run_id: str | None = typer.Option(None),
    w1: float = typer.Option(1.0),
    w2: float = typer.Option(0.3),
    w3: float = typer.Option(2.0),
    pareto: bool = typer.Option(False, help="Use Pareto multi-objective mode"),
) -> None:
    ensure_dir(out)
    resolved_run_id = run_id or f"recommend_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    recommendation_path = out / f"{resolved_run_id}_recommendation_report.json"

    if pareto:
        from hpcopt.recommend.engine import generate_pareto_recommendation
        result = generate_pareto_recommendation(
            baseline_report_path=baseline_report,
            candidate_report_paths=candidate_report,
            out_path=recommendation_path,
        )
    else:
        result = generate_recommendation_report(
            baseline_report_path=baseline_report, candidate_report_paths=candidate_report,
            out_path=recommendation_path, fidelity_report_path=fidelity_report,
            w1=w1, w2=w2, w3=w3,
        )

    manifest = build_manifest(
        command="hpcopt recommend generate",
        inputs=[baseline_report, *candidate_report, *([fidelity_report] if fidelity_report else [])],
        outputs=[result.report_path],
        params={"run_id": resolved_run_id, "weights": {"w1": w1, "w2": w2, "w3": w3}, "pareto": pareto},
        seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_recommend_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Recommendation status: {result.payload.get('status', 'unknown')}")
    typer.echo(f"Recommendation report: {result.report_path}")


# ──────────────────────── Report ────────────────────────

@report_app.command("export")
def report_export_cmd(
    run_id: str = typer.Option(...),
    out: Path = typer.Option(Path("outputs/reports")),
    format: str = typer.Option("both", help="json|md|both"),
) -> None:
    if format not in {"json", "md", "both"}:
        raise typer.BadParameter("format must be one of: json|md|both")
    ensure_dir(out)
    result = export_run_report(run_id=run_id, out_dir=out)
    if format in {"json", "both"}:
        typer.echo(f"Export json: {result.json_path}")
    if format in {"md", "both"}:
        typer.echo(f"Export md: {result.md_path}")


@report_app.command("benchmark")
def report_benchmark_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    raw_trace: Path | None = typer.Option(None, exists=True, readable=True),
    policy: str = typer.Option("FIFO_STRICT"),
    capacity_cpus: int = typer.Option(64, min=1),
    samples: int = typer.Option(3, min=1, max=20),
    regression_max_drop: float = typer.Option(0.10),
    history_window: int = typer.Option(5, min=1),
    strict_regression: bool = typer.Option(False),
    out: Path = typer.Option(Path("outputs/reports")),
    history: Path = typer.Option(Path("outputs/reports/benchmark_history.jsonl")),
    run_id: str | None = typer.Option(None),
) -> None:
    if policy not in SUPPORTED_POLICIES:
        raise typer.BadParameter(f"Unsupported policy '{policy}'")
    ensure_dir(out)
    resolved_run_id = run_id or f"benchmark_{trace.stem}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    report_path = out / f"{resolved_run_id}_benchmark_report.json"
    result = run_benchmark_suite(
        trace_dataset=trace, report_path=report_path, history_path=history,
        raw_trace=raw_trace, policy_id=policy, capacity_cpus=capacity_cpus,
        samples=samples, regression_max_drop=regression_max_drop, history_window=history_window,
    )
    manifest = build_manifest(
        command="hpcopt report benchmark", inputs=[trace], outputs=[result.report_path, result.history_path],
        params={"run_id": resolved_run_id, "policy": policy, "samples": samples}, seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_benchmark_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Benchmark status: {result.status}")
    typer.echo(f"Benchmark report: {result.report_path}")
    if strict_regression and result.regression_fail:
        raise typer.Exit(code=1)


# ──────────────────────── Credibility ────────────────────────

@credibility_app.command("run-suite")
def credibility_run_suite_cmd(
    config: Path = typer.Option(
        Path("configs/credibility/default_sweep.yaml"), help="Sweep config YAML",
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Raw trace directory"),
    out: Path = typer.Option(Path("outputs/credibility"), help="Output directory"),
    reference_suite_config: Path = typer.Option(Path("configs/data/reference_suite.yaml")),
    fidelity_config: Path = typer.Option(Path("configs/simulation/fidelity_gate.yaml")),
    strict_invariants: bool = typer.Option(True),
) -> None:
    from hpcopt.orchestrate.credibility import run_suite_credibility
    result = run_suite_credibility(
        reference_suite_config=reference_suite_config,
        sweep_config_path=config if config.exists() else None,
        raw_dir=raw_dir, output_dir=out, fidelity_config=fidelity_config,
        strict_invariants=strict_invariants,
    )
    typer.echo(f"Suite status: {result.status}")
    for tr in result.per_trace:
        typer.echo(f"  {tr.trace_id}: {tr.status} (fidelity={tr.fidelity_status}, rec={tr.recommendation_status})")


@credibility_app.command("dossier")
def credibility_dossier_cmd(
    input_dir: Path = typer.Option(Path("outputs/credibility"), help="Credibility run output directory"),
    out: Path = typer.Option(Path("outputs/credibility/dossier"), help="Dossier output directory"),
) -> None:
    from hpcopt.artifacts.credibility_dossier import assemble_credibility_dossier
    result = assemble_credibility_dossier(input_dir=input_dir, output_path=out)
    typer.echo(f"Dossier JSON: {result.json_path}")
    typer.echo(f"Dossier MD: {result.md_path}")


# ──────────────────────── Analysis ────────────────────────

@analysis_app.command("sensitivity-sweep")
def analysis_sensitivity_sweep_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True),
    capacity_cpus: int = typer.Option(64, min=1),
    model_dir: Path | None = typer.Option(None),
    out: Path = typer.Option(Path("outputs/reports")),
    seed: int = typer.Option(42),
    k_values: str = typer.Option("0.0,0.25,0.5,0.75,1.0,1.5", help="Comma-separated k values"),
) -> None:
    from hpcopt.analysis.sensitivity import run_guard_k_sweep, build_sensitivity_report
    ensure_dir(out)
    trace_df = pd.read_parquet(trace)
    k_list = [float(k.strip()) for k in k_values.split(",")]
    resolved_model = resolve_runtime_model_dir(model_dir)

    sweep = run_guard_k_sweep(
        trace_df=trace_df, capacity_cpus=capacity_cpus, k_values=k_list,
        model_dir=resolved_model, seed=seed,
    )
    report_path = out / f"sensitivity_sweep_{trace.stem}.json"
    result = build_sensitivity_report(sweep_results=sweep, out_path=report_path)
    typer.echo(f"Sensitivity report: {result.report_path}")
    if result.payload["analysis"]["optimal_k"] is not None:
        typer.echo(f"Optimal k: {result.payload['analysis']['optimal_k']}")


@analysis_app.command("feature-importance")
def analysis_feature_importance_cmd(
    model_dir: Path = typer.Option(..., exists=True, readable=True),
    dataset: Path = typer.Option(..., exists=True, readable=True),
    out: Path = typer.Option(Path("outputs/reports")),
    n_repeats: int = typer.Option(10, min=1),
    seed: int = typer.Option(42),
) -> None:
    from hpcopt.analysis.feature_importance import build_importance_report
    ensure_dir(out)
    report_path = out / f"feature_importance_{model_dir.name}.json"
    result = build_importance_report(
        model_dir=model_dir, dataset_path=dataset, out_path=report_path,
        n_repeats=n_repeats, seed=seed,
    )
    typer.echo(f"Feature importance report: {result.report_path}")


# ──────────────────────── Model Management ────────────────────────

@model_app.command("list")
def model_list_cmd() -> None:
    from hpcopt.models.registry import ModelRegistry
    registry = ModelRegistry()
    models = registry.list()
    if not models:
        typer.echo("No models registered.")
        return
    for m in models:
        status = m.get("status", "unknown")
        marker = " [PRODUCTION]" if status == "production" else ""
        typer.echo(f"  {m['model_id']} ({status}){marker} - {m.get('registered_at', 'unknown')}")


@model_app.command("promote")
def model_promote_cmd(
    model_id: str = typer.Option(..., help="Model ID to promote to production"),
) -> None:
    from hpcopt.models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.promote(model_id)
    typer.echo(f"Model '{model_id}' promoted to production.")


@model_app.command("archive")
def model_archive_cmd(
    model_id: str = typer.Option(..., help="Model ID to archive"),
) -> None:
    from hpcopt.models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.archive(model_id)
    typer.echo(f"Model '{model_id}' archived.")


@model_app.command("drift-check")
def model_drift_check_cmd(
    eval_dataset: Path = typer.Option(..., exists=True, readable=True),
    model_dir: Path | None = typer.Option(None),
    out: Path = typer.Option(Path("outputs/reports")),
) -> None:
    from hpcopt.models.drift import compute_drift_report
    ensure_dir(out)
    resolved_model = resolve_runtime_model_dir(model_dir)
    if resolved_model is None:
        raise typer.BadParameter("No model directory found.")
    report = compute_drift_report(model_dir=resolved_model, eval_dataset_path=eval_dataset)
    report_path = out / f"drift_report_{resolved_model.name}.json"
    write_json(report_path, report.to_dict())
    status = "drift_detected" if report.overall_drift_detected else "ok"
    typer.echo(f"Drift status: {status}")
    typer.echo(f"Drift report: {report_path}")


# ──────────────────────── Artifacts ────────────────────────

@artifacts_app.command("cleanup")
def artifacts_cleanup_cmd(
    outputs_dir: Path = typer.Option(Path("outputs"), help="Outputs directory to clean"),
    max_age_days: int = typer.Option(90, min=1, help="Maximum age in days"),
    dry_run: bool = typer.Option(True, help="Preview only, do not delete"),
) -> None:
    from hpcopt.artifacts.retention import cleanup_artifacts
    result = cleanup_artifacts(outputs_dir=outputs_dir, max_age_days=max_age_days, dry_run=dry_run)
    summary = result.get("summary", {})
    count_key = "would_delete_count" if dry_run else "deleted_count"
    count = summary.get(count_key, 0)
    typer.echo(f"Cleanup {'preview' if dry_run else 'completed'}: {count} artifacts")


# ──────────────────────── Data ────────────────────────

@data_app.command("lock-reference-suite")
def lock_reference_suite_cmd(
    config: Path = typer.Option(Path("configs/data/reference_suite.yaml"), exists=True, readable=True),
    raw_dir: Path = typer.Option(Path("data/raw")),
    out: Path = typer.Option(Path("outputs/reports/reference_suite_lock_report.json")),
    strict_missing: bool = typer.Option(False),
) -> None:
    report = lock_reference_suite_hashes(config_path=config, raw_dir=raw_dir, out_report_path=out, strict_missing=strict_missing)
    typer.echo(f"Suite lock updated: {report['updated']}")
    typer.echo(f"Missing files: {len(report['missing_files'])}")


# ──────────────────────── Serve ────────────────────────

@serve_app.command("api")
def serve_api_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8080, min=1, max=65535, help="Bind port"),
) -> None:
    import uvicorn
    uvicorn.run("hpcopt.api.app:app", host=host, port=port, reload=False)


def run() -> None:
    from hpcopt.utils.logging import setup_logging
    try:
        setup_logging(level="INFO", format_mode="structured")
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app()
