from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.artifacts.report_export import export_run_report
from hpcopt.data.reference_suite import (
    assert_reference_by_filename_and_hash,
    assert_reference_trace_hash_match,
    lock_reference_suite_hashes,
)
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


@ingest_app.command("swf")
def ingest_swf_cmd(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input SWF or SWF.GZ path"),
    out: Path = typer.Option(Path("data/curated"), help="Output curated dataset directory"),
    dataset_id: Optional[str] = typer.Option(None, help="Dataset ID override"),
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


@profile_app.command("trace")
def profile_trace_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    out: Path = typer.Option(Path("outputs/reports"), help="Profile report output directory"),
    dataset_id: Optional[str] = typer.Option(None, help="Dataset ID override"),
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


@stress_app.command("gen")
def stress_gen_cmd(
    scenario: str = typer.Option(..., help="heavy_tail|low_congestion|user_skew|burst_shock"),
    out: Path = typer.Option(Path("data/curated"), help="Output dataset directory"),
    n_jobs: int = typer.Option(5000, min=100, help="Number of jobs"),
    seed: int = typer.Option(42, help="Random seed"),
    alpha: float = typer.Option(1.2, help="heavy_tail alpha"),
    target_util: float = typer.Option(0.35, help="low_congestion target util"),
    top_user_share: float = typer.Option(0.65, help="user_skew top-user share"),
    burst_factor: int = typer.Option(4, help="burst_shock burst factor"),
    burst_duration_sec: int = typer.Option(1800, help="burst_shock burst duration"),
) -> None:
    params = {
        "alpha": alpha,
        "target_util": target_util,
        "top_user_share": top_user_share,
        "burst_factor": burst_factor,
        "burst_duration_sec": burst_duration_sec,
    }
    result = generate_stress_scenario(
        scenario=scenario,
        out_dir=out,
        n_jobs=n_jobs,
        seed=seed,
        params=params,
    )
    typer.echo(f"Stress dataset: {result.dataset_path}")
    typer.echo(f"Stress metadata: {result.metadata_path}")


@stress_app.command("run")
def stress_run_cmd(
    scenario: str = typer.Option(..., help="Scenario previously generated"),
    policy: Path = typer.Option(..., exists=True, readable=True, help="Policy config"),
    model: str = typer.Option(..., help="Model version or artifact id"),
) -> None:
    # Placeholder execution hook. Full simulation integration is part of P0-12+ work.
    typer.echo("Stress run orchestration is scaffolded but not yet implemented.")
    typer.echo(f"Scenario: {scenario}")
    typer.echo(f"Policy: {policy}")
    typer.echo(f"Model: {model}")


@features_app.command("build")
def features_build_cmd() -> None:
    typer.echo("Feature pipeline CLI is scaffolded; implementation follows P0-08/P0-09.")


@train_app.command("runtime")
def train_runtime_cmd(
    dataset: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    out: Path = typer.Option(Path("outputs/models"), help="Model output directory"),
    model_id: Optional[str] = typer.Option(None, help="Model id override"),
    seed: int = typer.Option(42, help="Training seed"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Run manifest output directory"),
) -> None:
    resolved_model_id = model_id or f"runtime_quantile_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    result = train_runtime_quantile_models(
        dataset_path=dataset,
        out_dir=out,
        model_id=resolved_model_id,
        seed=seed,
    )
    manifest = build_manifest(
        command="hpcopt train runtime",
        inputs=[dataset],
        outputs=[result.metrics_path, result.metadata_path],
        params={
            "model_id": resolved_model_id,
            "model_dir": str(result.model_dir),
            "seed": seed,
        },
        seeds=[seed],
    )
    manifest_path = report_out / f"{resolved_model_id}_train_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Model dir: {result.model_dir}")
    typer.echo(f"Metrics: {result.metrics_path}")
    typer.echo(f"Metadata: {result.metadata_path}")
    typer.echo(f"Train manifest: {manifest_path}")


@simulate_app.command("run")
def simulate_run_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    policy: str = typer.Option(
        "FIFO_STRICT",
        help="FIFO_STRICT|EASY_BACKFILL_BASELINE|ML_BACKFILL_P50",
    ),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    out: Path = typer.Option(Path("outputs/simulations"), help="Simulation artifact output directory"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(False, help="Fail on first invariant violation"),
    runtime_model_dir: Optional[Path] = typer.Option(
        None,
        help="Optional runtime quantile model directory (required for ML predictions; fallback used if omitted)",
    ),
    runtime_guard_k: float = typer.Option(
        0.5,
        min=0.0,
        max=2.0,
        help="ML backfill runtime guard coefficient",
    ),
    strict_uncertainty_mode: bool = typer.Option(
        False,
        help="ML policy strict mode: backfill gate uses p90 instead of runtime_guard",
    ),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
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
        trace_df=trace_df,
        policy_id=policy,
        capacity_cpus=capacity_cpus,
        run_id=resolved_run_id,
        strict_invariants=strict_invariants,
        runtime_predictor=runtime_predictor,
        runtime_guard_k=runtime_guard_k,
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
    write_json(
        sim_report_path,
        {
            "run_id": resolved_run_id,
            "policy_id": policy,
            "status": "ok",
            "metrics": result.metrics,
            "objective_metrics": result.objective_metrics,
            "fallback_accounting": result.fallback_accounting,
            "model_dir": str(resolved_model_dir) if resolved_model_dir is not None else None,
            "source_trace_reference_suite": source_reference_match,
            "jobs_artifact": str(jobs_path),
            "queue_artifact": str(queue_path),
        },
    )
    write_json(invariant_path, result.invariant_report)

    manifest = build_manifest(
        command="hpcopt simulate run",
        inputs=[trace, reference_suite_config],
        outputs=[jobs_path, queue_path, sim_report_path, invariant_path],
        params={
            "run_id": resolved_run_id,
            "policy_id": policy,
            "capacity_cpus": capacity_cpus,
            "strict_invariants": strict_invariants,
            "runtime_model_dir": str(resolved_model_dir) if resolved_model_dir else None,
            "runtime_guard_k": runtime_guard_k,
            "strict_uncertainty_mode": strict_uncertainty_mode,
            "source_trace_reference_suite": source_reference_match,
        },
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_{policy.lower()}_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Simulation report: {sim_report_path}")
    typer.echo(f"Invariant report: {invariant_path}")
    typer.echo(f"Jobs artifact: {jobs_path}")
    typer.echo(f"Queue artifact: {queue_path}")
    typer.echo(f"Manifest: {manifest_path}")


@simulate_app.command("fidelity-gate")
def simulate_fidelity_gate_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    config: Path = typer.Option(
        Path("configs/simulation/fidelity_gate.yaml"),
        exists=True,
        readable=True,
        help="Fidelity gate threshold config",
    ),
    out: Path = typer.Option(Path("outputs/reports"), help="Fidelity report output directory"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(True, help="Fail if invariants fail during baseline replay"),
) -> None:
    ensure_dir(out)
    resolved_run_id = run_id or f"fidelity_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    report_path = out / f"{resolved_run_id}_fidelity_report.json"

    result = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=capacity_cpus,
        out_path=report_path,
        run_id=resolved_run_id,
        config_path=config,
        strict_invariants=strict_invariants,
    )
    manifest = build_manifest(
        command="hpcopt simulate fidelity-gate",
        inputs=[trace, config],
        outputs=[result.report_path],
        params={
            "run_id": resolved_run_id,
            "capacity_cpus": capacity_cpus,
            "strict_invariants": strict_invariants,
        },
        config_paths=[config],
        seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_fidelity_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Fidelity status: {result.status}")
    typer.echo(f"Fidelity report: {result.report_path}")
    typer.echo(f"Fidelity manifest: {manifest_path}")


@simulate_app.command("batsim-config")
def simulate_batsim_config_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset or Batsim workload json"),
    policy: str = typer.Option("FIFO_STRICT", help="Policy id"),
    out: Path = typer.Option(Path("outputs/simulations"), help="Output directory"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier override"),
    platform_path: Optional[Path] = typer.Option(None, help="Optional Batsim platform xml; generated if omitted"),
    workload_path: Optional[Path] = typer.Option(None, help="Optional Batsim workload json; generated from parquet if omitted"),
    capacity_cpus: Optional[int] = typer.Option(None, min=1, help="Optional CPU capacity override"),
    edc_mode: str = typer.Option("library_file", help="library_file|library_str|socket_file|socket_str"),
    edc_library_path: Optional[str] = typer.Option(
        None,
        help="EDC library path (e.g., /home/<user>/.nix-profile/lib/libfcfs.so)",
    ),
    edc_socket_endpoint: Optional[str] = typer.Option(None, help="EDC socket endpoint for socket modes"),
    edc_init_json: str = typer.Option("{}", help="EDC init JSON payload"),
    export_prefix: Optional[Path] = typer.Option(None, help="Optional Batsim export prefix path"),
    use_wsl_defaults: bool = typer.Option(
        True,
        help="Auto-resolve default fcfs EDC path from WSL HOME when library mode is used",
    ),
    wsl_distro: str = typer.Option("Ubuntu", help="WSL distro name for default-path resolution"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
) -> None:
    if edc_mode not in SUPPORTED_EDC_MODES:
        raise typer.BadParameter(f"Unsupported edc_mode '{edc_mode}'. Supported: {sorted(SUPPORTED_EDC_MODES)}")

    if workload_path is not None and not workload_path.exists():
        raise typer.BadParameter(f"workload_path does not exist: {workload_path}")
    if platform_path is not None and not platform_path.exists():
        raise typer.BadParameter(f"platform_path does not exist: {platform_path}")

    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"batsim_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    config = build_batsim_run_config(
        run_id=resolved_run_id,
        trace_dataset=trace,
        policy_id=policy,
        out_dir=out,
        platform_path=platform_path,
        workload_path=workload_path,
        capacity_cpus=capacity_cpus,
        edc_mode=edc_mode,
        edc_library_path=edc_library_path,
        edc_socket_endpoint=edc_socket_endpoint,
        edc_init_json=edc_init_json,
        export_prefix=export_prefix,
        use_wsl_defaults=use_wsl_defaults,
        wsl_distro=wsl_distro,
    )
    resolved_workload = Path(config.payload["inputs"]["workload_path"])
    resolved_platform = Path(config.payload["inputs"]["platform_path"])
    resolved_edc_init = Path(config.payload["inputs"]["edc_init_file"])
    manifest = build_manifest(
        command="hpcopt simulate batsim-config",
        inputs=[trace, *( [workload_path] if workload_path else [] ), *( [platform_path] if platform_path else [] )],
        outputs=[config.config_path, resolved_workload, resolved_platform, resolved_edc_init],
        params={
            "run_id": resolved_run_id,
            "policy_id": policy,
            "capacity_cpus": capacity_cpus,
            "platform_path": str(platform_path) if platform_path else None,
            "workload_path": str(workload_path) if workload_path else None,
            "edc_mode": edc_mode,
            "edc_library_path": edc_library_path,
            "edc_socket_endpoint": edc_socket_endpoint,
            "use_wsl_defaults": use_wsl_defaults,
            "wsl_distro": wsl_distro,
        },
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_batsim_config_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Batsim config: {config.config_path}")
    typer.echo(f"Manifest: {manifest_path}")


@simulate_app.command("batsim-run")
def simulate_batsim_run_cmd(
    config: Path = typer.Option(..., exists=True, readable=True, help="Batsim run config json"),
    batsim_bin: str = typer.Option("batsim", help="Batsim binary path/name"),
    dry_run: bool = typer.Option(True, help="Do not execute batsim, only emit command"),
    use_wsl: bool = typer.Option(
        False,
        help="Run batsim via WSL (for Linux-only batsim installs on Windows hosts)",
    ),
    wsl_distro: str = typer.Option("Ubuntu", help="WSL distro name when --use-wsl is enabled"),
    wsl_load_nix_profile: bool = typer.Option(
        True,
        help="Source ~/.nix-profile/etc/profile.d/nix.sh in WSL before launching batsim",
    ),
    normalize_to_sim_report: bool = typer.Option(
        True,
        help="Normalize successful Batsim output into standard sim-report artifacts",
    ),
    simulation_out: Path = typer.Option(
        Path("outputs/simulations"),
        help="Simulation artifact output directory for normalized Batsim outputs",
    ),
    emit_fidelity_report: bool = typer.Option(
        True,
        help="When normalized output exists, emit candidate-vs-observed fidelity report if trace parquet is available",
    ),
    fidelity_config: Optional[Path] = typer.Option(
        Path("configs/simulation/fidelity_gate.yaml"),
        help="Optional fidelity gate config for candidate fidelity report",
    ),
    out: Path = typer.Option(Path("outputs/reports"), help="Run report directory"),
) -> None:
    ensure_dir(out)
    ensure_dir(simulation_out)
    result = invoke_batsim_run(
        config_path=config,
        batsim_bin=batsim_bin,
        dry_run=dry_run,
        use_wsl=use_wsl,
        wsl_distro=wsl_distro,
        wsl_load_nix_profile=wsl_load_nix_profile,
    )

    report_payload: dict[str, object] = {
        "config_path": str(config),
        "status": result.status,
        "reason": result.reason,
        "returncode": result.returncode,
        "command": result.command,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    normalized = None
    if (
        normalize_to_sim_report
        and not dry_run
        and result.status == "ok"
    ):
        try:
            normalized = normalize_batsim_run_outputs(
                config_path=config,
                report_out_dir=out,
                simulation_out_dir=simulation_out,
            )
            report_payload["normalized_sim_report"] = {
                "run_id": normalized.run_id,
                "policy_id": normalized.policy_id,
                "jobs_csv_path": str(normalized.jobs_csv_path),
                "jobs_artifact_path": str(normalized.jobs_artifact_path),
                "queue_artifact_path": str(normalized.queue_artifact_path),
                "sim_report_path": str(normalized.sim_report_path),
                "invariant_report_path": str(normalized.invariant_report_path),
            }
        except Exception as exc:
            report_payload["normalized_sim_report"] = {
                "status": "failed",
                "reason": str(exc),
            }

    if emit_fidelity_report and normalized is not None:
        try:
            cfg_payload = json.loads(config.read_text(encoding="utf-8"))
            trace_path_raw = cfg_payload.get("inputs", {}).get("trace_dataset")
            trace_path = Path(trace_path_raw) if trace_path_raw else None
            if trace_path is not None and (not trace_path.exists()) and str(trace_path).startswith("/mnt/"):
                parts = str(trace_path).split("/")
                if len(parts) >= 4 and len(parts[2]) == 1:
                    win_tail = "\\".join(parts[3:])
                    trace_path = Path(f"{parts[2].upper()}:\\{win_tail}")
            if trace_path is not None and trace_path.exists() and trace_path.suffix.lower() == ".parquet":
                trace_df = pd.read_parquet(trace_path)
                sim_jobs = pd.read_parquet(normalized.jobs_artifact_path)
                sim_queue = pd.read_parquet(normalized.queue_artifact_path)
                capacity_raw = cfg_payload.get("resources", {}).get("capacity_cpus")
                capacity_cpus = int(capacity_raw) if capacity_raw is not None else 1
                fidelity_out = out / f"{normalized.run_id}_{normalized.policy_id.lower()}_fidelity_candidate_report.json"
                fidelity_cfg = (
                    fidelity_config
                    if (fidelity_config is not None and fidelity_config.exists())
                    else None
                )
                fidelity_result = run_candidate_fidelity_report(
                    trace_df=trace_df,
                    simulated_jobs=sim_jobs,
                    simulated_queue=sim_queue,
                    capacity_cpus=capacity_cpus,
                    out_path=fidelity_out,
                    run_id=normalized.run_id,
                    policy_id=normalized.policy_id,
                    config_path=fidelity_cfg,
                )
                report_payload["candidate_fidelity_report"] = {
                    "status": fidelity_result.status,
                    "report_path": str(fidelity_result.report_path),
                }
            else:
                report_payload["candidate_fidelity_report"] = {
                    "status": "skipped",
                    "reason": "trace_dataset_not_parquet_or_missing",
                }
        except Exception as exc:
            report_payload["candidate_fidelity_report"] = {
                "status": "failed",
                "reason": str(exc),
            }

    report_path = out / f"{config.stem}_batsim_run_report.json"
    write_json(report_path, report_payload)
    typer.echo(f"Batsim run status: {result.status}")
    typer.echo(f"Run report: {report_path}")
    if normalized is not None:
        typer.echo(f"Normalized sim report: {normalized.sim_report_path}")


@simulate_app.command("replay-baselines")
def simulate_replay_baselines_cmd(
    trace: Path = typer.Option(..., exists=True, readable=True, help="Canonical parquet dataset"),
    capacity_cpus: int = typer.Option(64, min=1, help="Cluster CPU capacity"),
    out: Path = typer.Option(Path("outputs/simulations"), help="Simulation artifact output directory"),
    report_out: Path = typer.Option(Path("outputs/reports"), help="Report output directory"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier override"),
    strict_invariants: bool = typer.Option(True, help="Fail on first invariant violation"),
    reference_suite_config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
        help="Reference suite config for trace hash checks",
    ),
) -> None:
    ensure_dir(out)
    ensure_dir(report_out)
    resolved_run_id = run_id or f"baseline_replay_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    trace_df = pd.read_parquet(trace)
    trace_metadata_path = trace.with_suffix(".metadata.json")
    source_reference_match = None
    if trace_metadata_path.exists():
        trace_meta = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        source_reference_match = assert_reference_by_filename_and_hash(
            filename=str(trace_meta.get("source_trace_filename", "")),
            sha256_observed=trace_meta.get("source_trace_sha256"),
            config_path=reference_suite_config,
        )

    policies = ("FIFO_STRICT", "EASY_BACKFILL_BASELINE")
    combined: dict[str, dict[str, object]] = {}
    outputs: list[Path] = []

    for policy in policies:
        sim = run_simulation_from_trace(
            trace_df=trace_df,
            policy_id=policy,
            capacity_cpus=capacity_cpus,
            run_id=f"{resolved_run_id}_{policy.lower()}",
            strict_invariants=strict_invariants,
        )
        jobs_path = out / f"{resolved_run_id}_{policy.lower()}_jobs.parquet"
        queue_path = out / f"{resolved_run_id}_{policy.lower()}_queue.parquet"
        inv_path = report_out / f"{resolved_run_id}_{policy.lower()}_invariants.json"
        sim.jobs_df.to_parquet(jobs_path, index=False)
        sim.queue_series_df.to_parquet(queue_path, index=False)
        write_json(inv_path, sim.invariant_report)
        outputs.extend([jobs_path, queue_path, inv_path])
        combined[policy] = {
            "metrics": sim.metrics,
            "objective_metrics": sim.objective_metrics,
            "fallback_accounting": sim.fallback_accounting,
            "jobs_artifact": str(jobs_path),
            "queue_artifact": str(queue_path),
            "invariant_report": str(inv_path),
        }

    summary_path = report_out / f"{resolved_run_id}_baseline_replay_report.json"
    write_json(
        summary_path,
        {
            "run_id": resolved_run_id,
            "trace": str(trace),
            "capacity_cpus": capacity_cpus,
            "source_trace_reference_suite": source_reference_match,
            "policies": combined,
        },
    )
    outputs.append(summary_path)

    manifest = build_manifest(
        command="hpcopt simulate replay-baselines",
        inputs=[trace, reference_suite_config],
        outputs=outputs,
        params={
            "run_id": resolved_run_id,
            "capacity_cpus": capacity_cpus,
            "strict_invariants": strict_invariants,
            "source_trace_reference_suite": source_reference_match,
        },
        config_paths=[reference_suite_config],
        seeds=[],
    )
    manifest_path = report_out / f"{resolved_run_id}_baseline_replay_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Baseline replay report: {summary_path}")
    typer.echo(f"Manifest: {manifest_path}")


@recommend_app.command("generate")
def recommend_generate_cmd(
    baseline_report: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Baseline simulation report json (e.g., EASY backfill)",
    ),
    candidate_report: list[Path] = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Candidate simulation report json(s)",
    ),
    fidelity_report: Optional[Path] = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Optional fidelity report json for guardrail gating",
    ),
    out: Path = typer.Option(Path("outputs/reports"), help="Recommendation output directory"),
    run_id: Optional[str] = typer.Option(None, help="Run identifier override"),
    w1: float = typer.Option(1.0, help="Weighted score w1 (delta p95 BSLD)"),
    w2: float = typer.Option(0.3, help="Weighted score w2 (delta utilization)"),
    w3: float = typer.Option(2.0, help="Weighted score w3 (fairness penalty)"),
) -> None:
    ensure_dir(out)
    resolved_run_id = run_id or f"recommend_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    recommendation_path = out / f"{resolved_run_id}_recommendation_report.json"
    result = generate_recommendation_report(
        baseline_report_path=baseline_report,
        candidate_report_paths=candidate_report,
        out_path=recommendation_path,
        fidelity_report_path=fidelity_report,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    outputs = [result.report_path]
    inputs = [baseline_report, *candidate_report]
    if fidelity_report:
        inputs.append(fidelity_report)
    manifest = build_manifest(
        command="hpcopt recommend generate",
        inputs=inputs,
        outputs=outputs,
        params={
            "run_id": resolved_run_id,
            "weights": {"w1": w1, "w2": w2, "w3": w3},
        },
        seeds=[],
    )
    manifest_path = out / f"{resolved_run_id}_recommend_manifest.json"
    write_manifest(manifest_path, manifest)

    typer.echo(f"Recommendation status: {result.payload['status']}")
    typer.echo(f"Recommendation report: {result.report_path}")
    typer.echo(f"Manifest: {manifest_path}")


@report_app.command("export")
def report_export_cmd(
    run_id: str = typer.Option(..., help="Run identifier"),
    out: Path = typer.Option(Path("outputs/reports"), help="Output directory"),
    format: str = typer.Option("both", help="json|md|both"),
) -> None:
    ensure_dir(out)
    result = export_run_report(run_id=run_id, out_dir=out)
    if format not in {"json", "md", "both"}:
        raise typer.BadParameter("format must be one of: json|md|both")
    if format in {"json", "both"}:
        typer.echo(f"Export json: {result.json_path}")
    if format in {"md", "both"}:
        typer.echo(f"Export md: {result.md_path}")


@data_app.command("lock-reference-suite")
def lock_reference_suite_cmd(
    config: Path = typer.Option(
        Path("configs/data/reference_suite.yaml"),
        exists=True,
        readable=True,
        help="Reference suite config yaml",
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory containing reference suite traces"),
    out: Path = typer.Option(
        Path("outputs/reports/reference_suite_lock_report.json"),
        help="Lock report output path",
    ),
    strict_missing: bool = typer.Option(
        False,
        help="Fail if any configured reference-suite file is missing from raw dir",
    ),
) -> None:
    report = lock_reference_suite_hashes(
        config_path=config,
        raw_dir=raw_dir,
        out_report_path=out,
        strict_missing=strict_missing,
    )
    typer.echo(f"Suite lock updated: {report['updated']}")
    typer.echo(f"Missing files: {len(report['missing_files'])}")
    typer.echo(f"Report: {out}")


@serve_app.command("api")
def serve_api_cmd(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, min=1, max=65535, help="Bind port"),
) -> None:
    import uvicorn

    uvicorn.run("hpcopt.api.app:app", host=host, port=port, reload=False)


def run() -> None:
    app()
