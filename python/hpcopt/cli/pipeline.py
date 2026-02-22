"""CLI commands for profiling, feature building, analysis, and credibility."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import typer

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.features.pipeline import build_feature_dataset
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir
from hpcopt.profile.trace_profile import build_trace_profile
from hpcopt.utils.io import ensure_dir

profile_app = typer.Typer(help="Trace profiling commands")
features_app = typer.Typer(help="Feature pipeline commands")
analysis_app = typer.Typer(help="Analysis commands")
credibility_app = typer.Typer(help="Credibility protocol commands")


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
        params={
            "dataset_id": ds_id,
            "n_folds": n_folds,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
        },
        seeds=[],
    )
    manifest_path = report_out / f"{ds_id}_features_manifest.json"
    write_manifest(manifest_path, manifest)
    typer.echo(f"Feature dataset: {result.feature_dataset_path}")
    typer.echo(f"Features manifest: {manifest_path}")


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
    from hpcopt.analysis.sensitivity import build_sensitivity_report, run_guard_k_sweep
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
