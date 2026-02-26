"""CLI commands for model training and hyperparameter tuning."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer
import yaml

from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.models.runtime_quantile import train_runtime_quantile_models
from hpcopt.utils.io import ensure_dir

train_app = typer.Typer(help="Training commands")


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
        dataset_path=dataset,
        out_dir=out,
        model_id=resolved_model_id,
        seed=seed,
        hyperparams=hp,
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
        dataset_path=dataset,
        out_path=report_path,
        quantile=quantile,
        seed=seed,
        n_trials=n_trials,
        n_folds=n_folds,
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
