"""CLI commands for model management, data contracts, and serving."""

from __future__ import annotations

from pathlib import Path

import typer

from hpcopt.data.reference_suite import lock_reference_suite_hashes
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir
from hpcopt.utils.io import ensure_dir, write_json

model_app = typer.Typer(help="Model management commands")
data_app = typer.Typer(help="Dataset contract commands")
serve_app = typer.Typer(help="Service commands")


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


# ──────────────────────── Data ────────────────────────


@data_app.command("lock-reference-suite")
def lock_reference_suite_cmd(
    config: Path = typer.Option(Path("configs/data/reference_suite.yaml"), exists=True, readable=True),
    raw_dir: Path = typer.Option(Path("data/raw")),
    out: Path = typer.Option(Path("outputs/reports/reference_suite_lock_report.json")),
    strict_missing: bool = typer.Option(False),
) -> None:
    report = lock_reference_suite_hashes(
        config_path=config,
        raw_dir=raw_dir,
        out_report_path=out,
        strict_missing=strict_missing,
    )
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
