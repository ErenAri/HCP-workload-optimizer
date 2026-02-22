"""CLI commands for reporting, recommendations, and artifact management."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer

from hpcopt.artifacts.benchmark import run_benchmark_suite
from hpcopt.artifacts.manifest import build_manifest, write_manifest
from hpcopt.artifacts.report_export import export_run_report
from hpcopt.recommend.engine import generate_recommendation_report
from hpcopt.simulate.core import SUPPORTED_POLICIES
from hpcopt.utils.io import ensure_dir

report_app = typer.Typer(help="Reporting commands")
recommend_app = typer.Typer(help="Recommendation commands")
artifacts_app = typer.Typer(help="Artifact management commands")


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
