"""Aggregate credibility dossier: assembles per-trace results into a comprehensive report."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DossierResult:
    json_path: Path
    md_path: Path
    payload: dict[str, Any]


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _collect_trace_reports(trace_dir: Path) -> dict[str, Any]:
    """Collect all reports for a single trace from its output directory."""
    reports_dir = trace_dir / "reports"
    if not reports_dir.exists():
        return {"status": "missing", "reports_dir": str(reports_dir)}

    collected: dict[str, Any] = {"trace_dir": str(trace_dir)}
    for json_file in sorted(reports_dir.glob("*.json")):
        payload = _safe_load_json(json_file)
        if payload is None:
            continue

        name = json_file.stem
        if "fidelity_report" in name:
            collected["fidelity"] = payload
        elif "recommendation_report" in name:
            collected["recommendation"] = payload
        elif "sim_report" in name:
            policy = payload.get("policy_id", "unknown")
            collected.setdefault("simulations", {})[policy] = payload
        elif "benchmark_report" in name:
            collected["benchmark"] = payload
        elif "stress_report" in name:
            collected["stress"] = payload
        elif "trace_profile" in name:
            collected["profile"] = payload
        elif "credibility_manifest" in name:
            collected["manifest"] = payload

    return collected


def _build_fidelity_summary(per_trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize fidelity status across traces."""
    summary: list[dict[str, str]] = []
    for trace_id, data in per_trace.items():
        fidelity = data.get("fidelity", {})
        status = str(fidelity.get("status", "not_run"))
        summary.append({"trace_id": trace_id, "status": status})
    return {
        "traces": summary,
        "all_pass": all(s["status"] == "pass" for s in summary),
    }


def _build_recommendation_summary(per_trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize recommendation acceptance/rejection across traces."""
    accepted = 0
    blocked = 0
    failure_modes: list[dict[str, Any]] = []

    for trace_id, data in per_trace.items():
        rec = data.get("recommendation", {})
        status = str(rec.get("status", "not_run"))
        if status == "accepted":
            accepted += 1
        else:
            blocked += 1
            for fm in rec.get("failure_modes", []):
                failure_modes.append({"trace_id": trace_id, **fm})

    return {
        "accepted_count": accepted,
        "blocked_count": blocked,
        "failure_modes": failure_modes,
    }


def _build_fallback_summary(per_trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize fallback telemetry across ML policy runs."""
    fallbacks: list[dict[str, Any]] = []
    for trace_id, data in per_trace.items():
        sims = data.get("simulations", {})
        ml_sim = sims.get("ML_BACKFILL_P50", {})
        fa = ml_sim.get("fallback_accounting")
        if fa:
            fallbacks.append({"trace_id": trace_id, **fa})
    return {"per_trace": fallbacks}


def _build_no_improvement_summary(per_trace: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect no-improvement narratives."""
    narratives: list[dict[str, Any]] = []
    for trace_id, data in per_trace.items():
        rec = data.get("recommendation", {})
        narrative = rec.get("no_improvement_narrative")
        if narrative:
            narratives.append({"trace_id": trace_id, **narrative})
    return narratives


def assemble_credibility_dossier(
    input_dir: Path,
    output_path: Path,
) -> DossierResult:
    """Assemble the aggregate credibility dossier from per-trace credibility runs."""
    ensure_dir(output_path)

    # Discover trace directories
    per_trace: dict[str, dict[str, Any]] = {}
    if input_dir.exists():
        for entry in sorted(input_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                reports_dir = entry / "reports"
                if reports_dir.exists():
                    per_trace[entry.name] = _collect_trace_reports(entry)

    # Also check for suite summary
    suite_summary = _safe_load_json(input_dir / "credibility_suite_summary.json")

    # Build summaries
    fidelity_summary = _build_fidelity_summary(per_trace)
    recommendation_summary = _build_recommendation_summary(per_trace)
    fallback_summary = _build_fallback_summary(per_trace)
    no_improvement_narratives = _build_no_improvement_summary(per_trace)

    # Sensitivity sweep summary (if present)
    sensitivity_summaries: list[dict[str, Any]] = []
    for trace_id, data in per_trace.items():
        reports_dir = Path(data.get("trace_dir", "")) / "reports"
        for f in sorted(reports_dir.glob("*sensitivity*")) if reports_dir.exists() else []:
            payload = _safe_load_json(f)
            if payload:
                sensitivity_summaries.append({"trace_id": trace_id, **payload.get("analysis", {})})

    payload = {
        "suite_summary": suite_summary,
        "traces_collected": list(per_trace.keys()),
        "fidelity_summary": fidelity_summary,
        "recommendation_summary": recommendation_summary,
        "fallback_telemetry_summary": fallback_summary,
        "no_improvement_narratives": no_improvement_narratives,
        "sensitivity_sweep_summary": sensitivity_summaries,
        "per_trace_data": per_trace,
    }

    json_path = output_path / "credibility_dossier.json"
    write_json(json_path, payload)

    # Build markdown version
    md_lines = [
        "# Credibility Dossier",
        "",
        f"## Suite Status: {suite_summary.get('status', 'unknown') if suite_summary else 'no_suite_summary'}",
        "",
        f"Traces collected: {', '.join(per_trace.keys()) or 'none'}",
        "",
        "## Fidelity Summary",
        "",
        f"All pass: {fidelity_summary['all_pass']}",
        "",
    ]
    for trace_status in fidelity_summary["traces"]:
        md_lines.append(f"- **{trace_status['trace_id']}**: {trace_status['status']}")

    md_lines.extend(
        [
            "",
            "## Recommendation Summary",
            "",
            f"- Accepted: {recommendation_summary['accepted_count']}",
            f"- Blocked: {recommendation_summary['blocked_count']}",
            "",
        ]
    )
    if recommendation_summary["failure_modes"]:
        md_lines.append("### Failure Modes")
        for fm in recommendation_summary["failure_modes"]:
            md_lines.append(f"- **{fm.get('trace_id')}** / {fm.get('policy_id')}: {fm.get('rejection_reasons', [])}")
        md_lines.append("")

    md_lines.extend(
        [
            "## Fallback Telemetry",
            "",
        ]
    )
    for fb in fallback_summary["per_trace"]:
        tid = fb.get("trace_id", "unknown")
        pred_rate = fb.get("prediction_used_rate", 0)
        md_lines.append(f"- **{tid}**: prediction_used_rate={pred_rate:.2%}")

    if no_improvement_narratives:
        md_lines.extend(
            [
                "",
                "## No-Improvement Narratives",
                "",
            ]
        )
        for narrative in no_improvement_narratives:
            md_lines.append(f"### {narrative.get('trace_id', 'unknown')}")
            md_lines.append(f"- Summary: {narrative.get('summary', '')}")
            causes = narrative.get("likely_causes", [])
            if causes:
                md_lines.append(f"- Causes: {', '.join(str(c) for c in causes)}")
            regime = narrative.get("workload_regime")
            if regime:
                md_lines.append(f"- Workload regime: {regime}")
            md_lines.append("")

    if sensitivity_summaries:
        md_lines.extend(["## Sensitivity Sweep Summary", ""])
        for ss in sensitivity_summaries:
            md_lines.append(f"- **{ss.get('trace_id')}**: optimal_k={ss.get('optimal_k')}")
        md_lines.append("")

    md_path = output_path / "credibility_dossier.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return DossierResult(json_path=json_path, md_path=md_path, payload=payload)
