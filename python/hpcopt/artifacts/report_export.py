from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpcopt.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load JSON from %s: %s", path, exc)
        return None
    return None


@dataclass(frozen=True)
class ReportExportResult:
    json_path: Path
    md_path: Path
    payload: dict[str, Any]


def _validate_run_id(run_id: str) -> None:
    """Validate run_id to prevent path traversal attacks."""
    if not run_id:
        raise ValueError("run_id must not be empty")
    # Block path separators, parent directory references, and glob-special chars
    forbidden = {"\\", "/", "..", "*", "?", "[", "]"}
    for ch in forbidden:
        if ch in run_id:
            raise ValueError(f"run_id contains forbidden character or sequence {ch!r}: {run_id!r}")


def export_run_report(
    run_id: str,
    out_dir: Path,
    report_dir: Path = Path("outputs/reports"),
    simulation_dir: Path = Path("outputs/simulations"),
    model_dir: Path = Path("outputs/models"),
) -> ReportExportResult:
    _validate_run_id(run_id)
    ensure_dir(out_dir)

    report_files = sorted(report_dir.glob(f"*{run_id}*"))
    simulation_files = sorted(simulation_dir.glob(f"*{run_id}*"))
    model_files = sorted(model_dir.glob(f"*{run_id}*"))

    report_json_objects: list[dict[str, Any]] = []
    for path in report_files:
        if path.suffix.lower() == ".json":
            payload = _safe_load_json(path)
            if payload is not None:
                report_json_objects.append({"path": str(path), "payload": payload})

    summary = {
        "run_id": run_id,
        "exported_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "artifacts": {
            "report_files": [str(path) for path in report_files],
            "simulation_files": [str(path) for path in simulation_files],
            "model_files": [str(path) for path in model_files],
        },
        "report_json_objects": report_json_objects,
    }

    json_path = out_dir / f"{run_id}_export.json"
    write_json(json_path, summary)

    md_lines = [
        f"# Run Export: {run_id}",
        "",
        f"- Exported at: {summary['exported_at_utc']}",
        f"- Report files: {len(report_files)}",
        f"- Simulation files: {len(simulation_files)}",
        f"- Model files: {len(model_files)}",
        "",
        "## Report Files",
    ]
    for path in report_files:
        md_lines.append(f"- `{path}`")

    md_lines.append("")
    md_lines.append("## Simulation Files")
    for path in simulation_files:
        md_lines.append(f"- `{path}`")

    md_lines.append("")
    md_lines.append("## Model Files")
    for path in model_files:
        md_lines.append(f"- `{path}`")

    md_path = out_dir / f"{run_id}_export.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return ReportExportResult(json_path=json_path, md_path=md_path, payload=summary)
