from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hpcopt.utils.io import write_json


@dataclass(frozen=True)
class ReferenceTrace:
    trace_id: str
    filename: str
    source: str
    sha256: str | None


@dataclass(frozen=True)
class ReferenceSuite:
    suite_id: str
    traces: tuple[ReferenceTrace, ...]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_suite(config_path: Path) -> ReferenceSuite:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference suite config must be a mapping")
    traces_raw = payload.get("traces")
    if not isinstance(traces_raw, list) or not traces_raw:
        raise ValueError("reference suite config must include non-empty traces list")

    traces: list[ReferenceTrace] = []
    for item in traces_raw:
        if not isinstance(item, dict):
            raise ValueError("trace entries must be objects")
        for key in ("id", "filename", "source"):
            if key not in item:
                raise ValueError(f"trace entry missing required key '{key}'")
        sha = item.get("sha256")
        if sha is not None:
            sha = str(sha).strip().lower()
            if sha in {"", "null", "none"}:
                sha = None
        traces.append(
            ReferenceTrace(
                trace_id=str(item["id"]),
                filename=str(item["filename"]),
                source=str(item["source"]),
                sha256=sha,
            )
        )
    suite_id = str(payload.get("suite_id", "reference_suite"))
    return ReferenceSuite(suite_id=suite_id, traces=tuple(traces))


def lock_reference_suite_hashes(
    config_path: Path,
    raw_dir: Path,
    out_report_path: Path | None = None,
    strict_missing: bool = True,
) -> dict[str, Any]:
    suite = load_reference_suite(config_path)
    cfg_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_payload, dict):
        raise ValueError("invalid reference suite config")

    updated = False
    locked: list[dict[str, Any]] = []
    missing_files: list[str] = []
    mismatch_files: list[dict[str, str]] = []

    for idx, trace in enumerate(suite.traces):
        file_path = raw_dir / trace.filename
        record = {
            "id": trace.trace_id,
            "filename": trace.filename,
            "exists": file_path.exists(),
            "sha256_previous": trace.sha256,
            "sha256_computed": None,
            "status": "missing",
        }
        if not file_path.exists():
            missing_files.append(trace.filename)
            locked.append(record)
            continue

        computed = _sha256_path(file_path)
        record["sha256_computed"] = computed
        if trace.sha256 and trace.sha256 != computed:
            mismatch_files.append(
                {
                    "filename": trace.filename,
                    "sha256_config": trace.sha256,
                    "sha256_computed": computed,
                }
            )
        if trace.sha256 != computed:
            cfg_payload["traces"][idx]["sha256"] = computed
            updated = True
        record["status"] = "locked"
        locked.append(record)

    if strict_missing and missing_files:
        raise FileNotFoundError(
            "missing reference suite files in raw dir: "
            + ", ".join(sorted(missing_files))
        )
    if mismatch_files:
        # mismatch is informational; the lock action updates config to current file hash.
        updated = True

    if updated:
        config_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding="utf-8")

    report = {
        "suite_id": suite.suite_id,
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "updated": updated,
        "missing_files": missing_files,
        "mismatch_files": mismatch_files,
        "trace_lock_status": locked,
    }
    if out_report_path is not None:
        write_json(out_report_path, report)
    return report


def match_trace_to_reference(
    trace_path: Path,
    config_path: Path,
) -> dict[str, Any] | None:
    suite = load_reference_suite(config_path)
    filename = trace_path.name
    for trace in suite.traces:
        if trace.filename == filename:
            sha = _sha256_path(trace_path) if trace_path.exists() else None
            return {
                "suite_id": suite.suite_id,
                "trace_id": trace.trace_id,
                "filename": trace.filename,
                "sha256_expected": trace.sha256,
                "sha256_observed": sha,
                "sha256_match": (trace.sha256 == sha) if trace.sha256 and sha else False,
            }
    return None


def assert_reference_trace_hash_match(trace_path: Path, config_path: Path) -> dict[str, Any] | None:
    match = match_trace_to_reference(trace_path=trace_path, config_path=config_path)
    if match is None:
        return None
    expected = match["sha256_expected"]
    observed = match["sha256_observed"]
    if not expected:
        raise ValueError(
            f"reference suite trace '{match['trace_id']}' has no locked sha256 in config: {config_path}"
        )
    if observed != expected:
        raise ValueError(
            f"reference suite hash mismatch for '{match['trace_id']}': expected={expected}, observed={observed}"
        )
    return match


def match_reference_by_filename_and_hash(
    filename: str,
    sha256_observed: str | None,
    config_path: Path,
) -> dict[str, Any] | None:
    suite = load_reference_suite(config_path)
    for trace in suite.traces:
        if trace.filename == filename:
            return {
                "suite_id": suite.suite_id,
                "trace_id": trace.trace_id,
                "filename": trace.filename,
                "sha256_expected": trace.sha256,
                "sha256_observed": sha256_observed,
                "sha256_match": (
                    (trace.sha256 == sha256_observed)
                    if trace.sha256 is not None and sha256_observed is not None
                    else False
                ),
            }
    return None


def assert_reference_by_filename_and_hash(
    filename: str,
    sha256_observed: str | None,
    config_path: Path,
) -> dict[str, Any] | None:
    match = match_reference_by_filename_and_hash(
        filename=filename,
        sha256_observed=sha256_observed,
        config_path=config_path,
    )
    if match is None:
        return None
    expected = match["sha256_expected"]
    if not expected:
        raise ValueError(
            f"reference suite trace '{match['trace_id']}' has no locked sha256 in config: {config_path}"
        )
    if sha256_observed != expected:
        raise ValueError(
            f"reference suite hash mismatch for '{match['trace_id']}': expected={expected}, observed={sha256_observed}"
        )
    return match
