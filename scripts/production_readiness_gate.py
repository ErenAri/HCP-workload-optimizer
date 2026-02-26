from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_STATUSES = {"done", "todo", "waived"}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checklist must be a mapping: {path}")
    return payload


def _parse_utc_timestamp(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _validate_shape(payload: dict[str, Any], checklist_path: Path) -> list[str]:
    errors: list[str] = []
    metadata = payload.get("metadata")
    checks = payload.get("checks")

    if not isinstance(metadata, dict):
        return [f"{checklist_path}: missing or invalid 'metadata' mapping"]
    if not isinstance(checks, list):
        return [f"{checklist_path}: missing or invalid 'checks' list"]

    for field in ("release_target", "reviewed_at_utc", "owner"):
        if not isinstance(metadata.get(field), str) or not str(metadata.get(field)).strip():
            errors.append(f"{checklist_path}: metadata.{field} must be a non-empty string")

    seen_ids: set[str] = set()
    for idx, item in enumerate(checks):
        prefix = f"{checklist_path}: checks[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        check_id = item.get("id")
        title = item.get("title")
        status = item.get("status")
        required = item.get("required")
        evidence = item.get("evidence")

        if not isinstance(check_id, str) or not check_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif check_id in seen_ids:
            errors.append(f"{prefix}.id duplicates '{check_id}'")
        else:
            seen_ids.add(check_id)

        if not isinstance(title, str) or not title.strip():
            errors.append(f"{prefix}.title must be a non-empty string")
        if not isinstance(required, bool):
            errors.append(f"{prefix}.required must be boolean")
        if not isinstance(status, str) or status not in VALID_STATUSES:
            errors.append(f"{prefix}.status must be one of {sorted(VALID_STATUSES)}")
        if not isinstance(evidence, str):
            errors.append(f"{prefix}.evidence must be a string")
        if status == "waived":
            note = item.get("note")
            if not isinstance(note, str) or not note.strip():
                errors.append(f"{prefix} waived checks must include non-empty note")

    return errors


def _validate_release_gate(payload: dict[str, Any], checklist_path: Path) -> list[str]:
    errors: list[str] = []
    metadata = payload["metadata"]
    checks = payload["checks"]

    reviewed_raw = str(metadata["reviewed_at_utc"])
    try:
        reviewed_at = _parse_utc_timestamp(reviewed_raw)
    except ValueError as exc:
        return [f"{checklist_path}: metadata.reviewed_at_utc is not valid ISO-8601 UTC timestamp: {exc}"]

    now = dt.datetime.now(tz=dt.UTC)
    age_days = (now - reviewed_at).days
    if age_days > 30:
        errors.append(f"{checklist_path}: checklist is stale ({age_days} days old); update metadata.reviewed_at_utc")

    for item in checks:
        if not item.get("required", False):
            continue
        check_id = str(item.get("id", "<missing-id>"))
        status = str(item.get("status", ""))
        evidence = str(item.get("evidence", "")).strip()
        if status != "done":
            errors.append(f"{checklist_path}: required check '{check_id}' must be 'done' for release (got '{status}')")
        if not evidence:
            errors.append(f"{checklist_path}: required check '{check_id}' must include non-empty evidence")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production readiness checklist.")
    parser.add_argument(
        "--checklist",
        type=Path,
        default=Path("configs/release/production_readiness.yaml"),
        help="Path to production readiness checklist YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "release"),
        default="validate",
        help="validate: schema/shape checks only; release: strict release gating checks.",
    )
    args = parser.parse_args()

    checklist_path = args.checklist
    if not checklist_path.exists():
        print(f"ERROR: checklist file not found: {checklist_path}")
        return 1

    try:
        payload = _load_yaml(checklist_path)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        print(f"ERROR: failed to parse checklist YAML: {exc}")
        return 1

    errors = _validate_shape(payload, checklist_path)
    if not errors and args.mode == "release":
        errors.extend(_validate_release_gate(payload, checklist_path))

    if errors:
        print("Production readiness gate: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Production readiness gate: PASS (mode={args.mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
