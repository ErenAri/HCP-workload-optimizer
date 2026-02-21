from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hpcopt.api.app import app


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _collect_methods(path_item: dict[str, Any]) -> set[str]:
    return {k.lower() for k in path_item.keys() if k.lower() in HTTP_METHODS}


def _compare_openapi(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    base_paths = baseline.get("paths", {})
    cur_paths = current.get("paths", {})
    if not isinstance(base_paths, dict) or not isinstance(cur_paths, dict):
        return ["OpenAPI paths object missing in baseline or current spec."]

    for path, base_item in base_paths.items():
        if path not in cur_paths:
            errors.append(f"Removed API path: {path}")
            continue
        cur_item = cur_paths[path]
        if not isinstance(base_item, dict) or not isinstance(cur_item, dict):
            continue
        base_methods = _collect_methods(base_item)
        cur_methods = _collect_methods(cur_item)
        missing_methods = sorted(base_methods - cur_methods)
        for method in missing_methods:
            errors.append(f"Removed API operation: {method.upper()} {path}")

        for method in sorted(base_methods & cur_methods):
            base_op = base_item.get(method, {})
            cur_op = cur_item.get(method, {})
            if not isinstance(base_op, dict) or not isinstance(cur_op, dict):
                continue
            base_responses = base_op.get("responses", {})
            cur_responses = cur_op.get("responses", {})
            if not isinstance(base_responses, dict) or not isinstance(cur_responses, dict):
                continue
            for status in base_responses:
                if status not in cur_responses:
                    errors.append(f"Removed response status {status} for {method.upper()} {path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAPI compatibility against baseline.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("schemas/openapi_baseline.json"),
        help="Baseline OpenAPI JSON path.",
    )
    args = parser.parse_args()

    baseline_path = args.baseline
    if not baseline_path.exists():
        print(f"ERROR: baseline not found: {baseline_path}")
        return 1

    try:
        baseline = _load_json(baseline_path)
    except Exception as exc:  # pragma: no cover - CLI defensive path
        print(f"ERROR: failed to load baseline OpenAPI: {exc}")
        return 1

    current = app.openapi()
    errors = _compare_openapi(baseline=baseline, current=current)
    if errors:
        print("OpenAPI compatibility check: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("OpenAPI compatibility check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
