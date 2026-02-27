from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _extract_quoted_tokens(payload: str) -> list[tuple[str, str]]:
    return re.findall(r'"([^"]+)"|\'([^\']+)\'', payload)


def _normalize_tokens(raw_tokens: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for a, b in raw_tokens:
        token = a or b
        if token:
            out.append(token)
    return out


def _extract_supported_policies(core_text: str) -> list[str]:
    match = re.search(r"SUPPORTED_POLICIES\s*=\s*\{([^}]+)\}", core_text, flags=re.DOTALL)
    if not match:
        return []
    return _normalize_tokens(_extract_quoted_tokens(match.group(1)))


def _extract_backends(runtime_text: str) -> list[str]:
    match = re.search(r"BACKENDS\s*=\s*\[([^\]]+)\]", runtime_text, flags=re.DOTALL)
    if not match:
        return []
    return _normalize_tokens(_extract_quoted_tokens(match.group(1)))


def _extract_feature_columns(runtime_text: str) -> list[str]:
    match = re.search(r"FEATURE_COLUMNS\s*=\s*\[(.*?)\]", runtime_text, flags=re.DOTALL)
    if not match:
        return []
    return _normalize_tokens(_extract_quoted_tokens(match.group(1)))


def validate_docs_consistency(repo_root: Path) -> list[str]:
    errors: list[str] = []

    readme = (repo_root / "README.md").read_text(encoding="utf-8", errors="replace")
    docs_ml = (repo_root / "docs/05-ml-runtime-modeling.md").read_text(encoding="utf-8", errors="replace")
    docs_iface = (repo_root / "docs/07-interfaces-cli-and-api.md").read_text(encoding="utf-8", errors="replace")
    core_text = (repo_root / "python/hpcopt/simulate/core.py").read_text(encoding="utf-8", errors="replace")
    runtime_text = (repo_root / "python/hpcopt/models/runtime_quantile.py").read_text(encoding="utf-8", errors="replace")

    policies = _extract_supported_policies(core_text)
    if not policies:
        errors.append("Could not extract SUPPORTED_POLICIES from python/hpcopt/simulate/core.py")
    for policy in policies:
        if policy not in readme:
            errors.append(f"README.md is missing policy reference: {policy}")
        if policy not in docs_iface:
            errors.append(f"docs/07-interfaces-cli-and-api.md is missing policy reference: {policy}")

    backends = _extract_backends(runtime_text)
    if not backends:
        errors.append("Could not extract BACKENDS from python/hpcopt/models/runtime_quantile.py")
    for backend in backends:
        if backend not in readme:
            errors.append(f"README.md is missing runtime backend reference: {backend}")
        if backend not in docs_ml:
            errors.append(f"docs/05-ml-runtime-modeling.md is missing runtime backend reference: {backend}")
        if backend not in docs_iface:
            errors.append(f"docs/07-interfaces-cli-and-api.md is missing runtime backend reference: {backend}")

    feature_columns = _extract_feature_columns(runtime_text)
    lookback_features = [
        feature
        for feature in feature_columns
        if feature.endswith("_lookback") or feature == "queue_congestion_at_submit_jobs"
    ]
    if not lookback_features:
        errors.append("Could not extract lookback features from FEATURE_COLUMNS in runtime_quantile.py")
    for feature in lookback_features:
        if feature not in docs_ml:
            errors.append(f"docs/05-ml-runtime-modeling.md is missing lookback feature reference: {feature}")

    required_interface_phrases = [
        "--backend",
        "ML_BACKFILL_P10",
        "predictor ensemble",
    ]
    for phrase in required_interface_phrases:
        if phrase not in docs_iface:
            errors.append(f"docs/07-interfaces-cli-and-api.md is missing required phrase: '{phrase}'")

    required_readme_phrases = [
        "--backend",
        "ML_BACKFILL_P10",
    ]
    for phrase in required_readme_phrases:
        if phrase not in readme:
            errors.append(f"README.md is missing required phrase: '{phrase}'")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate docs consistency with core runtime/simulation interfaces.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root path.",
    )
    args = parser.parse_args()

    try:
        errors = validate_docs_consistency(args.repo_root)
    except OSError as exc:
        print(f"Docs consistency: FAIL\n- Could not read required file: {exc}")
        return 1

    if errors:
        print("Docs consistency: FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Docs consistency: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
