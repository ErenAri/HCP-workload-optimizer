from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from hpcopt.utils.io import write_json


def _sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cmd_version(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or None
    except Exception:
        return None


def _lock_hash(paths: list[Path]) -> str | None:
    existing = [path for path in paths if path.exists() and path.is_file()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in sorted(existing):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update((_sha256_path(path) or "").encode("utf-8"))
    return digest.hexdigest()


def build_manifest(
    command: str,
    inputs: list[Path],
    outputs: list[Path],
    params: dict[str, Any],
    policy_spec_path: Path | None = Path("claudedocs/policy_spec_baselines_mvp.md"),
    config_paths: list[Path] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    config_paths = config_paths or []
    config_snapshots = [
        {
            "path": str(path),
            "sha256": _sha256_path(path),
            "content": path.read_text(encoding="utf-8", errors="replace")
            if path.exists() and path.is_file() and path.stat().st_size <= 256_000
            else None,
        }
        for path in config_paths
    ]
    lock_hash = _lock_hash(
        [Path("pyproject.toml"), Path("rust/Cargo.toml"), *config_paths]
    )
    return {
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "command": command,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "package_versions": {
            "hpc-workload-optimizer": _pkg_version("hpc-workload-optimizer"),
            "pandas": _pkg_version("pandas"),
            "pyarrow": _pkg_version("pyarrow"),
            "typer": _pkg_version("typer"),
        },
        "tool_versions": {
            "cargo": _cmd_version(["cargo", "--version"]),
            "rustc": _cmd_version(["rustc", "--version"]),
        },
        "environment": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256_path(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256_path(path)}
            for path in outputs
        ],
        "policy_hash_sha256": _sha256_path(policy_spec_path) if policy_spec_path else None,
        "dependency_lock_hash_sha256": lock_hash,
        "config_snapshots": config_snapshots,
        "seeds": seeds or [],
        "params": params,
        "immutable": True,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    # Contract note: once written, manifest is treated immutable by workflow.
    manifest = dict(payload)
    manifest["manifest_hash_sha256"] = None
    write_json(path, manifest)
    manifest_hash = _sha256_path(path)
    manifest["manifest_hash_sha256"] = manifest_hash
    write_json(path, manifest)
    return path
