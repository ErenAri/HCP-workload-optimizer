from __future__ import annotations

import datetime as dt
import hashlib
import logging
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from hpcopt.utils.io import sha256_path as _sha256_path
from hpcopt.utils.io import write_json

logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.debug("Could not resolve git commit: %s", exc)
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
            timeout=10,
        )
        return result.stdout.strip() or result.stderr.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.debug("Could not resolve version for %s: %s", cmd[0], exc)
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


def _pip_freeze_snapshot() -> list[str]:
    """Capture installed package versions via pip freeze."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.debug("Could not capture pip freeze: %s", exc)
        return []


def _os_fingerprint() -> dict[str, str]:
    """Collect OS/machine fingerprint for reproducibility."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def build_manifest(
    command: str,
    inputs: list[Path],
    outputs: list[Path],
    params: dict[str, Any],
    policy_spec_path: Path | None = Path("design_docs/policy_spec_baselines_mvp.md"),
    config_paths: list[Path] | None = None,
    seeds: list[int] | None = None,
    model_hash: str | None = None,
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
    lock_hash = _lock_hash([Path("pyproject.toml"), Path("rust/Cargo.toml"), *config_paths])
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
            "scikit-learn": _pkg_version("scikit-learn"),
            "fastapi": _pkg_version("fastapi"),
        },
        "tool_versions": {
            "cargo": _cmd_version(["cargo", "--version"]),
            "rustc": _cmd_version(["rustc", "--version"]),
        },
        "environment": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
        },
        "os_fingerprint": _os_fingerprint(),
        "pip_freeze_snapshot": _pip_freeze_snapshot(),
        "inputs": [{"path": str(path), "sha256": _sha256_path(path)} for path in inputs],
        "outputs": [{"path": str(path), "sha256": _sha256_path(path)} for path in outputs],
        "policy_hash_sha256": _sha256_path(policy_spec_path) if policy_spec_path else None,
        "dependency_lock_hash_sha256": lock_hash,
        "config_snapshots": config_snapshots,
        "seeds": seeds or [],
        "params": params,
        "model_hash": model_hash,
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
