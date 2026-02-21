from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def sha256_path(path: Path) -> str | None:
    """Compute SHA-256 hex digest of a file, or return None if not a regular file."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
