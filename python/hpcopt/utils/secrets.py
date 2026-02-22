"""File-based secret loading for API keys.

Supports three loading strategies (checked in order):
1. ``HPCOPT_API_KEYS_FILE`` env var  → reads file, one key per line
2. ``/run/secrets/hpcopt_api_keys``  → Docker/K8s secret mount
3. ``HPCOPT_API_KEYS`` env var       → comma-separated (legacy, warns)

The file is re-read on every call so key rotation doesn't require a restart.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DOCKER_SECRET_PATH = Path("/run/secrets/hpcopt_api_keys")


def _read_keys_file(path: Path) -> set[str]:
    """Read one API key per line, stripping blanks and comments."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Cannot read API keys file: %s", path)
        return set()
    return {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}


def load_api_keys() -> set[str]:
    """Return the current set of valid API keys (re-reads on every call)."""
    # Strategy 1: explicit file path via env var
    keys_file = os.getenv("HPCOPT_API_KEYS_FILE")
    if keys_file:
        return _read_keys_file(Path(keys_file))

    # Strategy 2: Docker / Kubernetes secret mount
    if _DOCKER_SECRET_PATH.exists():
        return _read_keys_file(_DOCKER_SECRET_PATH)

    # Strategy 3: legacy comma-separated env var
    raw = os.getenv("HPCOPT_API_KEYS", "")
    if raw:
        logger.debug("Loading API keys from HPCOPT_API_KEYS env var (consider migrating to file-based secrets)")
        return {k.strip() for k in raw.split(",") if k.strip()}

    return set()
