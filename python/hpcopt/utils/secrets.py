"""File-based secret loading for API keys.

Supports three loading strategies (checked in order):
1. ``HPCOPT_API_KEYS_FILE`` env var  → reads file, one key per line
2. ``/run/secrets/hpcopt_api_keys``  → Docker/K8s secret mount
3. ``HPCOPT_API_KEYS`` env var       → comma-separated (legacy, warns)

Keys are cached with a short TTL (default 30 s) to reduce per-request I/O
while still supporting rotation without a restart.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DOCKER_SECRET_PATH = Path("/run/secrets/hpcopt_api_keys")

_CACHE_TTL_SEC = float(os.getenv("HPCOPT_API_KEYS_CACHE_TTL", "30"))
_cached_keys: set[str] = set()
_cache_ts: float = 0.0
_CACHE_LOCK = threading.Lock()


_READ_TIMEOUT_SEC = 5.0


def _read_keys_file(path: Path) -> set[str]:
    """Read one API key per line, stripping blanks and comments.

    Enforces a timeout via a thread to prevent hangs on stale NFS mounts.
    """
    import concurrent.futures

    def _read() -> str:
        return path.read_text(encoding="utf-8")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            text = pool.submit(_read).result(timeout=_READ_TIMEOUT_SEC)
    except concurrent.futures.TimeoutError:
        logger.warning("Timed out reading API keys file after %.0fs: %s", _READ_TIMEOUT_SEC, path)
        return set()
    except OSError:
        logger.warning("Cannot read API keys file: %s", path)
        return set()
    return {line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")}


def _load_api_keys_uncached() -> set[str]:
    """Load keys from the configured source without caching."""
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


def load_api_keys() -> set[str]:
    """Return the current set of valid API keys (cached with TTL)."""
    global _cached_keys, _cache_ts
    with _CACHE_LOCK:
        now = time.monotonic()
        if now - _cache_ts < _CACHE_TTL_SEC and _cached_keys:
            return _cached_keys
        _cached_keys = _load_api_keys_uncached()
        _cache_ts = now
        return _cached_keys


def invalidate_api_keys_cache() -> None:
    """Force the next ``load_api_keys`` call to re-read from source."""
    global _cache_ts
    with _CACHE_LOCK:
        _cache_ts = 0.0
