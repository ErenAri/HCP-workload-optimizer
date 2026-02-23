"""Token-bucket rate limiter for the hpcopt REST API.

Provides per-endpoint rate limiting keyed by (api_key, endpoint). Limits are
configurable via environment variables. A public testing API is exposed so
tests can override limits and clear buckets without reaching into private state.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_RATE_WINDOW_SEC = 60.0  # sliding window for token-bucket rate limiter
RATE_LIMIT = int(os.getenv("HPCOPT_RATE_LIMIT", "60"))  # requests per minute (global default)
PER_ENDPOINT_LIMITS: dict[str, int] = {
    "/v1/runtime/predict": int(os.getenv("HPCOPT_RATE_LIMIT_PREDICT", str(RATE_LIMIT))),
    "/v1/resource-fit/predict": int(os.getenv("HPCOPT_RATE_LIMIT_RESOURCE_FIT", str(RATE_LIMIT))),
    "/v1/admin/log-level": int(os.getenv("HPCOPT_RATE_LIMIT_ADMIN", "10")),
}
_RATE_MAX_BUCKETS = 10_000  # cap to prevent unbounded memory growth
_RATE_STALE_AGE_SEC = 90.0  # proactively age out buckets older than this
_RATE_BUCKETS: dict[str, list[float]] = {}
_RATE_LOCK = threading.Lock()
_RATE_LAST_PRUNE: float = 0.0  # monotonic timestamp of last proactive prune


def check_rate_limit(api_key: str | None, path: str = "") -> tuple[bool, int]:
    """Token-bucket rate limiter keyed by (api_key, endpoint). Returns (allowed, retry_after_sec)."""
    limit = PER_ENDPOINT_LIMITS.get(path, RATE_LIMIT)
    if limit <= 0:
        return True, 0
    bucket_key = f"{api_key or '__anonymous__'}:{path}"
    now = time.time()
    window_start = now - _RATE_WINDOW_SEC

    with _RATE_LOCK:
        global _RATE_LAST_PRUNE
        # Proactively evict stale buckets every 30 s or when cap is exceeded.
        stale_cutoff = now - _RATE_STALE_AGE_SEC
        if len(_RATE_BUCKETS) > _RATE_MAX_BUCKETS or now - _RATE_LAST_PRUNE > 30.0:
            stale_keys = [
                k for k, v in _RATE_BUCKETS.items()
                if not v or v[-1] <= stale_cutoff
            ]
            for k in stale_keys:
                del _RATE_BUCKETS[k]
            _RATE_LAST_PRUNE = now

        bucket = _RATE_BUCKETS.get(bucket_key, [])
        bucket = [ts for ts in bucket if ts > window_start]

        if len(bucket) >= limit:
            oldest = bucket[0]
            retry_after = max(1, int(oldest + _RATE_WINDOW_SEC - now))
            _RATE_BUCKETS[bucket_key] = bucket
            return False, retry_after
        bucket.append(now)
        _RATE_BUCKETS[bucket_key] = bucket
    return True, 0


def reset_for_testing() -> None:
    """Clear all rate-limit buckets and reset prune timestamp."""
    global _RATE_LAST_PRUNE
    with _RATE_LOCK:
        _RATE_BUCKETS.clear()
        _RATE_LAST_PRUNE = 0.0


def set_limits_for_testing(
    global_limit: int | None = None,
    per_endpoint: dict[str, int] | None = None,
) -> tuple[int, dict[str, int]]:
    """Override limits for a test. Returns (old_global_limit, old_per_endpoint) for restore."""
    global RATE_LIMIT, PER_ENDPOINT_LIMITS
    old_limit = RATE_LIMIT
    old_per_endpoint = dict(PER_ENDPOINT_LIMITS)
    if global_limit is not None:
        RATE_LIMIT = global_limit
    if per_endpoint is not None:
        PER_ENDPOINT_LIMITS = per_endpoint
    return old_limit, old_per_endpoint


def restore_limits_for_testing(
    global_limit: int,
    per_endpoint: dict[str, int],
) -> None:
    """Restore limits after a test."""
    global RATE_LIMIT, PER_ENDPOINT_LIMITS
    RATE_LIMIT = global_limit
    PER_ENDPOINT_LIMITS = per_endpoint
