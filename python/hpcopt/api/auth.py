"""API key authentication for the hpcopt REST API.

Centralises the set of paths exempt from authentication and rate limiting,
and provides a single function to check whether a request is authorised.
Keys are loaded via ``hpcopt.utils.secrets.load_api_keys`` which supports
file-based, Docker/K8s mount, and legacy env-var strategies with TTL caching.
"""
from __future__ import annotations

import hmac
import logging

from hpcopt.utils.secrets import load_api_keys

logger = logging.getLogger(__name__)

EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/v1/system/status",
})


def check_api_key_auth(path: str, provided_key: str) -> bool:
    """Return True if the request is authorised.

    A request is authorised when any of the following hold:
    - *path* is in ``EXEMPT_PATHS``
    - No API keys are configured (auth disabled)
    - *provided_key* is in the configured key set
    """
    if path in EXEMPT_PATHS:
        return True
    api_keys = load_api_keys()
    if not api_keys:
        return True
    return any(hmac.compare_digest(provided_key, k) for k in api_keys)
