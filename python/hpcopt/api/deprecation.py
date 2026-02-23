"""Deprecation configuration loading for the hpcopt REST API.

Loads deprecated endpoint entries from ``configs/api/deprecation.yaml`` and
provides a public testing API so tests can inject/reset entries without
reaching into private module state.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DEPRECATION_CONFIG_PATH = Path("configs/api/deprecation.yaml")
_DEPRECATION_ENTRIES: list[dict[str, str]] = []
_LOCK = threading.Lock()


def load_deprecation_config() -> list[dict[str, str]]:
    """Load deprecated endpoint entries from the deprecation config file (cached)."""
    global _DEPRECATION_ENTRIES
    with _LOCK:
        if _DEPRECATION_ENTRIES:
            return _DEPRECATION_ENTRIES
        try:
            import yaml
            if _DEPRECATION_CONFIG_PATH.exists():
                data = yaml.safe_load(_DEPRECATION_CONFIG_PATH.read_text(encoding="utf-8"))
                _DEPRECATION_ENTRIES = data.get("deprecated_endpoints", []) or []
        except (OSError, ValueError, yaml.YAMLError):
            logger.debug("Could not load deprecation config", exc_info=True)
        return _DEPRECATION_ENTRIES


def set_entries_for_testing(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace entries for testing. Returns previous entries for restore."""
    global _DEPRECATION_ENTRIES
    with _LOCK:
        old = _DEPRECATION_ENTRIES
        _DEPRECATION_ENTRIES = entries
    return old


def reset_for_testing() -> None:
    """Clear cached deprecation entries."""
    global _DEPRECATION_ENTRIES
    with _LOCK:
        _DEPRECATION_ENTRIES = []
