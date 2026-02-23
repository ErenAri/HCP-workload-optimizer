"""Thread-safe runtime predictor cache with pre-warming support.

Caches the ``RuntimeQuantilePredictor`` instance so it is loaded once and
reused across requests. Provides a ``warm_cache()`` function for eager
loading at startup and a public testing API to reset state between tests.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from hpcopt.models.runtime_quantile import (
    RuntimeQuantilePredictor,
    resolve_runtime_model_dir,
)

logger = logging.getLogger(__name__)

_RUNTIME_PREDICTOR_CACHE: dict[str, RuntimeQuantilePredictor | Path | None] = {
    "model_dir": None,
    "predictor": None,
}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_predictor_with_retry(model_dir: Path) -> RuntimeQuantilePredictor:
    """Load the predictor with retry on transient I/O errors."""
    from hpcopt.utils.resilience import retry

    @retry(max_attempts=3, backoff_base=0.5, exceptions=(OSError, IOError))
    def _load() -> RuntimeQuantilePredictor:
        return RuntimeQuantilePredictor(model_dir)

    return _load()


def get_runtime_predictor() -> tuple[RuntimeQuantilePredictor | None, Path | None]:
    """Resolve model directory server-side only (env var or convention)."""
    resolved = resolve_runtime_model_dir()
    if resolved is None:
        return None, None

    with _MODEL_CACHE_LOCK:
        cached_dir = _RUNTIME_PREDICTOR_CACHE["model_dir"]
        if isinstance(cached_dir, Path) and cached_dir == resolved:
            cached_predictor = _RUNTIME_PREDICTOR_CACHE["predictor"]
            if isinstance(cached_predictor, RuntimeQuantilePredictor):
                return cached_predictor, resolved

        try:
            predictor = _load_predictor_with_retry(resolved)
        except (OSError, IOError):
            logger.error("Failed to load model from %s after retries", resolved)
            return None, None
        _RUNTIME_PREDICTOR_CACHE["model_dir"] = resolved
        _RUNTIME_PREDICTOR_CACHE["predictor"] = predictor
        return predictor, resolved


def warm_cache() -> bool:
    """Pre-load the predictor into cache. Returns True if a model was loaded."""
    predictor, _ = get_runtime_predictor()
    return predictor is not None


def is_loaded() -> bool:
    """Return True if a predictor is currently cached."""
    with _MODEL_CACHE_LOCK:
        return isinstance(_RUNTIME_PREDICTOR_CACHE.get("predictor"), RuntimeQuantilePredictor)


def reset_for_testing() -> None:
    """Clear the model cache. Intended for test fixtures only."""
    with _MODEL_CACHE_LOCK:
        _RUNTIME_PREDICTOR_CACHE["model_dir"] = None
        _RUNTIME_PREDICTOR_CACHE["predictor"] = None
