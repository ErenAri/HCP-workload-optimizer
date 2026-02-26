"""Concurrency tests: model cache, rate limiter, registry, secrets."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest


def test_model_cache_concurrent_access(tmp_path: Path) -> None:
    """50 threads requesting predictor simultaneously should not crash."""
    from hpcopt.api.model_cache import get_runtime_predictor, reset_for_testing

    reset_for_testing()
    results: list[object] = []
    errors: list[Exception] = []

    def _get() -> None:
        try:
            p, _ = get_runtime_predictor()
            results.append(p)
        except Exception as e:
            errors.append(e)

    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=None):
        threads = [threading.Thread(target=_get) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(errors) == 0
    assert len(results) == 50
    reset_for_testing()


def test_rate_limiter_concurrent_consumption() -> None:
    """50 threads consuming from the same bucket concurrently."""
    from hpcopt.api.rate_limit import (
        check_rate_limit,
        reset_for_testing,
        restore_limits_for_testing,
        set_limits_for_testing,
    )

    reset_for_testing()
    old = set_limits_for_testing(global_limit=30, per_endpoint={})
    allowed_count = 0
    lock = threading.Lock()

    def _consume() -> None:
        nonlocal allowed_count
        allowed, _ = check_rate_limit("concurrent_key", "/test")
        if allowed:
            with lock:
                allowed_count += 1

    try:
        threads = [threading.Thread(target=_consume) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count == 30
    finally:
        restore_limits_for_testing(*old)
        reset_for_testing()


def test_registry_concurrent_registration(tmp_path: Path) -> None:
    """10 threads registering models concurrently should all succeed."""
    from hpcopt.models.registry import ModelRegistry

    reg = ModelRegistry(registry_path=tmp_path / "concurrent_reg.jsonl")
    errors: list[Exception] = []

    def _register(idx: int) -> None:
        try:
            d = tmp_path / f"model_{idx}"
            d.mkdir(exist_ok=True)
            reg.register(f"model_{idx}", str(d))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_register, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(reg.list()) == 10


def test_secrets_concurrent_load_during_cache_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """20 threads calling load_api_keys while cache is expiring."""
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("key-1\nkey-2\nkey-3\n", encoding="utf-8")
    monkeypatch.setenv("HPCOPT_API_KEYS_FILE", str(keys_file))

    from hpcopt.utils.secrets import invalidate_api_keys_cache, load_api_keys

    results: list[set[str]] = []
    errors: list[Exception] = []

    def _load() -> None:
        try:
            invalidate_api_keys_cache()  # force reload
            keys = load_api_keys()
            results.append(keys)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_load) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 20
    for keys in results:
        assert "key-1" in keys
        assert "key-2" in keys
        assert "key-3" in keys

    monkeypatch.delenv("HPCOPT_API_KEYS_FILE", raising=False)
    invalidate_api_keys_cache()
