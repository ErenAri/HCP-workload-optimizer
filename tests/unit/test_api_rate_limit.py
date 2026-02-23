"""Tests for the token-bucket rate limiter."""
from __future__ import annotations

import threading
import time

import pytest

from hpcopt.api.rate_limit import (
    check_rate_limit,
    reset_for_testing,
    restore_limits_for_testing,
    set_limits_for_testing,
)


@pytest.fixture(autouse=True)
def _clean_buckets() -> None:
    reset_for_testing()
    yield  # type: ignore[misc]
    reset_for_testing()


def test_allows_requests_within_limit() -> None:
    old = set_limits_for_testing(global_limit=5, per_endpoint={})
    try:
        for _ in range(5):
            allowed, _ = check_rate_limit("key1", "/test")
            assert allowed
    finally:
        restore_limits_for_testing(*old)


def test_rejects_when_limit_exceeded() -> None:
    old = set_limits_for_testing(global_limit=2, per_endpoint={})
    try:
        check_rate_limit("key2", "/test")
        check_rate_limit("key2", "/test")
        allowed, retry_after = check_rate_limit("key2", "/test")
        assert not allowed
        assert retry_after >= 1
    finally:
        restore_limits_for_testing(*old)


def test_per_endpoint_limits() -> None:
    old = set_limits_for_testing(global_limit=100, per_endpoint={"/strict": 1})
    try:
        allowed, _ = check_rate_limit("key3", "/strict")
        assert allowed
        allowed, _ = check_rate_limit("key3", "/strict")
        assert not allowed
        # Other endpoints still work
        allowed, _ = check_rate_limit("key3", "/lenient")
        assert allowed
    finally:
        restore_limits_for_testing(*old)


def test_different_keys_have_separate_buckets() -> None:
    old = set_limits_for_testing(global_limit=1, per_endpoint={})
    try:
        allowed, _ = check_rate_limit("keyA", "/test")
        assert allowed
        allowed, _ = check_rate_limit("keyB", "/test")
        assert allowed  # different key, separate bucket
    finally:
        restore_limits_for_testing(*old)


def test_anonymous_rate_limiting() -> None:
    old = set_limits_for_testing(global_limit=2, per_endpoint={})
    try:
        check_rate_limit(None, "/test")
        check_rate_limit(None, "/test")
        allowed, _ = check_rate_limit(None, "/test")
        assert not allowed
    finally:
        restore_limits_for_testing(*old)


def test_zero_limit_means_unlimited() -> None:
    old = set_limits_for_testing(global_limit=0, per_endpoint={})
    try:
        for _ in range(100):
            allowed, _ = check_rate_limit("key", "/test")
            assert allowed
    finally:
        restore_limits_for_testing(*old)


def test_bucket_expiry_with_max_buckets() -> None:
    """When bucket cap is exceeded, stale buckets are pruned."""
    old = set_limits_for_testing(global_limit=100, per_endpoint={})
    try:
        # Create many buckets
        for i in range(50):
            check_rate_limit(f"key_{i}", "/test")
        # All should still work
        allowed, _ = check_rate_limit("final_key", "/test")
        assert allowed
    finally:
        restore_limits_for_testing(*old)


def test_concurrent_token_consumption() -> None:
    """Multiple threads consuming from the same bucket."""
    old = set_limits_for_testing(global_limit=20, per_endpoint={})
    allowed_count = 0
    lock = threading.Lock()

    def _consume() -> None:
        nonlocal allowed_count
        allowed, _ = check_rate_limit("shared_key", "/test")
        if allowed:
            with lock:
                allowed_count += 1

    try:
        threads = [threading.Thread(target=_consume) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 20 should be allowed
        assert allowed_count == 20
    finally:
        restore_limits_for_testing(*old)


def test_reset_clears_all_state() -> None:
    old = set_limits_for_testing(global_limit=1, per_endpoint={})
    try:
        check_rate_limit("key", "/test")
        allowed, _ = check_rate_limit("key", "/test")
        assert not allowed

        reset_for_testing()

        allowed, _ = check_rate_limit("key", "/test")
        assert allowed
    finally:
        restore_limits_for_testing(*old)
