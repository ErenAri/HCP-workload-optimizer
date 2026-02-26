"""Tests for retry decorator and circuit breaker."""

from __future__ import annotations

import time

import pytest
from hpcopt.utils.resilience import CircuitBreaker, retry


class TestRetryDecorator:
    def test_succeeds_immediately(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01, exceptions=(ValueError,))
        def ok():
            nonlocal call_count
            call_count += 1
            return 42

        assert ok() == 42
        assert call_count == 1

    def test_retries_then_succeeds(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01, exceptions=(OSError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_exhausts_attempts_then_raises(self) -> None:
        call_count = 0

        @retry(max_attempts=2, backoff_base=0.01, exceptions=(IOError,))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise IOError("permanent")

        with pytest.raises(IOError, match="permanent"):
            always_fail()
        assert call_count == 2

    def test_does_not_catch_unrelated_exceptions(self) -> None:
        @retry(max_attempts=3, backoff_base=0.01, exceptions=(OSError,))
        def bad():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            bad()

    def test_preserves_function_name(self) -> None:
        @retry(max_attempts=2, backoff_base=0.01)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        assert cb.state == "closed"

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_success_resets_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == "closed"

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.06)
        assert cb.state == "half_open"

    def test_decorator_usage_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)

        @cb
        def ok():
            return "result"

        assert ok() == "result"
        assert cb.state == "closed"

    def test_decorator_usage_failure_opens_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60.0)

        @cb
        def fail():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                fail()

        assert cb.state == "open"

    def test_open_circuit_rejects_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=60.0)

        @cb
        def fn():
            return "ok"

        # Trigger opening
        cb.record_failure()
        assert cb.state == "open"

        with pytest.raises(CircuitBreaker.CircuitOpenError):
            fn()

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

        @cb
        def fn():
            return "ok"

        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.06)
        assert cb.state == "half_open"

        # Successful call in half_open → closed
        assert fn() == "ok"
        assert cb.state == "closed"
