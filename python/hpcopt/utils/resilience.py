"""Resilience primitives: retry decorator and circuit breaker."""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (OSError, IOError),
) -> Callable[[F], F]:
    """Decorator that retries on specified exceptions with exponential backoff."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = backoff_base * (2 ** (attempt - 1))
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__,
                            attempt,
                            max_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__,
                            max_attempts,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class CircuitBreaker:
    """Simple circuit breaker with failure threshold and auto-reset timeout.

    States:
      - CLOSED: normal operation, calls pass through
      - OPEN: calls are rejected immediately (raises ``CircuitOpenError``)
      - HALF_OPEN: one probe call allowed; success resets, failure re-opens
    """

    class CircuitOpenError(RuntimeError):
        """Raised when the circuit is open and calls are rejected."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open" and (time.time() - self._last_failure_time) >= self._reset_timeout:
                self._state = "half_open"
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning("Circuit breaker opened after %d failures", self._failure_count)

    def __call__(self, fn: F) -> F:
        """Use as a decorator."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.state == "open":
                raise self.CircuitOpenError(f"Circuit open for {fn.__name__}")
            try:
                result = fn(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper  # type: ignore[return-value]
