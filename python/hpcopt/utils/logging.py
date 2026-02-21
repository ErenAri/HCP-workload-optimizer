"""Structured logging utilities for hpcopt.

Provides JSON-formatted log output with correlation ID propagation
for request tracing across the HPC workload optimizer pipeline.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Correlation-ID management (thread-safe via contextvars)
# ---------------------------------------------------------------------------

_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current execution context."""
    _correlation_id_var.set(cid)


def get_correlation_id() -> str:
    """Return the current correlation ID, or an empty string if unset."""
    return _correlation_id_var.get()


def new_correlation_id() -> str:
    """Generate a new UUID-based correlation ID and store it in context."""
    cid = uuid.uuid4().hex[:16]
    set_correlation_id(cid)
    return cid


# ---------------------------------------------------------------------------
# Filter: injects correlation_id into every LogRecord
# ---------------------------------------------------------------------------


class CorrelationIDFilter(logging.Filter):
    """Logging filter that attaches ``correlation_id`` to each record.

    The value is read from the module-level :pydata:`_correlation_id_var`
    context variable so that it automatically follows ``asyncio`` tasks and
    threads that copy the context.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Formatter: JSON structured output
# ---------------------------------------------------------------------------


class StructuredFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields included:
    - ``timestamp``      – ISO-8601 UTC string
    - ``level``          – log level name (INFO, WARNING, …)
    - ``logger``         – logger name
    - ``message``        – formatted message
    - ``correlation_id`` – request / pipeline trace ID (may be empty)

    Any *extra* keys attached to the record are also included under an
    ``extra`` sub-object so that callers can pass structured context via
    ``logger.info("msg", extra={...})``.
    """

    # Keys that belong to the standard LogRecord and should not leak into
    # the ``extra`` sub-object.
    _BUILTIN_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
            "correlation_id",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": self._iso_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "correlation_id": getattr(record, "correlation_id", ""),
        }

        # Collect extra fields supplied by the caller.
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._BUILTIN_ATTRS and not k.startswith("_")
        }
        if extra:
            payload["extra"] = extra

        # Append exception info when present.
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exception"] = record.exc_text

        return json.dumps(payload, default=str)

    @staticmethod
    def _iso_timestamp(record: logging.LogRecord) -> str:
        return time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
        ) + f".{int(record.msecs):03d}Z"


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

_STANDARD_FORMAT = (
    "%(asctime)s [%(levelname)-8s] %(name)s "
    "(%(correlation_id)s) %(message)s"
)


def setup_logging(
    level: str = "INFO",
    format_mode: str = "structured",
) -> None:
    """Configure the root logger for *hpcopt*.

    Parameters
    ----------
    level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    format_mode:
        ``"structured"`` for JSON output (default), anything else for a
        human-readable line format.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any pre-existing handlers to avoid duplicate output.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)

    if format_mode == "structured":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(_STANDARD_FORMAT))

    # Always inject correlation_id so both formatters can reference it.
    handler.addFilter(CorrelationIDFilter())

    root.addHandler(handler)
