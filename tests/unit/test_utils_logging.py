"""Tests for structured logging utilities."""
from __future__ import annotations

import json
import logging


def test_correlation_id_roundtrip() -> None:
    from hpcopt.utils.logging import set_correlation_id, get_correlation_id
    set_correlation_id("test-123")
    assert get_correlation_id() == "test-123"


def test_new_correlation_id() -> None:
    from hpcopt.utils.logging import new_correlation_id, get_correlation_id
    cid = new_correlation_id()
    assert isinstance(cid, str)
    assert len(cid) > 0
    assert get_correlation_id() == cid


def test_correlation_id_filter() -> None:
    from hpcopt.utils.logging import CorrelationIDFilter, set_correlation_id
    set_correlation_id("filter-test-456")
    f = CorrelationIDFilter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    assert f.filter(record) is True
    assert getattr(record, "correlation_id", None) == "filter-test-456"


def test_structured_formatter_json() -> None:
    from hpcopt.utils.logging import StructuredFormatter, set_correlation_id
    set_correlation_id("fmt-789")
    formatter = StructuredFormatter()
    record = logging.LogRecord("test.logger", logging.WARNING, "file.py", 10, "hello %s", ("world",), None)
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "WARNING"
    assert "timestamp" in parsed


def test_setup_logging_structured() -> None:
    from hpcopt.utils.logging import setup_logging
    setup_logging(level="DEBUG", format_mode="structured")
    root = logging.getLogger()
    assert root.level <= logging.DEBUG


def test_setup_logging_plain() -> None:
    from hpcopt.utils.logging import setup_logging
    setup_logging(level="INFO", format_mode="plain")
    root = logging.getLogger()
    assert root.level <= logging.INFO
