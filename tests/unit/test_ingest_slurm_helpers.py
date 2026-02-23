"""Tests for Slurm ingestion helper functions."""
from __future__ import annotations

from hpcopt.ingest.slurm import _classify_job_id, _parse_elapsed, _parse_reqmem


# --- _parse_elapsed ---

def test_parse_elapsed_hhmmss() -> None:
    assert _parse_elapsed("01:30:00") == 5400


def test_parse_elapsed_dd_hhmmss() -> None:
    assert _parse_elapsed("2-01:00:00") == 2 * 86400 + 3600


def test_parse_elapsed_mmss() -> None:
    assert _parse_elapsed("05:30") == 330


def test_parse_elapsed_seconds_only() -> None:
    assert _parse_elapsed("120") == 120


def test_parse_elapsed_empty() -> None:
    assert _parse_elapsed("") is None
    assert _parse_elapsed("unknown") is None
    assert _parse_elapsed("n/a") is None


def test_parse_elapsed_bad_day() -> None:
    assert _parse_elapsed("abc-01:00:00") is None


def test_parse_elapsed_bad_parts() -> None:
    assert _parse_elapsed("a:b:c") is None


# --- _parse_reqmem ---

def test_parse_reqmem_megabytes() -> None:
    assert _parse_reqmem("4000Mc") == 4000


def test_parse_reqmem_gigabytes() -> None:
    assert _parse_reqmem("8Gn") == 8 * 1024


def test_parse_reqmem_kilobytes() -> None:
    result = _parse_reqmem("1024Kc")
    assert result == 1  # 1024K / 1024 = 1 MB


def test_parse_reqmem_terabytes() -> None:
    assert _parse_reqmem("1Tc") == 1024 * 1024


def test_parse_reqmem_empty() -> None:
    assert _parse_reqmem("") is None
    assert _parse_reqmem("0") is None
    assert _parse_reqmem("unknown") is None


def test_parse_reqmem_invalid() -> None:
    assert _parse_reqmem("abcMc") is None


# --- _classify_job_id ---

def test_classify_plain_job() -> None:
    job_id, array_idx, is_step = _classify_job_id("12345")
    assert job_id == "12345"
    assert array_idx is None
    assert is_step is False


def test_classify_array_job() -> None:
    job_id, array_idx, is_step = _classify_job_id("12345_42")
    assert job_id == "12345_42"
    assert array_idx == "42"
    assert is_step is False


def test_classify_job_step() -> None:
    job_id, array_idx, is_step = _classify_job_id("12345.batch")
    assert job_id == "12345"
    assert is_step is True


def test_classify_array_step() -> None:
    job_id, array_idx, is_step = _classify_job_id("12345_42.0")
    assert job_id == "12345_42"
    assert is_step is True
