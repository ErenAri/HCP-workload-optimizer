"""Negative / error-path tests for API, simulation, ingest, and model cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from hpcopt.api.app import app
from hpcopt.api.rate_limit import reset_for_testing as reset_rate_limit

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_rate_limit()
    yield  # type: ignore[misc]
    reset_rate_limit()


# --- API error paths ---


def test_malformed_json_body() -> None:
    response = client.post(
        "/v1/runtime/predict",
        content=b"{bad json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_missing_required_field() -> None:
    response = client.post("/v1/runtime/predict", json={})
    assert response.status_code == 422


def test_wrong_type_for_field() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": "not_a_number"},
    )
    assert response.status_code == 422


def test_negative_cpus() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": -1},
    )
    assert response.status_code == 422


def test_zero_cpus() -> None:
    response = client.post(
        "/v1/runtime/predict",
        json={"requested_cpus": 0},
    )
    assert response.status_code == 422


def test_empty_candidate_node_cpus() -> None:
    response = client.post(
        "/v1/resource-fit/predict",
        json={"requested_cpus": 4, "candidate_node_cpus": []},
    )
    assert response.status_code == 422


# --- Ingest error paths ---


def test_swf_empty_file(tmp_path: Path) -> None:
    from hpcopt.ingest.swf import ingest_swf

    empty = tmp_path / "empty.swf"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No parsable SWF"):
        ingest_swf(empty, tmp_path / "out", "empty", tmp_path / "rep")


def test_slurm_empty_file(tmp_path: Path) -> None:
    from hpcopt.ingest.slurm import ingest_slurm

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No parsable Slurm"):
        ingest_slurm(empty, tmp_path / "out", "empty", tmp_path / "rep")


def test_slurm_malformed_elapsed() -> None:
    from hpcopt.ingest.slurm import _parse_elapsed

    assert _parse_elapsed("") is None
    assert _parse_elapsed("Unknown") is None
    assert _parse_elapsed("not:a:time:at:all:really") is None


def test_slurm_overflow_timestamp() -> None:
    from hpcopt.ingest.slurm import _parse_slurm_datetime

    assert _parse_slurm_datetime("Unknown") is None
    assert _parse_slurm_datetime("None") is None
    assert _parse_slurm_datetime("not-a-date") is None


# --- Model cache error paths ---


def test_model_cache_transient_io_error() -> None:
    """IOError during model load should fail gracefully."""
    from hpcopt.api.model_cache import get_runtime_predictor, reset_for_testing

    reset_for_testing()

    fake_dir = Path("/tmp/hpcopt_test_nonexistent_model_dir_abc123")
    with patch("hpcopt.api.model_cache.resolve_runtime_model_dir", return_value=fake_dir):
        predictor, _ = get_runtime_predictor()
    assert predictor is None
    reset_for_testing()


# --- Simulation error paths ---


def test_simulation_unsupported_policy() -> None:
    import pandas as pd
    from hpcopt.simulate.core import run_simulation_from_trace

    df = pd.DataFrame(
        {
            "job_id": [1, 2],
            "submit_ts": [100, 200],
            "runtime_actual_sec": [60, 120],
            "requested_cpus": [4, 8],
        }
    )
    with pytest.raises((ValueError, KeyError)):
        run_simulation_from_trace(
            trace_df=df,
            policy_id="DOES_NOT_EXIST_POLICY",
            capacity_cpus=64,
            run_id="test_bad_policy",
        )
