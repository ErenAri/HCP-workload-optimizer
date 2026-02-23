"""Shared test fixtures for the HPC Workload Optimizer test suite."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """TestClient with a clean rate-limit, model cache, and API key cache state."""
    from hpcopt.api.app import app
    from hpcopt.api.model_cache import reset_for_testing as reset_model_cache
    from hpcopt.api.rate_limit import reset_for_testing as reset_rate_limit
    from hpcopt.utils.secrets import invalidate_api_keys_cache

    reset_rate_limit()
    reset_model_cache()
    invalidate_api_keys_cache()
    yield TestClient(app)
    reset_rate_limit()
    invalidate_api_keys_cache()


@pytest.fixture
def sample_trace_path() -> Path:
    """Path to the sample SWF trace fixture."""
    return Path(__file__).parent / "fixtures" / "sample_trace.swf"


@pytest.fixture
def adapter_snapshot_path() -> Path:
    """Path to the adapter snapshot fixture."""
    return Path(__file__).parent / "fixtures" / "adapter_snapshot_case.json"


@pytest.fixture
def stress_dataset(tmp_path: Path):
    """Generate a small stress dataset for tests that need training data."""
    from hpcopt.simulate.stress import generate_stress_scenario

    return generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=120,
        seed=42,
        params={"alpha": 1.25},
    )
