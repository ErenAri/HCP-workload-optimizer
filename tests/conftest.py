"""Shared test fixtures for the HPC Workload Optimizer test suite."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """TestClient with a clean rate-limit and model cache state."""
    import hpcopt.api.app as api_module
    from hpcopt.api.app import app

    api_module._RATE_BUCKETS.clear()
    api_module._RUNTIME_PREDICTOR_CACHE["model_dir"] = None
    api_module._RUNTIME_PREDICTOR_CACHE["predictor"] = None
    yield TestClient(app)
    api_module._RATE_BUCKETS.clear()


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
