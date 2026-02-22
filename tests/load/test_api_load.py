"""Load tests for the API: concurrent requests, p95 latency verification."""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient

from hpcopt.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_runtime_request(client: TestClient) -> float:
    """Make a runtime predict request and return latency in seconds."""
    payload = {"requested_cpus": 4, "requested_runtime_sec": 3600}
    start = time.perf_counter()
    response = client.post("/v1/runtime/predict", json=payload)
    duration = time.perf_counter() - start
    assert response.status_code == 200
    return duration


@pytest.mark.load
def test_concurrent_runtime_predictions(client: TestClient) -> None:
    """Test that API handles concurrent requests within latency bounds."""
    n_requests = 20
    max_workers = 4
    latencies: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_make_runtime_request, client) for _ in range(n_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())

    assert len(latencies) == n_requests

    p50 = statistics.median(latencies)
    sorted_lat = sorted(latencies)
    p95_idx = int(0.95 * len(sorted_lat))
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

    # Fallback predictions should be fast (< 1s p95 for test client)
    assert p95 < 2.0, f"p95 latency too high: {p95:.3f}s"
    assert p50 < 1.0, f"p50 latency too high: {p50:.3f}s"


@pytest.mark.load
def test_health_under_load(client: TestClient) -> None:
    """Health endpoint remains responsive under load."""
    latencies: list[float] = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.get("/health")
        latencies.append(time.perf_counter() - start)
        assert response.status_code == 200

    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    assert p95 < 0.5
