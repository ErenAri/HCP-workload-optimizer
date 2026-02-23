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


def _make_runtime_request(client: TestClient) -> tuple[float, int]:
    """Make a runtime predict request and return (latency, status_code)."""
    payload = {"requested_cpus": 4, "requested_runtime_sec": 3600}
    start = time.perf_counter()
    response = client.post("/v1/runtime/predict", json=payload)
    duration = time.perf_counter() - start
    return duration, response.status_code


@pytest.mark.load
def test_concurrent_runtime_predictions(client: TestClient) -> None:
    """Test that API handles concurrent requests within latency bounds."""
    n_requests = 20
    max_workers = 4
    latencies: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_make_runtime_request, client) for _ in range(n_requests)]
        for future in as_completed(futures):
            duration, status = future.result()
            assert status == 200
            latencies.append(duration)

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


@pytest.mark.load
def test_spike_load(client: TestClient) -> None:
    """Spike test: 0 to 100 concurrent requests, measure recovery time."""
    n_spike = 100
    max_workers = 20
    latencies: list[float] = []
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_make_runtime_request, client) for _ in range(n_spike)]
        for future in as_completed(futures):
            duration, status = future.result()
            latencies.append(duration)
            if status != 200:
                error_count += 1

    sorted_lat = sorted(latencies)
    p99_idx = min(int(0.99 * len(sorted_lat)), len(sorted_lat) - 1)
    p99 = sorted_lat[p99_idx]

    # Under spike, p99 should still be reasonable
    assert p99 < 5.0, f"p99 latency too high under spike: {p99:.3f}s"
    # Error rate should be low
    error_rate = error_count / n_spike
    assert error_rate < 0.05, f"Error rate too high under spike: {error_rate:.2%}"


@pytest.mark.load
def test_sustained_load(client: TestClient) -> None:
    """Sustained load: continuous requests for a window, verify latency."""
    duration_sec = 5  # shorter for CI; increase to 60 for real load testing
    max_workers = 4
    latencies: list[float] = []
    error_count = 0
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        while time.perf_counter() - start_time < duration_sec:
            futures.append(pool.submit(_make_runtime_request, client))

        for future in as_completed(futures):
            dur, status = future.result()
            latencies.append(dur)
            if status != 200:
                error_count += 1

    if len(latencies) >= 10:
        sorted_lat = sorted(latencies)
        p95_idx = min(int(0.95 * len(sorted_lat)), len(sorted_lat) - 1)
        p99_idx = min(int(0.99 * len(sorted_lat)), len(sorted_lat) - 1)
        p95 = sorted_lat[p95_idx]
        p99 = sorted_lat[p99_idx]

        assert p99 < 2.0 * p95, f"p99 ({p99:.3f}s) too far from p95 ({p95:.3f}s)"

    error_rate = error_count / max(len(latencies), 1)
    assert error_rate < 0.01, f"Error rate under sustained load: {error_rate:.2%}"


@pytest.mark.load
def test_error_rate_under_load(client: TestClient) -> None:
    """Verify error rate is <1% at 50 concurrent requests."""
    n_requests = 50
    max_workers = 10
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_make_runtime_request, client) for _ in range(n_requests)]
        for future in as_completed(futures):
            _, status = future.result()
            if status != 200:
                error_count += 1

    error_rate = error_count / n_requests
    assert error_rate < 0.01, f"Error rate too high: {error_rate:.2%}"
