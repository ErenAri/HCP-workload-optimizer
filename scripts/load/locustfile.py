"""Locust load test profile for HPC Workload Optimizer API.

Weighted endpoint mix simulating realistic traffic:
  - 40% runtime predictions
  - 20% resource-fit predictions
  - 20% health/ready checks
  - 10% system status
  - 10% recommendation lookups

Usage (headless):
    locust -f scripts/load/locustfile.py --headless \\
        -u 50 -r 5 -t 60s --host http://localhost:8080 \\
        --csv outputs/load_results
"""
from __future__ import annotations

import json

from locust import HttpUser, between, task


class HPCOptUser(HttpUser):
    """Simulated API consumer."""

    wait_time = between(0.1, 0.5)

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "test-benchmark-key",
    }

    @task(4)
    def runtime_predict(self) -> None:
        """POST /v1/runtime/predict — most common endpoint."""
        self.client.post(
            "/v1/runtime/predict",
            data=json.dumps({
                "requested_cpus": 8,
                "requested_runtime_sec": 3600.0,
                "queue_id": "batch",
                "user_id": "benchuser",
            }),
            headers=self.headers,
        )

    @task(2)
    def resource_fit(self) -> None:
        """POST /v1/resource-fit/predict."""
        self.client.post(
            "/v1/resource-fit/predict",
            data=json.dumps({
                "requested_cpus": 16,
                "candidate_node_cpus": [16, 32, 64, 128],
            }),
            headers=self.headers,
        )

    @task(2)
    def health_ready(self) -> None:
        """GET /health + /ready — monitoring probes."""
        self.client.get("/health")
        self.client.get("/ready")

    @task(1)
    def system_status(self) -> None:
        """GET /v1/system/status."""
        self.client.get("/v1/system/status", headers=self.headers)

    @task(1)
    def recommendation_miss(self) -> None:
        """GET /v1/recommendations — expected 404."""
        with self.client.get(
            "/v1/recommendations/benchmark-run-0001",
            headers=self.headers,
            catch_response=True,
        ) as resp:
            if resp.status_code == 404:
                resp.success()
