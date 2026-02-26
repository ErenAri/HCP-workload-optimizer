#!/usr/bin/env python3
"""Docker container smoke test.

Verifies all API endpoints respond correctly when running in Docker.
Usage:
    docker compose up --build -d
    python scripts/docker_smoke_test.py
    docker compose down
"""

from __future__ import annotations

import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://localhost:8080"
API_KEY = "test-benchmark-key"
ADMIN_KEY = "admin-benchmark-key"

CHECKS: list[dict] = []


def check(
    name: str,
    method: str,
    path: str,
    body: dict | None = None,
    expected_status: int = 200,
    api_key: str | None = API_KEY,
    validate: callable = None,
) -> bool:
    """Run a single endpoint check."""
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, headers=headers, method=method)

    start = time.perf_counter()
    try:
        resp = urlopen(req, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = resp.status
        resp_body = (
            json.loads(resp.read().decode())
            if resp.headers.get("Content-Type", "").startswith("application/json")
            else resp.read().decode()
        )
    except HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = e.code
        resp_body = e.read().decode()
    except URLError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        CHECKS.append({"name": name, "status": "FAIL", "error": str(e), "latency_ms": round(elapsed_ms, 1)})
        return False

    passed = status == expected_status
    if passed and validate:
        try:
            validate(resp_body)
        except Exception as e:
            passed = False
            CHECKS.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "error": f"Validation: {e}",
                    "http_status": status,
                    "latency_ms": round(elapsed_ms, 1),
                }
            )
            return False

    CHECKS.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "http_status": status,
            "expected_status": expected_status,
            "latency_ms": round(elapsed_ms, 1),
        }
    )
    return passed


def wait_for_health(timeout: int = 30) -> bool:
    """Wait for container to be healthy."""
    print(f"Waiting for container health (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urlopen(Request(f"{BASE_URL}/health", method="GET"), timeout=3)
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"  Container healthy in {elapsed:.1f}s")
                CHECKS.append({"name": "startup_time", "value_sec": round(elapsed, 2), "status": "PASS"})
                return True
        except (URLError, HTTPError):
            pass
        time.sleep(1)

    CHECKS.append({"name": "startup_time", "status": "FAIL", "error": f"Timed out after {timeout}s"})
    return False


def main() -> int:
    print("=" * 60)
    print("HPC Workload Optimizer - Docker Smoke Test")
    print("=" * 60)

    # Wait for health
    if not wait_for_health():
        print("\nFAIL: Container did not become healthy")
        return 1

    # Health endpoint
    check("GET /health", "GET", "/health", api_key=None, validate=lambda r: assert_field(r, "status", "ok"))

    # Readiness endpoint
    check("GET /ready", "GET", "/ready", api_key=None)

    # System status
    check("GET /v1/system/status", "GET", "/v1/system/status", api_key=None)

    # Runtime prediction
    check(
        "POST /v1/runtime/predict",
        "POST",
        "/v1/runtime/predict",
        body={"requested_cpus": 8, "requested_runtime_sec": 3600},
        validate=lambda r: assert_fields_exist(r, ["runtime_p50_sec", "runtime_p90_sec", "runtime_guard_sec"]),
    )

    # Runtime prediction (minimal)
    check("POST /v1/runtime/predict (minimal)", "POST", "/v1/runtime/predict", body={"requested_cpus": 1})

    # Resource-fit prediction
    check(
        "POST /v1/resource-fit/predict",
        "POST",
        "/v1/resource-fit/predict",
        body={"requested_cpus": 8, "candidate_node_cpus": [16, 32, 64]},
        validate=lambda r: assert_fields_exist(r, ["fragmentation_risk", "recommendation"]),
    )

    # Metrics endpoint
    check("GET /metrics", "GET", "/metrics", api_key=None, validate=lambda r: "hpcopt" in str(r) if r else True)

    # Admin log level (requires admin key)
    check("POST /v1/admin/log-level (admin)", "POST", "/v1/admin/log-level", body={"level": "INFO"}, api_key=ADMIN_KEY)

    # Admin log level (non-admin key should 403)
    check(
        "POST /v1/admin/log-level (non-admin)",
        "POST",
        "/v1/admin/log-level",
        body={"level": "DEBUG"},
        expected_status=403,
        api_key=API_KEY,
    )

    # Auth enforcement (no key should 401)
    check(
        "POST /predict (no key → 401)",
        "POST",
        "/v1/runtime/predict",
        body={"requested_cpus": 1},
        expected_status=401,
        api_key=None,
    )

    # Validation (invalid input → 422)
    check(
        "POST /predict (invalid → 422)", "POST", "/v1/runtime/predict", body={"requested_cpus": 0}, expected_status=422
    )

    # Recommendation not found → 404
    check(
        "GET /recommendations/nonexistent → 404", "GET", "/v1/recommendations/nonexistent-run-id", expected_status=404
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for c in CHECKS if c.get("status") == "PASS")
    failed = sum(1 for c in CHECKS if c.get("status") == "FAIL")

    for c in CHECKS:
        icon = "✓" if c["status"] == "PASS" else "✗"
        latency = f" ({c['latency_ms']}ms)" if "latency_ms" in c else ""
        extra = (
            f" [expected={c.get('expected_status')}, got={c.get('http_status')}]"
            if c.get("status") == "FAIL" and "http_status" in c
            else ""
        )
        print(f"  {icon} {c['name']}{latency}{extra}")

    print(f"\n{passed} passed, {failed} failed")

    # Save results to JSON
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "checks": CHECKS,
        "summary": {"passed": passed, "failed": failed, "total": len(CHECKS)},
    }
    with open("outputs/docker_smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/docker_smoke_results.json")

    return 0 if failed == 0 else 1


def assert_field(obj: dict, key: str, value: object) -> None:
    assert obj.get(key) == value, f"Expected {key}={value}, got {obj.get(key)}"


def assert_fields_exist(obj: dict, keys: list[str]) -> None:
    for k in keys:
        assert k in obj, f"Missing field: {k}"


if __name__ == "__main__":
    sys.exit(main())
