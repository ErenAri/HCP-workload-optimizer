# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-26

### Added
- Gymnasium-compatible RL policy search environment (`SchedulingEnv`) with random/grid search helpers.
- Expanded CLI/reporting test coverage and bridge/tracing tests.
- Version consistency validation script (`scripts/verify_version_consistency.py`) integrated into CI/release workflows.

### Changed
- Package version aligned to `2.0.0`.
- Test configuration now registers `performance` marker and filters known third-party warning noise.

## [1.2.0] - 2026-02-26

### Added
- Rust simulation runner integration path and parity checks.
- Expanded benchmark, deployment, and operations documentation.
- Docker smoke/load verification and release automation hardening.

## [1.0.0] - 2026-02-25

### Added
- End-to-end pipeline: ingest (SWF/Slurm/PBS) -> profile -> features -> train -> simulate -> recommend
- Runtime quantile regression models (p10/p50/p90) with gradient boosting
- Resource-fit prediction API for node-to-job matching
- Discrete-event scheduling simulator (FIFO, EASY_BACKFILL, ML_BACKFILL_P50)
- Simulation fidelity gate with invariant checking
- Recommendation engine with weighted objective scoring and constraint contracts
- Credibility protocol with hash-locked reference suites
- REST API (FastAPI) with RFC 7807 error responses, RBAC, rate limiting
- Prometheus metrics endpoint (`/metrics`)
- Docker + Kubernetes deployment manifests
- CI pipeline with ruff, mypy, bandit, pytest (330+ tests)
- CLI (`hpcopt`) for all pipeline stages
- Docker smoke test script (`scripts/docker_smoke_test.py`)
- Locust load test profile (`scripts/load/locustfile.py`)
- Quickstart demo (`examples/quickstart.py`)

### Validated Performance
- Container: 13/13 endpoint smoke tests pass, 128 MB memory, <1s startup
- Load test: 170 req/s, p50=6ms, p95=53ms (50 concurrent users)
- Model: 42.3% MAE improvement vs global mean on CTC-SP2 (77K jobs)
- Prediction interval coverage: 78.1% (p10-p90)

## [Unreleased]

### Planned
- Rust simulation engine (10-50x speedup)
- LightGBM / XGBoost model backends
- Reinforcement learning policy search
- Slurm live adapter

[2.0.0]: https://github.com/ErenAri/HCP-workload-optimizer/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/ErenAri/HCP-workload-optimizer/compare/v1.0.0...v1.2.0
[1.0.0]: https://github.com/ErenAri/HCP-workload-optimizer/releases/tag/v1.0.0
[Unreleased]: https://github.com/ErenAri/HCP-workload-optimizer/compare/v2.0.0...HEAD
