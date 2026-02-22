# Changelog

All notable changes to HPC Workload Optimizer are documented here.

## [Unreleased]

### Added
- End-to-end pipeline integration test (ingest -> features -> train -> predict)
- Resilience module: retry decorator with exponential backoff, circuit breaker
- Retry on transient I/O errors during model loading
- Request draining on graceful shutdown (503 for new requests during drain)
- Cross-platform advisory file locking for model registry
- Backup-on-write for registry (.jsonl.bak)
- SBOM generation (SPDX) in release workflow
- OpenTelemetry instrumentation (optional, via `[tracing]` extra)
- Structured audit trail for model lifecycle operations
- Per-endpoint rate limiting (configurable via env vars)
- Dynamic log level endpoint (`POST /v1/admin/log-level`)
- Environment-specific configs (`configs/environments/{dev,staging,prod}.yaml`)
- Load tests wired into CI (main branch only)
- Pre-commit configuration (ruff, bandit, file checks)
- CONTRIBUTING.md and CHANGELOG.md

### Changed
- Rate limiter now keys buckets by `(api_key, endpoint)` instead of just `api_key`
- Model registry docstring updated to reflect cross-process file locking

### Security
- File-based API key management with 3-tier loading
- Bandit SAST scanning in CI
- Schema hardening: all 10 schemas locked with `additionalProperties: false`
- Docker secrets mount support

## [0.1.0] - Initial Release

### Added
- Multi-format ingestion pipeline (SWF, Slurm, PBS/Torque)
- Shadow ingestion daemon for incremental scheduler polling
- Time-safe feature engineering with chronological cross-validation
- Runtime quantile training, inference, and hyperparameter tuning
- Resource-fit modeling (fragmentation classifier + node size regressor)
- Model registry with register/promote/archive lifecycle
- Drift detection (PSI + metric degradation)
- Deterministic simulation for baseline and ML policies
- Stress scenario generation (4 regimes) and automated testing
- Recommendation engine with single-objective and Pareto modes
- Full credibility protocol with multi-trace orchestration
- Benchmark suite with regression gate and history tracking
- Batsim config/run wrappers with normalization
- Production-ready FastAPI with Prometheus metrics
- Docker containerization with pinned base images
- Modular CLI architecture (6 domain modules)
- Rust SWF parser and simulation runner with saturating arithmetic
- Cross-language adapter parity testing
- 89+ tests with 58% coverage gate in CI
