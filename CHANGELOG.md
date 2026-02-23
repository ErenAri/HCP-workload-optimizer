# Changelog

All notable changes to HPC Workload Optimizer are documented here.

## [1.2.0] - 2026-02-23

### Added
- **API Security**: Request body size limit (1MB middleware), Pydantic input bounds (le=, max_length=, extra="forbid"), admin RBAC with `admin-` key prefix for `/v1/admin/*` paths
- **Kubernetes resilience**: PodDisruptionBudget (`k8s/pdb.yaml`, minAvailable: 1), NetworkPolicy (`k8s/network-policy.yaml`, ingress from ingress-nginx + monitoring only), preStop lifecycle hook (sleep 5s for connection draining)
- **RFC 7807 Problem Details** error responses (replaces `{"error": {...}}` format) with `type`, `title`, `status`, `detail`, `instance` fields
- **Circuit breaker** on prediction path (5-failure threshold, 60s reset, fallback on open)
- **New Prometheus metrics**: `rate_limit_rejections_total`, `auth_failures_total`, `cache_hits_total`, `model_load_duration_seconds`
- **GET /v1/recommendations/{run_id}** endpoint for retrieving stored recommendation results
- **Model card generation** (`models/model_card.py`): dataset characteristics, performance metrics, fairness/bias evaluation, limitations — output alongside model artifacts at training time
- **Startup config validation**: validates environment variables (`HPCOPT_RATE_LIMIT`, `HPCOPT_REQUEST_TIMEOUT_SEC`, `HPCOPT_ENV`) at API startup with fail-fast on invalid config
- **Audit logging** for admin log-level changes (who, when, old→new level)
- **Ingestion file size guards**: 2GB max file size, 1M line length, 50M row cap (SWF, Slurm, PBS parsers)
- **Timeout on secrets file reads** (5s) to prevent hangs on stale NFS mounts
- **Docker Compose resource limits** (cpus: 1.0, memory: 512M) and read-only volume mounts
- 20+ new test files: security tests, concurrency tests, error path tests, model cache, rate limit, metrics, registry, drift, resource-fit, PBS, shadow, retention, report export, feature importance, config validation, env config, logging, tuning, credibility dossier, tracing, sensitivity, recommendation engine, Slurm helpers
- **Load tests**: spike (0→100 concurrent), sustained (5s continuous), error rate verification (<1%), tail latency assertions (p99 < 2x p95)
- **Property-based tests** strengthened: max_examples=100, CPU conservation law, temporal ordering invariant, metric monotonicity

### Changed
- Coverage gate raised from 58% to **82%** (324 tests, 83% actual coverage)
- `/ready` endpoint returns 503 when degraded (disk low or shutting down)
- Broad `except Exception` replaced with specific exception types across 11 modules (16 locations)
- Error responses migrated from `{"error": {"code", "message", "trace_id"}}` to RFC 7807 format

### Security
- Admin RBAC: `admin-` prefixed API keys required for `/v1/admin/*` paths (403 for non-admin keys)
- Request body capped at 1MB (413 PAYLOAD_TOO_LARGE)
- Input bounds: `requested_cpus` ≤ 100,000; `queue_depth_jobs` ≤ 1,000,000; `requested_runtime_sec` ≤ 31,536,000; `candidate_node_cpus` max length 1,000
- Extra fields rejected on all request models (`extra="forbid"`)
- Kubernetes NetworkPolicy restricts ingress to ingress-nginx and monitoring namespaces

## [1.1.0] - 2026-02-23

### Changed
- **API architecture refactoring**: decomposed `api/app.py` into 4 focused modules:
  - `api/auth.py` -- API key authentication with `EXEMPT_PATHS` constant and `check_api_key_auth()` (replaces unused legacy ASGI middleware)
  - `api/rate_limit.py` -- token-bucket rate limiter with public testing API (`set_limits_for_testing()`, `reset_for_testing()`)
  - `api/model_cache.py` -- thread-safe runtime predictor cache with startup pre-warming (`warm_cache()`) and public testing API (`reset_for_testing()`)
  - `api/deprecation.py` -- deprecation config loading with public testing API (`set_entries_for_testing()`, `reset_for_testing()`)
- Tests now use public testing APIs from extracted modules instead of reaching into private module globals
- `patch` targets updated from `hpcopt.api.app` to source modules (e.g., `hpcopt.api.model_cache.resolve_runtime_model_dir`)

### Added
- Request timeout: all requests subject to configurable timeout (default 30s via `HPCOPT_REQUEST_TIMEOUT_SEC`), returns `504 GATEWAY_TIMEOUT` on expiry
- Model cache pre-warming at API startup for faster cold-start response
- `test_request_timeout_returns_504` contract test

## [1.0.0] - 2026-02-22

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
- Pre-commit configuration (ruff, bandit, file checks, mypy, hadolint)
- CONTRIBUTING.md and CHANGELOG.md
- **Production readiness improvements:**
  - Property-based tests (Hypothesis) for simulation, fidelity, adapter, objective, and recommendation
  - Coverage gate raised from 58% to 75%
  - Automated E2E smoke test in CI pipeline
  - Kubernetes manifests (Deployment, Service, ConfigMap, Secret, HPA, ServiceMonitor)
  - OpenTelemetry Collector configuration (`k8s/otel-collector.yaml`)
  - Alertmanager configuration with PagerDuty + Slack routing
  - API deprecation sunset mechanism (RFC 8594/9745 headers)
  - `.env.example` and secrets bootstrap script
  - `py.typed` PEP 561 marker
  - Codecov configuration for PR coverage reporting
  - Operations documentation: logging, scaling, persistent state, tracing
  - `LICENSE` file (Proprietary)

### Changed
- Rate limiter now keys buckets by `(api_key, endpoint)` instead of just `api_key`
- Model registry docstring updated to reflect cross-process file locking
- Version bumped from 0.1.0 to 1.0.0
- mypy configuration tightened: `disallow_untyped_defs = true`
- Removed deprecated `version` key from `docker-compose.yaml`
- Pre-commit hooks expanded with mypy and hadolint

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
