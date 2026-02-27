# HPC Workload Optimizer — Common Development Commands
# Cross-platform Makefile for consistent DX

.PHONY: lint typecheck test test-unit test-integration test-load coverage coverage-gate docs-check openapi-check readiness-check security serve docker-build docker-run rust-check clean verify help

# ---------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------

lint: ## Run ruff linter
	ruff check python/

typecheck: ## Run mypy type checker
	mypy python/hpcopt/ --ignore-missing-imports

security: ## Run Bandit SAST scanner
	bandit -r python/hpcopt/ -ll -ii

# ---------------------------------------------------------------
# Testing
# ---------------------------------------------------------------

test: ## Run full test suite
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-load: ## Run load tests
	pytest tests/load/ -v -m load

coverage: ## Run tests with coverage reporting
	pytest tests/ -v --cov=hpcopt --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=88

coverage-gate: coverage ## Enforce package-level coverage floors
	python scripts/check_coverage_thresholds.py --coverage-xml coverage.xml --global-fail-under 88 --package-threshold api=88 --package-threshold models=89 --package-threshold simulate=86

docs-check: ## Validate docs consistency against CLI/runtime interfaces
	python scripts/check_docs_consistency.py

openapi-check: ## Validate OpenAPI backward compatibility
	python scripts/check_openapi_compat.py --baseline schemas/openapi_baseline.json

readiness-check: ## Validate production readiness checklist shape
	python scripts/production_readiness_gate.py --mode validate

# ---------------------------------------------------------------
# Local Development
# ---------------------------------------------------------------

serve: ## Start local API server
	uvicorn hpcopt.api.app:app --reload --host 0.0.0.0 --port 8000

# ---------------------------------------------------------------
# Docker
# ---------------------------------------------------------------

docker-build: ## Build Docker image
	docker build -t hpcopt-api:dev .

docker-run: docker-build ## Run Docker container locally
	docker run --rm -p 8000:8000 hpcopt-api:dev

# ---------------------------------------------------------------
# Rust
# ---------------------------------------------------------------

rust-check: ## Run Rust checks (check, clippy, test)
	cd rust && cargo check --workspace
	cd rust && cargo clippy --workspace -- -D warnings
	cd rust && cargo test --workspace

# ---------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------

lock: ## Regenerate dependency lockfile
	pip-compile --generate-hashes --output-file=requirements.lock requirements.txt

# ---------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

verify: ## Run full CI-equivalent verification
	$(MAKE) lint typecheck security coverage-gate docs-check openapi-check readiness-check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
