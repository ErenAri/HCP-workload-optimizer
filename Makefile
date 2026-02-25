# HPC Workload Optimizer — Common Development Commands
# Cross-platform Makefile for consistent DX

.PHONY: lint typecheck test test-unit test-integration test-load coverage security serve docker-build docker-run rust-check clean verify help

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
	pytest tests/ -v --cov=hpcopt --cov-report=term-missing --cov-fail-under=82

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
	$(MAKE) lint typecheck security test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
