# Contributing to HPC Workload Optimizer

## Development Setup

```bash
# Clone and install
git clone <repo-url>
cd HCP-workload-optimizer
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Code Quality

All code must pass:

- **Lint**: `ruff check python/`
- **Type-check**: `mypy python/hpcopt/ --ignore-missing-imports`
- **Tests**: `pytest tests/ -v` (89+ tests, 58% coverage gate)
- **Security**: `bandit -r python/hpcopt/ -ll -ii`

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run without slow tests (Rust cross-language parity)
pytest tests/ -v -m "not slow"

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/load/ -v -m load
```

## Rust Components

```bash
cd rust
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo build --release
```

## Branch Strategy

- `main` is the primary integration branch
- Feature branches should target `main`
- CI must pass before merge

## Commit Messages

Use concise, imperative-mood commit messages:
- `Add retry decorator for model loading`
- `Fix race condition in registry writes`
- `Harden adapter schema with enum constraints`

## Architecture

See [docs/02-system-architecture.md](docs/02-system-architecture.md) for the full architecture overview.

## Schema Changes

All schemas in `schemas/` enforce `additionalProperties: false`. When modifying schemas:

1. Update the schema JSON file
2. Verify `pytest tests/unit/test_schema_validation.py -v` passes
3. Update any corresponding documentation in `docs/08-reproducibility-and-artifacts.md`

## Security

- Never commit secrets or API keys
- Use file-based secrets loading (see `docs/security/secrets-management.md`)
- All new code is scanned by Bandit SAST in CI
