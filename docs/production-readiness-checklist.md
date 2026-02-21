# Production Readiness Checklist

This project uses a machine-validated release checklist:

- canonical checklist file: `configs/release/production_readiness.yaml`
- validator script: `scripts/production_readiness_gate.py`
- staging verification workflow: `.github/workflows/staging-verify.yml`
- recurring ops drills: `.github/workflows/ops-drill.yml`

Validation modes:

- `validate`: schema/shape validation (runs on CI)
- `release`: strict release gating (all required checks must be `done` with evidence)

## Commands

Validate checklist structure:

```bash
python scripts/production_readiness_gate.py --mode validate
```

Run strict release gate:

```bash
python scripts/production_readiness_gate.py --mode release
```

## How To Update

Before cutting a release tag:

1. Update `metadata.reviewed_at_utc` in `configs/release/production_readiness.yaml`.
2. For every required control, set `status: done` and provide concrete `evidence`.
3. Run strict gate locally.
4. Push the checklist update before creating the release tag.

If any required item is `todo` or has empty evidence, the release workflow will fail.

## Operational Evidence

Controls are evidenced by repository artifacts and automated gates, including:

- runbooks in `docs/runbooks/`,
- SLO/ownership/DR docs in `docs/ops/`,
- security policies in `docs/security/`,
- API compatibility policy in `docs/api/versioning-and-deprecation.md`,
- OpenAPI baseline and checker (`schemas/openapi_baseline.json`, `scripts/check_openapi_compat.py`).
