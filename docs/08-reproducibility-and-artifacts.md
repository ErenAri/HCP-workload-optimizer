# Reproducibility, Schemas, and Artifacts

## 1. Reproducibility Philosophy

The project treats reproducibility as a contract requirement, not a convenience.

Each meaningful command emits machine-readable artifacts plus a run manifest that captures:

- command identity,
- input and output hashes,
- environment and tool versions,
- policy hash and config snapshots,
- seed values (where applicable),
- immutable manifest self-hash.

## 2. Manifest Implementation

Module:
- `python/hpcopt/artifacts/manifest.py`

Primary functions:
- `build_manifest(...)`
- `write_manifest(...)`

Manifest features:

- `git_commit` (when repository metadata is available),
- Python package versions,
- `cargo` and `rustc` versions,
- environment fingerprint,
- policy spec hash,
- dependency lock hash,
- optional config inline snapshots (size-limited).

## 3. Schema Contracts

### `schemas/run_manifest.schema.json`

Defines structure for immutable run manifests.

### `schemas/invariant_report.schema.json`

Defines invariant report shape:
- run ID,
- strict mode flag,
- step count,
- violation records.

### `schemas/fidelity_report.schema.json`

Defines fidelity report essentials:
- run ID,
- pass/fail status,
- aggregate metrics,
- distribution metrics.

### Adapter Schemas

- `schemas/adapter_snapshot.schema.json`
- `schemas/adapter_decision.schema.json` (with `enum` constraint on `policy_id`: `FIFO_STRICT`, `EASY_BACKFILL_BASELINE`, `ML_BACKFILL_P50`)

These enforce scheduler boundary compatibility.

### Schema Hardening

All 10 schemas enforce `additionalProperties: false` at root level (except `run_manifest.environment` which allows arbitrary keys). This prevents silent schema drift. The `invariant_report` schema also constrains `severity` to `["critical", "warning", "info", null]`.

Automated enforcement: `tests/unit/test_schema_validation.py` validates every `schemas/*.schema.json` file for well-formedness and `additionalProperties` lockdown on every test run.

### Configuration Schemas

- `schemas/policy_config.schema.json` -- validates policy configuration (guard coefficients, fairness thresholds, backfill settings).
- `schemas/fidelity_gate_config.schema.json` -- validates fidelity gate threshold configuration.
- `schemas/reference_suite_config.schema.json` -- validates reference suite trace definitions.

### Evaluation Schemas

- `schemas/credibility_dossier.schema.json` -- validates credibility dossier structure.
- `schemas/sensitivity_report.schema.json` -- validates sensitivity sweep output.

### Config Validation

Module: `python/hpcopt/utils/config_validation.py`

`validate_config(path, schema_name)` loads a YAML config and validates against the matching JSON Schema from the `schemas/` directory. Returns `{"valid": bool, "errors": list[str]}` with JSON-path-annotated error messages. Gracefully degrades if `jsonschema` is not installed.

## 4. Artifact Families

### Data Artifacts

- canonical dataset parquet (`data/curated/*.parquet`)
- dataset metadata (`data/curated/*.metadata.json`)
- quality report (`outputs/reports/*_quality_report.json`)

### Modeling Artifacts

- quantile model files (`outputs/models/<model_id>/p10.joblib`, etc.)
- resource-fit model files (`fragmentation_classifier.joblib`, `node_size_regressor.joblib`)
- model metrics and metadata
- latest pointer (`outputs/models/runtime_latest.json`)
- model registry (`outputs/models/registry.jsonl`)
- drift reports (`*_drift_report.json`)
- tuning reports (`tuning_q*_report.json`)

### Simulation Artifacts

- scheduled jobs parquet (`*_jobs.parquet`)
- queue series parquet (`*_queue.parquet`)
- simulation report (`*_sim_report.json`)
- invariant report (`*_invariants.json`)

### Evaluation and Decision Artifacts

- fidelity reports (`*_fidelity_report.json`, candidate variants),
- recommendation reports (`*_recommendation_report.json`),
- recommendation manifests (`*_recommend_manifest.json`),
- sensitivity reports (`sensitivity_sweep_*.json`),
- feature importance reports (`feature_importance_*.json`),
- stress reports (`*_stress_report.json`) with degrade signatures,
- credibility dossiers (`dossier.json`, `dossier.md`),
- export bundles (`*_export.json`, `*_export.md`).

## 5. Batsim Normalization Contract

Normalized outputs from Batsim runs conform to the same downstream artifact schema as native simulation.

Normalization output includes:

- jobs artifact,
- queue artifact,
- simulation report with objective metrics,
- invariant report (post-hoc consistency checks),
- fallback accounting marked as external EDC source.

This enables uniform recommendation workflows regardless of backend.

## 6. Artifact Retention

Module: `python/hpcopt/artifacts/retention.py`

Command:
```bash
hpcopt artifacts cleanup --outputs-dir outputs --max-age-days 90
```

Scans for stale artifacts and optionally deletes them (default: dry run). Protected from cleanup:
- current production model directory (resolved from model registry),
- artifacts referenced by credibility dossier export files,
- model registry file itself.

Empty directories are cleaned up after file deletion.

## 7. Structured Logging

Module: `python/hpcopt/utils/logging.py`

All CLI commands emit structured JSON logs to stderr with:
- ISO-8601 UTC timestamp,
- log level,
- logger name,
- correlation ID (propagated via `contextvars`),
- optional extra fields,
- exception details when applicable.

Correlation IDs can be set explicitly or auto-generated for request tracing.

## 8. Validation and Test Coverage

Current tests validate (324 tests, 83% coverage with 82% CI gate):

- ingestion (SWF, Slurm, PBS) and schema-critical fields,
- trace profile sections,
- feature pipeline with chronological splits,
- runtime model training and monotonic quantiles,
- adapter contract behavior,
- cross-language parity for decisions (mandatory in CI),
- fidelity report generation,
- recommendation guardrails,
- Batsim config/run/normalization path,
- benchmark suite with regression detection,
- stress scenario generation and stress run CLI,
- reproducibility (deterministic replay, seed stability, fidelity determinism, feature pipeline determinism),
- credibility protocol integration,
- API endpoints (health, ready, predict, resource-fit, auth, validation),
- API load/concurrency behavior,
- manifest hash persistence,
- CLI commands across all 14 groups (ingest, train, simulate, pipeline, model, report),
- JSON schema validation (well-formedness and `additionalProperties` lockdown for all 10 schemas),
- file-based secrets loading (all 3 paths + missing file handling).

Coverage enforcement: `pytest-cov` runs in CI with `--cov-fail-under=58` on Python 3.11 and 3.12 matrix.

## 7. Known Reproducibility Caveat

If command execution occurs outside a Git context, `git_commit` can be `null` in manifests.  
This behavior is explicit and should be interpreted as environment limitation rather than silent omission.

## 8. Recommended Archival Unit

For each published experiment, archive:

- all generated report JSON files,
- corresponding manifests,
- simulation artifacts,
- model metadata and version pointers,
- fidelity and recommendation outputs,
- command lines used to generate results.

This archive should be sufficient for third-party replay under equivalent tooling.

