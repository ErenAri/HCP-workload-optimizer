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
- `schemas/adapter_decision.schema.json`

These enforce scheduler boundary compatibility.

## 4. Artifact Families

### Data Artifacts

- canonical dataset parquet (`data/curated/*.parquet`)
- dataset metadata (`data/curated/*.metadata.json`)
- quality report (`outputs/reports/*_quality_report.json`)

### Modeling Artifacts

- quantile model files (`outputs/models/<model_id>/p10.joblib`, etc.)
- model metrics and metadata
- latest pointer (`outputs/models/runtime_latest.json`)

### Simulation Artifacts

- scheduled jobs parquet (`*_jobs.parquet`)
- queue series parquet (`*_queue.parquet`)
- simulation report (`*_sim_report.json`)
- invariant report (`*_invariants.json`)

### Evaluation and Decision Artifacts

- fidelity reports (`*_fidelity_report.json`, candidate variants),
- recommendation reports (`*_recommendation_report.json`),
- recommendation manifests (`*_recommend_manifest.json`),
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

## 6. Validation and Test Coverage

Current tests validate:

- ingestion and schema-critical fields,
- trace profile sections,
- runtime model training and monotonic quantiles,
- adapter contract behavior,
- cross-language parity for decisions,
- fidelity report generation,
- recommendation guardrails,
- Batsim config/run/normalization path,
- manifest hash persistence.

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

