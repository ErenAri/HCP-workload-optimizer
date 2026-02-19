# System Architecture

## 1. Architectural Thesis

The system is organized as a contract-oriented pipeline:

1. ingest real workload traces into canonical data,
2. learn uncertainty-aware runtime models,
3. replay scheduling policies in a deterministic simulator path,
4. gate claims through fidelity and constraints,
5. emit actionable recommendations with provenance.

The architecture is split by language for practical reasons:

- Python: control plane, ML, policy evaluation, reporting, CLI/API orchestration.
- Rust: parser and high-assurance scheduler decision utilities.

## 2. Layered View

```text
Raw SWF/PWA traces
  -> Ingestion + Canonicalization
  -> Trace Profiling
  -> Runtime Quantile Training
  -> Policy Simulation (core + Batsim path)
  -> Fidelity + Objective Evaluation
  -> Recommendation + Artifact Export
```

## 3. Repository Structure

```text
python/hpcopt/
  ingest/        # SWF parser and canonical parquet generation
  profile/       # trace profiling and workload characterization
  models/        # runtime quantile training/inference
  simulate/      # policy core, adapter, fidelity, Batsim wrappers
  recommend/     # recommendation ranking and guardrails
  artifacts/     # manifest and export utilities
  api/           # FastAPI service
  cli/           # Typer command surface

rust/
  swf-parser/    # fast SWF line parser/statistics utility
  sim-runner/    # deterministic runner and adapter contract binaries

configs/
  data/reference_suite.yaml
  simulation/fidelity_gate.yaml
  simulation/policy_ml_backfill.yaml

schemas/
  run_manifest.schema.json
  invariant_report.schema.json
  fidelity_report.schema.json
  adapter_snapshot.schema.json
  adapter_decision.schema.json
```

## 4. Core Boundaries and Ownership

### Owned by HPCOpt

- policy decision semantics,
- transition and invariant logic,
- fidelity computation and thresholds,
- objective contract and recommendation gating,
- artifact manifests and reproducibility controls.

### Owned by Batsim (when used)

- event engine internals,
- low-level simulation runtime execution,
- platform/workload simulation mechanics.

### Boundary Contract

Input:
- scheduler state snapshot (`schemas/adapter_snapshot.schema.json`).

Output:
- ordered dispatch decision set (`schemas/adapter_decision.schema.json`).

Determinism at boundary:
- queue ordering by `(submit_ts, job_id)`,
- equal timestamp ordering rule: `complete -> submit -> dispatch`.

## 5. Simulation State Model

Normative state vector:

```text
State = {
  clock_ts,
  running_jobs,
  queued_jobs,
  reserved_slots,
  event_queue,
  free_resources,
  completed_jobs,
  accounting_counters
}
```

Allowed transitions:

- `job_submit`
- `job_start`
- `job_complete`

No other transition is permitted to mutate queue or resource state.

## 6. Control and Data Planes

Control plane:
- `hpcopt` CLI,
- optional FastAPI endpoints,
- manifest generation and artifact export.

Data plane:
- canonical parquet traces,
- trained runtime quantile artifacts,
- simulation outputs (`jobs.parquet`, `queue.parquet`),
- evaluation reports (`sim_report`, `invariants`, `fidelity`, `recommendation`).

## 7. Execution Modes

### Native simulation mode

- uses Python simulation core in `python/hpcopt/simulate/core.py`,
- deterministic and policy-contract aligned,
- suitable for baseline/candidate studies and fidelity gate.

### Batsim-backed mode

- run-config generation via `hpcopt simulate batsim-config`,
- optional execution via `hpcopt simulate batsim-run`,
- post-run normalization into standard artifact contract,
- optional candidate fidelity emission against observed trace.

This mode preserves a single downstream report interface regardless of simulator backend.

## 8. Architectural Risks and Mitigations

Risk: policy ambiguity during comparison.  
Mitigation: formal policy contract (`design_docs/policy_spec_baselines_mvp.md`) and adapter schemas.

Risk: simulator artifacts mistaken for performance gains.  
Mitigation: baseline and candidate fidelity gating prior to recommendation acceptance.

Risk: over-attribution to ML.  
Mitigation: mandatory fallback accounting and strict recommendation guardrails.

Risk: reproducibility drift.  
Mitigation: immutable run manifests with hashes, config snapshots, tool versions, and seeds.

