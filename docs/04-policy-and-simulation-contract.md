# Policy and Simulation Contract

## 1. Design Principle

The simulator is treated as the primary product artifact.  
ML output is consumed as policy input, but the scheduler contract, transition semantics, and invariants determine system legitimacy.

## 2. Supported Policies

Current policy set:

- `FIFO_STRICT`
- `EASY_BACKFILL_BASELINE`
- `ML_BACKFILL_P50`

These policies are implemented in:

- Python: `python/hpcopt/simulate/adapter.py`
- Rust parity binary: `rust/sim-runner/src/bin/adapter_contract.rs`

## 3. Deterministic Ordering Contract

Ordering constraints:

- queued jobs sorted by `(submit_ts, job_id)`,
- running jobs sorted by `(end_ts, job_id)`,
- equal timestamp event ordering:
  1. `job_complete`
  2. `job_submit`
  3. dispatch decision application (`job_start`).

Determinism is validated in unit tests, including cross-language decision parity.

## 4. Transition Model

Only the following transition classes may mutate scheduler state:

- `job_submit`
- `job_start`
- `job_complete`

At each simulation tick:

1. complete finished jobs and release resources,
2. enqueue arrivals at current timestamp,
3. compute dispatch decisions from policy adapter,
4. start feasible jobs and allocate resources,
5. execute invariant checks.

## 5. EASY Reservation Semantics

For `EASY_BACKFILL_BASELINE`:

- head-of-line (HoL) reservation time is computed from free resources and running job completions,
- backfill jobs are admissible only if they complete before or at HoL reservation time,
- HoL protection must not be violated by backfill.

## 6. ML Backfill Semantics

For `ML_BACKFILL_P50`:

- HoL reservation remains conservative,
- backfill gate uses uncertainty-aware runtime estimate:

```text
runtime_guard = p50 + runtime_guard_k * (p90 - p50)
```

- strict mode replaces guard by `p90` directly for backfill eligibility,
- if predictions are unavailable, fallback chain is:
  1. `runtime_requested_sec`
  2. `runtime_actual_sec`.

## 7. Invariants

Simulation invariants include:

- no job starts before submission,
- no negative free resources,
- resource conservation (`running + free == capacity`),
- no queue/running overlap for same job,
- no queued job in future relative to current clock.

Output structure:
- invariant report with step count and violations,
- strict mode aborts on first violation.

## 8. Adapter Contract Schemas

Snapshot schema:
- `schemas/adapter_snapshot.schema.json`

Decision schema:
- `schemas/adapter_decision.schema.json`

These schemas define boundary data contracts between orchestration and policy module.

## 9. Contract Validation

The project includes explicit tests for:

- missing required snapshot fields,
- equal-timestamp ordering,
- deterministic decision reproducibility,
- EASY reservation enforcement,
- ML strict uncertainty behavior,
- cross-language parity (Python vs Rust adapter decisions).

## 10. Batsim Boundary Integration

Batsim integration is currently boundary-scoped:

- config generation with explicit CLI args,
- execution wrapper (native or WSL),
- post-run normalization of Batsim CSV outputs into standard artifact contracts,
- candidate fidelity report emission against observed trace when available.

This ensures unified downstream evaluation even when simulation backend differs.

