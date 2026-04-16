# Policy and Simulation Contract

## 1. Design Principle

The simulator is treated as the primary product artifact.  
ML output is consumed as policy input, but the scheduler contract, transition semantics, and invariants determine system legitimacy.

## 2. Supported Policies

Current policy set:

- `FIFO_STRICT`
- `EASY_BACKFILL_BASELINE`
- `EASY_BACKFILL_TSAFRIR` (Tsafrir/Etsion/Feitelson, TPDS 2007 — user-history runtime estimates)
- `CONSERVATIVE_BACKFILL_BASELINE` (Mu'alem & Feitelson, TPDS 2001 — reservations for **all** queued jobs)
- `SJF_BACKFILL` — shortest-job-first ordering with EASY-style reservation/backfill
- `LJF_BACKFILL` — longest-job-first variant (research baseline)
- `FAIRSHARE_BACKFILL` — Slurm-style decayed-usage multifactor priority (default 7-day half-life)
- `ML_BACKFILL_P50`
- `ML_BACKFILL_P10`
- `RL_TRAINED` (MaskablePPO agent — RLScheduler-style env at `python/hpcopt/rl/env.py`, training entry point `scripts/train_rl_policy.py`; requires `pip install -e ".[rl]"`)

### 2.1 Conservative Backfill semantics

Conservative Backfill (CBF) differs from EASY in one key respect: **every**
queued job receives a future-start reservation at scheduling time, not just
the head-of-queue. A backfill candidate may run now only if doing so does
not push any of those reservations later. Reservations are committed
against a free-CPU **availability profile** — a list of `(time, free_after)`
events maintained in `python/hpcopt/simulate/availability_profile.py`.
This is strictly more predictable than EASY (each queued job's worst-case
start time is known at submission), at typically a small utilisation cost.

### 2.2 RL_TRAINED protocol

Patterned after Zhang, Dai, Bose, Li, Xu, Park, "RLScheduler" (SC'20):

* **Observation:** `Box(0, 1, shape=(MAX_QUEUE_SIZE=128, JOB_FEATURES=8), float32)`
  encoding the front 128 waiting jobs (requested cpus / capacity, runtime
  estimate normalised by 12h, wait so far, fits-now flag, queue position,
  free-cpus fraction, log10 runtime, validity flag). Padding rows are zero.
* **Action:** `Discrete(128)` — pick which queued job to dispatch next.
  Invalid slots (padding, or jobs that don't fit current free CPUs) are
  masked through `action_masks()` and combined with
  `sb3_contrib.MaskablePPO`.
* **Reward:** per-step, accumulated as `-bsld(job)` for each job that
  completes during the step's clock advance. Episode return ≈ `-mean_bsld`.
* **Policy network:** kernel-based — the same small MLP applied
  independently to each of the 128 slots produces one logit per slot;
  this is permutation-equivariant, the key trick from the RLScheduler
  paper (`KernelFeaturesExtractor` in `python/hpcopt/rl/features.py`).
* **Training defaults** (from the RLScheduler reference repo): clip 0.2,
  lr 3e-4, n_steps 4096, batch 256, γ=1.0, GAE λ=0.97, episodes of 256
  jobs sampled at random offsets in the trace.

When the simulator dispatches under `RL_TRAINED`, a loaded `RLPolicy` is
passed via `policy_context={"rl_policy": rl_policy}`. With no policy
loaded, the dispatcher falls back to FIFO so the policy is safe to
enumerate on systems without the `[rl]` extras.

### 2.3 Fairshare priority

`FAIRSHARE_BACKFILL` orders the queue by descending priority score, where
`score = -decayed_usage_cpu_seconds(user, t_submit)`. Decay follows
`usage(t) = usage(t0) · 0.5^((t-t0)/H)` with `H = 604800s` (7 days,
matching the Slurm `priority/multifactor` default). Priorities are
precomputed from the trace in `python/hpcopt/simulate/fairshare.py`,
respecting completion-time causality (only completions strictly before a
job's submit time contribute to that job's score).

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

