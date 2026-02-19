# HPC Workload Optimizer MVP Design Plan

Date: 2026-02-19  
Status: Approved design baseline  
Scope: MVP

## 1) Locked decisions

- Primary stack: Python + Rust.
- Python responsibilities: ML training, feature engineering, inference API, recommendation logic.
- Rust responsibilities: SWF trace parsing, normalization, high-throughput simulation orchestration utilities.
- First dataset: PWA/SWF traces only.
- Scheduler target for MVP: simulation-only with Batsim.
- MVP interface: CLI required; REST API is optional and starts only after a clean simulation "win" report.
- Dashboard: deferred to post-MVP.

## 2) MVP objective

Deliver an advisory decision engine that proves measurable scheduling gains in simulation using real HPC traces.

The MVP must show:
- Improved cluster efficiency metrics vs baseline policies.
- Actionable policy recommendations with confidence.
- Reproducible simulation and evaluation pipeline.
- A single canonical headline KPI for decision quality.

## 3) Explicit non-goals for MVP

- No direct Slurm/PBS/LSF production integration.
- No auto-application of scheduling policies.
- No RL scheduler in MVP.
- No multi-dataset support beyond PWA/SWF in initial milestone.

## 4) Success criteria (MVP acceptance gates)

Gate 0: Baseline fidelity
- Baseline policies are defined by contract in `claudedocs/policy_spec_baselines_mvp.md`.
- Replay historical trace with baseline policy and compare to observed metrics.
- Fail fidelity if divergence is greater than 20% on any core metric or greater than 15% on two or more core metrics.
- Core metrics for fidelity: mean wait, p95 wait, throughput, and makespan.
- Throughput must use the evaluation-window definition from `claudedocs/policy_spec_baselines_mvp.md` section 1.1.
- Distribution fidelity metrics are also required:
- wait-time KL divergence <= 0.20,
- slowdown KS statistic <= 0.15,
- queue-length time-series correlation >= 0.85.
- Queue-series construction and sampling are defined in `claudedocs/policy_spec_baselines_mvp.md` section 2.6 and must be used for both observed and simulated traces.

Gate A: Data and reproducibility
- Parse SWF traces into canonical schema with validation reports.
- Replay baseline policies deterministically.

Gate B: Model quality
- Runtime model improves over naive baseline (median/mean historical runtime baseline).
- Quantile interval coverage is calibrated within predefined tolerance.

Gate C: System value
- At least one policy configuration improves the canonical primary objective score.
- Primary objective: p95 bounded slowdown (BSLD).
- Secondary objective: utilization.
- No fairness or starvation hard constraint violations in recommended policies.

Gate D: Product usability
- CLI supports end-to-end run from raw SWF to recommendation report.
- REST API is optional for MVP and begins only after Gate C is satisfied by a clean win report.

## 5) Architecture plan

```text
PWA/SWF files
  -> Rust parser/validator
  -> Trace profile layer
  -> Canonical parquet datasets
  -> Python feature pipeline
  -> Python model training + registry
  -> Prediction artifacts
  -> Batsim simulation runs
  -> Python recommendation engine
  -> CLI reports + REST API responses
```

Component boundaries:
- `rust/swf-parser`: zero-copy SWF parsing, schema validation, fast conversion to canonical format.
- `rust/sim-runner`: simulation orchestration, typed config generation, and scheduler decision-module adapter for Batsim runs.
- `python/profile`: trace profiling, workload regime characterization, and profile reports.
- `python/features`: time-aware feature generation and data splits.
- `python/models`: runtime/memory/fit models with quantile outputs.
- `python/recommend`: recommendation scoring and guardrail checks.
- `python/api`: FastAPI service exposing prediction and simulation endpoints.
- `python/cli`: Typer/Click command group orchestrating offline pipelines.

Ownership boundary (explicit):
- We own:
- policy decision logic, candidate selection structures, uncertainty guards, invariant checks, and recommendation logic.
- Batsim owns:
- event progression engine and core resource simulation runtime.
- Interface contract:
- Input to decision module: scheduler-visible state snapshot.
- Output from decision module: ordered dispatch decisions with reservation constraints.
- Determinism enforcement:
- deterministic ordering, invariant checks, and state-hash checks are enforced at our module boundary.

## 6) Canonical data model (MVP subset)

Core job table fields:
- `job_id`
- `submit_ts`
- `start_ts`
- `end_ts`
- `wait_sec`
- `runtime_actual_sec`
- `runtime_requested_sec`
- `requested_cpus`
- `requested_mem` (nullable)
- `requested_nodes_or_procs`
- `status`
- `user_id`
- `group_id`
- `queue_id`
- `partition_id`

Derived fields:
- `runtime_overrequest_ratio`
- `job_size_class`
- `queue_congestion_at_submit`
- `user_runtime_median_lookback`
- `user_runtime_var_lookback`

Rules:
- Time-aware features only.
- No future leakage.
- Null-safe parsing with explicit missing-value handling.
- Entire pipeline must run when `requested_mem` is missing for all rows.

## 7) Trace profile layer (systems interpretation)

Before training or simulation, each trace must produce a profile artifact containing:
- Job size distribution.
- Runtime heavy-tail analysis.
- Over-request distribution.
- Congestion regime analysis.
- User skew metrics.

Purpose:
- Explain when recommendations are likely to transfer or fail.
- Enable "no-improvement narratives" with causal evidence.

CLI entrypoint:
- `hpcopt profile trace --dataset <id>`

## 8) Model plan (Python)

Model set for MVP:
- Runtime prediction: LightGBM quantile models (`p10`, `p50`, `p90`).
- Optional memory proxy model if memory fields are available in selected traces.
- Resource over-request predictor: binary or regression on over-request ratio.

Training policy:
- Rolling, time-based train/validation/test split.
- Backtesting across multiple chronological folds.

Evaluation:
- Runtime: MAE, MAPE, pinball loss.
- Uncertainty: interval coverage and calibration error.
- Operational: effect on simulated utilization and wait time.

## 9) Batsim-first simulation plan

Simulation strategy:
- Use Batsim as simulation core for scheduling policy replay.
- Feed predicted runtime distributions from Python outputs.
- Compare baseline policy vs candidate policies under same arrival stream.
- Use exact baseline definitions from `claudedocs/policy_spec_baselines_mvp.md`.

Policy scenarios:
- Baseline FIFO strict.
- Baseline EASY backfill.
- Optional fair-share approximation scenario (if enabled in policy spec).
- ML-informed backfill with predicted runtime quantiles.
- Short-job class prioritization by predicted runtime threshold.
- `ML_BACKFILL_P50` uses uncertainty-adjusted backfill guard (`runtime_guard = p50 + runtime_guard_k * (p90 - p50)`, default `runtime_guard_k = 0.5`) with optional strict mode (`completion(p90) <= T_h`).
- Sensitivity tooling sweeps `runtime_guard_k in {0.0, 0.5, 1.0, 1.5}` for policy robustness analysis.

Critical rule:
- Do not build a full custom simulator in MVP.
- Rust `sim-runner` owns orchestration and policy decision adaptation, while Batsim remains the simulation core engine.

Simulator core requirement:
- Treat simulator behavior as the primary system artifact; ML is an input component, not the system core.

Explicit engine state machine:

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

Transition discipline:
- Queue/resource state changes may occur only through these transitions.
- Scheduling decisions execute after submit/complete events.
- Deterministic tie-breaking is mandatory for identical inputs.
- For equal timestamps, process `job_complete`, then `job_submit`, then dispatch `job_start`.

Scheduler invariants (MVP):
- No job starts before `submit_ts`.
- Resource allocations never exceed cluster capacity.
- A job cannot be simultaneously queued and running.
- Head-of-line protection is preserved under EASY variants.
- Replay is deterministic for identical inputs and seed.
- Successful recommendations do not violate starvation/fairness thresholds.

Executable invariant contract:
- Use a dedicated invariant-check module in simulation core.
- Every simulation step returns an `InvariantStatus` struct.
- Strict mode (`--strict-invariants`) fails run immediately on first invariant breach.

## 10) Objective contract (MVP)

Primary KPI:
- p95 bounded slowdown (BSLD), where per-job slowdown is:
- `BSLD_i = (wait_i + runtime_i) / max(runtime_i, tau_sec)`, with `tau_sec = 60`.

Secondary KPI:
- Utilization percentage.
- For SWF MVP traces, CPU utilization is required and GPU utilization must be reported as `N/A` (not omitted).

Hard constraints:
- Fairness deviation and starvation must stay within configured thresholds.
- Recommendations violating hard constraints are rejected.
- Constraint formulas and thresholds are normatively defined in `claudedocs/policy_spec_baselines_mvp.md` section 2.7.

Anti-cherry-picking rule:
- The recommendation engine ranks candidates by primary KPI first, then secondary KPI, subject to hard constraints.

Formal analysis score (for reporting and research comparison):
- `score = w1 * delta_p95_bsld + w2 * delta_utilization - w3 * fairness_penalty`
- with `w1 > w2` and `w3` configured to strongly penalize fairness violations.
- This score is analytic; recommendation acceptance remains lexicographic with hard constraints.

Reference trace suite (locked for MVP credibility runs):
- `CTC-SP2-1996-3.1-cln.swf` (or `.swf.gz`)
- `SDSC-SP2-1998-4.2-cln.swf` (or `.swf.gz`)
- `HPC2N-2002-2.2-cln.swf` (or `.swf.gz`)

## 11) Recommendation engine plan

Input:
- Simulation metrics by policy scenario.
- Fairness/starvation diagnostics.
- Prediction uncertainty metrics.

Output template:
- Recommended policy change.
- Expected impact deltas.
- Confidence score.
- Constraint check status.
- Rollback instruction.
- Fallback accounting (`prediction_used`, `requested_fallback`, `actual_fallback`) with percentages.
- Failure mode section: rejected policies, no-improvement traces, and workload patterns where ML degrades primary KPI.
- No-improvement narrative per trace when `ML_BACKFILL_P50` does not beat baseline.

Example:
- `Set short-job queue threshold to 20 minutes predicted p50; expected -12% p95 wait, +5% utilization, fairness delta +1.3%.`

## 12) CLI design (MVP interface)

Proposed commands:
- `hpcopt ingest swf --input <path> --out <dataset>`
- `hpcopt profile trace --dataset <id>`
- `hpcopt features build --dataset <dataset> --asof <date>`
- `hpcopt train runtime --dataset <dataset> --config <yaml>`
- `hpcopt simulate run --trace <dataset> --policy <yaml> --model <version>`
- `hpcopt stress gen --scenario <name> [--param value ...]`
- `hpcopt stress run --scenario <name> --policy <yaml> --model <version>`
- `hpcopt recommend generate --sim-results <path>`
- `hpcopt report export --run-id <id> --format json|md`

CLI requirements:
- Deterministic run IDs.
- YAML config support for reproducibility.
- Structured logs for pipeline observability.

## 13) REST API design (MVP interface, optional after first win)

Endpoints:
- `POST /v1/predict/runtime`
- `POST /v1/simulations`
- `GET /v1/simulations/{simulation_id}`
- `POST /v1/recommendations/generate`
- `GET /v1/recommendations/{run_id}`

API behavior:
- Async simulation submission with polling endpoint.
- Versioned model metadata in all prediction responses.
- Confidence and uncertainty fields are mandatory.
- API work starts only after the first accepted win report from the CLI pipeline.

## 14) Suggested repository layout

```text
repo/
  claudedocs/
  rust/
    swf-parser/
    sim-runner/
  python/
    hpcopt/
      api/
      cli/
      profile/
      features/
      models/
      recommend/
      simulate/
  configs/
    data/
    model/
    simulation/
  data/
    raw/
    curated/
  outputs/
    models/
    simulations/
    reports/
```

## 15) Delivery plan (8-week MVP)

Week 1-2:
- SWF parser in Rust.
- Canonical schema + validation pipeline.
- Initial CLI ingestion command.

Week 3-4:
- Trace profile layer and profiling reports.
- Python feature pipeline with time-aware backtesting.
- Runtime quantile model training pipeline.
- Baseline metrics report.

Week 5-6:
- Batsim orchestration via Rust `sim-runner`.
- Baseline vs candidate policy simulations.
- Metric aggregation and fairness/starvation checks.
- Explicit state-machine and invariant validation checks.

Week 7:
- Baseline fidelity gate execution and report.
- Recommendation engine and first clean win report generation.
- Failure mode report generation and review.
- No-improvement narratives for non-winning traces.

Week 8:
- Hardening, reproducibility tests, and acceptance gate review.
- MVP handoff package with runbook and sample outputs.
- Start REST API only if first clean win report is achieved with gate pass.
- Synthetic stress suite execution and report.

## 16) Performance and robustness benchmarks

Benchmark requirements (MVP-Core):
- SWF parse throughput (`jobs/sec`).
- Simulation throughput (`events/sec`).
- Peak memory footprint for parser and simulator stages.
- End-to-end runtime for canonical pipeline on reference trace.

Complexity ownership clarification:
- Complexity targets apply to the scheduler decision module we implement (including custom Batsim scheduler module logic and data structures), not to Batsim internal engine implementation details.

Reporting requirements:
- Report benchmark environment metadata (CPU, RAM, OS, tool versions).
- Report median and p95 over at least three runs.
- Include benchmark deltas in run manifest when code changes affect parser or simulator.
- Persist benchmark snapshots to a benchmark history ledger.
- Fail CI/build when simulation throughput regresses by more than 10% against the median of the last 5 accepted snapshots (same benchmark profile).

Synthetic stress suite (mandatory):
- Burst workload injection.
- Long-job cluster block scenario.
- Skewed user-distribution scenario.
- Generated via executable scenario commands (`hpcopt stress gen`, `hpcopt stress run`).

Stress-suite outcome:
- Document behavior under stress, including where policy degrades.
- Include pass/fail against starvation and fairness constraints.

## 17) Quality, testing, and operations

Testing strategy:
- Rust parser unit tests with malformed and edge-case SWF rows.
- Cross-language contract tests for canonical schema compatibility.
- Scheduler-adapter contract tests for state snapshot schema, equal-timestamp event ordering, deterministic tie-breaking, and EASY reservation enforcement.
- Python model regression tests with fixed seed datasets.
- Simulation reproducibility tests with same inputs and seed.
- Invariant checks on every simulation run.
- Fidelity gate checks as hard blockers for recommendation emission.

Operational standards:
- Each pipeline stage emits machine-readable artifacts.
- Every recommendation references exact model version and simulation run ID.
- Failure mode defaults to safe advisory-only behavior.
- All fallback paths are explicitly logged and summarized in final reports.
- Reproducibility manifest is locked per run and immutable after completion.
- Manifest must include:
- git commit hash,
- Rust crate versions,
- Python package lock hash,
- random seed(s),
- policy hash,
- full config snapshot.

## 18) Post-MVP roadmap (deferred by design)

Phase 2:
- Add Alibaba/Google trace adapters.
- Add dashboard and richer policy drilldowns.
- Add tighter uncertainty-aware risk gating.

Phase 3:
- Introduce Slurm integration in shadow mode.
- Add approval workflow and audit trail for operator decisions.
- Evaluate bounded automation for low-risk queues.

## 19) Systems-first research artifact

For research-grade systems presentation, use:
- `claudedocs/systems_first_research_appendix.md`

This appendix provides:
- Formal model summary and transition obligations.
- Complexity analysis targets.
- Reservation correctness proof sketch.
- Determinism proof argument.
- Counterexample scenario catalog.
- Publication-grade artifact checklist.
