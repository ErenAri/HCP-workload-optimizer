# HPC Workload Optimizer Backlog (P0/P1/P2, Gate-Driven)

Date: 2026-02-19  
Applies to: `design_docs/mvp_design_plan_python_rust_batsim.md`, `design_docs/policy_spec_baselines_mvp.md`

## 1) Execution mode

- This backlog is gate-driven, not calendar-driven.
- Progress is accepted only by passing artifact gates and contract checks.
- Work priority order: `P0 -> P1 -> P2`.

## 2) Priority definition

- `P0`: required for MVP-Core credibility pack.
- `P1`: optional interface extension after clean P0 win.
- `P2`: deeper improvements and research hardening.

## 3) Complexity tiers

- `S`: localized change with low integration risk.
- `M`: multi-module change or moderate contract impact.
- `L`: cross-cutting implementation with high integration risk.

## 4) Gate sequence

Gate G0: Contract foundation
- Policy, schema, and invariant contracts exist and are internally consistent.

Gate G1: Deterministic simulation core
- Adapter, transition logic, and invariant enforcement run deterministically.

Gate G2: Fidelity and constraints
- Baseline fidelity passes (aggregate + distribution checks).
- Fairness/starvation constraints are enforceable from policy formulas.

Gate G3: Recommendation credibility pack
- Recommendations include fallback telemetry, failure modes, no-improvement narratives, stress behavior, and benchmark reports.

Gate G4: Optional API extension
- Starts only after G3 clean win report is accepted.

## 5) Reference trace suite (locked)

All MVP credibility runs must include this fixed PWA/SWF reference suite:
- `CTC-SP2-1996-3.1-cln.swf` (or `.swf.gz`)
- `SDSC-SP2-1998-4.2-cln.swf` (or `.swf.gz`)
- `HPC2N-2002-2.2-cln.swf` (or `.swf.gz`)

Each run manifest must record the exact trace file hash and suite member ID.

## 6) P0 backlog (must ship for MVP-Core)

| ID | Task | Depends on | Complexity | Blocking risk | Output / DoD |
|---|---|---|---|---|---|
| P0-01 | Write baseline policy spec contract | - | S | Baseline policy disputes | `design_docs/policy_spec_baselines_mvp.md` is normative and complete |
| P0-27 | Lock reference trace suite metadata and hashes | P0-01 | S | Non-comparable benchmark claims | Reference-suite IDs, sources, and file hashes are committed and manifest-checked |
| P0-02 | Define simulator state machine and invariant contract | P0-01 | S | Non-deterministic behavior | Formal transition and invariant contract exists |
| P0-03 | Define canonical schema v1 with nullable `requested_mem` | - | S | Schema mismatch across stages | Canonical schema with explicit nullability/fallback rules |
| P0-04 | Build Rust SWF parser crate (`rust/swf-parser`) | P0-03 | M | Ingestion instability | Typed parser for SWF 18 fields |
| P0-05 | Add SWF validation and quality report | P0-04 | M | Hidden data quality issues | Validation report with malformed/missing/fallback rates |
| P0-06 | Export canonical parquet from parser | P0-04 | S | Contract breaks into Python pipeline | Canonical parquet artifact with schema tags |
| P0-07 | Implement CLI ingest command (`hpcopt ingest swf`) | P0-05, P0-06 | S | Manual pipeline drift | Deterministic ingest command |
| P0-25 | Build trace profile layer + CLI (`hpcopt profile trace`) | P0-06 | M | Weak failure interpretation | Profile artifact with size/tail/over-request/congestion/user-skew metrics |
| P0-08 | Build Python feature pipeline (time-safe, null-safe) | P0-06 | L | Leakage and null handling errors | Feature pipeline works with all-memory-missing traces |
| P0-09 | Implement rolling time split + backtest harness | P0-08 | M | Invalid time splits | Chronological folds and split manifests |
| P0-10 | Train runtime quantile models (`p10/p50/p90`) | P0-09 | M | Weak uncertainty modeling | Model artifacts + training report |
| P0-11 | Implement naive/baseline runtime comparators | P0-09 | S | Inflated ML claims | Baseline lift report (median/mean/user-history) |
| P0-12 | Add Rust Batsim runner/config generator (`rust/sim-runner`) and scheduler decision-module adapter | P0-06, P0-01 | L | Ownership ambiguity at adapter boundary | Repeatable simulation orchestration with owned policy decision path |
| P0-26 | Implement scheduler-adapter contract tests | P0-12, P0-13, P0-15 | M | Adapter-contract mismatch | Tests assert state snapshot fields, equal-timestamp ordering, deterministic tie-breaking, EASY reservation enforcement |
| P0-13 | Implement baseline policy replay scenarios | P0-12, P0-01 | M | Baseline comparability risk | `FIFO_STRICT` and `EASY_BACKFILL_BASELINE` outputs |
| P0-14 | Implement ML policy replay scenarios | P0-10, P0-12, P0-01 | M | Unsafe ML dispatch behavior | `ML_BACKFILL_P50` outputs with uncertainty guard support |
| P0-15 | Implement transition and invariant validator in simulation loop | P0-02, P0-13 | M | Silent correctness regressions | Per-step invariant checks with strict mode failure |
| P0-16 | Build metric engine with objective contract | P0-13, P0-14 | M | Metric disputes | `p95 BSLD`, utilization, fairness/starvation formulas, weighted analysis score |
| P0-17 | Implement fallback telemetry accounting | P0-14, P0-16 | S | Silent fallback inflation | Prediction/requested/actual fallback percentages |
| P0-18 | Implement baseline fidelity gate | P0-13, P0-16 | M | Invalid simulator claims | Aggregate + distribution fidelity checks, queue-series contract enforced |
| P0-19 | Build recommendation engine with guardrails | P0-16, P0-18 | M | Unsafe policy recommendations | Recommendation blocking on constraints and fidelity |
| P0-20 | Implement failure mode report generator | P0-17, P0-19, P0-25 | M | No explanation of non-wins | Rejected policies, no-improvement traces, degradation narratives |
| P0-21 | Build performance benchmark suite | P0-07, P0-12, P0-14 | M | Hidden performance regressions | Parse/sim/pipeline benchmarks + snapshot ledger + regression gate |
| P0-22 | Add synthetic stress suite + report | P0-15, P0-19 | L | Replay-only overfitting | Burst/long-block/user-skew scenarios, executable via `hpcopt stress gen/run` |
| P0-23 | Report export and manifest lock (`json`, `md`) | P0-20, P0-21, P0-22 | S | Non-reproducible runs | Immutable manifest with commit/env/seed/policy/config hashes |
| P0-24 | Reproducibility test suite (fixed seeds) | P0-14, P0-23 | M | Irreproducible claims | Deterministic-run verification with tolerance bounds |

## 7) P1 backlog (optional after clean P0 win)

Unlock condition:
- Start only after G3 pass: fidelity pass + primary KPI win + hard-constraint compliance + credibility artifacts complete.

| ID | Task | Depends on | Complexity | Output / DoD |
|---|---|---|---|---|
| P1-01 | FastAPI skeleton with version contract | P0-23 | S | Running service with `/health` and build/version metadata |
| P1-02 | `POST /v1/predict/runtime` endpoint | P1-01 | S | Quantiles + model version + uncertainty in response |
| P1-03 | Simulation endpoints (`POST /v1/simulations`, `GET /v1/simulations/{id}`) | P1-01, P0-13 | M | Async submission and polling |
| P1-04 | Recommendation endpoints | P1-01, P0-19 | S | Recommendation retrieval by run ID |
| P1-05 | Structured logging + correlation IDs in API | P1-01 | S | Request-to-run traceability |
| P1-06 | CLI/API contract parity tests | P1-03, P1-04 | M | API outputs match CLI artifact contracts |
| P1-07 | API runbook and operational docs | P1-04 | S | Operator quickstart + troubleshooting |

## 8) P2 backlog (defer if rigor work remains)

| ID | Task | Depends on | Complexity | Output / DoD |
|---|---|---|---|---|
| P2-01 | Memory proxy model when trace fields are reliable | P0-10 | M | Optional memory predictions |
| P2-02 | Resource over-request classifier | P0-08 | M | Over-request probability feature |
| P2-03 | Multi-objective Pareto recommendation mode | P0-19 | M | Tradeoff-tagged recommendation set |
| P2-04 | Policy sensitivity/ablation tooling | P0-13 | M | Parameter effect-size reports including `runtime_guard_k` sweep (`0.0, 0.5, 1.0, 1.5`) |
| P2-05 | SWF profiling command for new traces | P0-06 | S | One-command trace health profiling |
| P2-06 | Artifact retention and cleanup tooling | P0-23 | S | Managed artifact footprint |
| P2-07 | API auth stub for future production integration | P1-01 | S | Auth middleware skeleton |
| P2-08 | CLI UX polish (dry-run, progress rendering) | P0-23 | S | Better operator ergonomics |
| P2-09 | Maintain systems research appendix package | P0-23 | M | Formal model/proof/counterexample appendix synced with implementation |

## 9) Critical path (gate-critical)

Critical chain:
- P0-01 -> P0-27 -> P0-02 -> P0-04 -> P0-06 -> P0-12 -> P0-13 -> P0-15 -> P0-26 -> P0-16 -> P0-18 -> P0-19 -> P0-23

Why this chain matters:
- It controls deterministic correctness, fidelity validity, and recommendation legitimacy.
- API work is intentionally outside this critical path.

## 10) Risk register with mitigations

Risk R1: SWF field quality inconsistency, especially memory columns  
Mitigation: P0-03 + P0-05 + P0-08.

Risk R2: Batsim integration overhead and boundary ambiguity  
Mitigation: P0-12 scoped adapter ownership + P0-26 adapter contract tests.

Risk R3: Baseline policy disputes invalidate comparisons  
Mitigation: P0-01 policy contract + P0-13 baseline replay outputs.

Risk R4: Simulator artifacts produce misleading gains  
Mitigation: P0-15 invariant validator + P0-18 fidelity gate.

Risk R5: Metric cherry-picking accusations  
Mitigation: P0-16 objective contract + weighted analysis score.

Risk R6: API scope steals value-proof focus  
Mitigation: P1 unlock gate only after clean G3 pass.

Risk R7: Silent fallback inflation overstates ML contribution  
Mitigation: P0-17 fallback telemetry + P0-23 artifact lock.

Risk R8: Performance regressions undermine systems credibility  
Mitigation: P0-21 benchmark ledger + regression gate.

Risk R9: Replay-only success fails under stress  
Mitigation: P0-22 executable stress suite.

Risk R10: Non-winning traces are not interpretable  
Mitigation: P0-25 trace profiling + P0-20 no-improvement narratives.

Risk R11: Reference results drift due to trace selection changes  
Mitigation: P0-27 locked reference-suite hashes and manifest checks.

## 11) Definition of done

MVP-Core done when all are true:
- All P0 items are complete.
- Baseline fidelity gate passes (aggregate + distribution + queue-series contract).
- At least one recommendation report improves primary KPI with hard constraints satisfied.
- Invariant strict mode and adapter contract tests pass.
- Benchmark report and regression history are present.
- Failure mode report and no-improvement narratives are present.
- Fallback telemetry is present for every ML policy run.
- Synthetic stress suite results are present with constraint pass/fail.
- Locked reference-suite traces and hashes are used for credibility runs.
- Reproducibility manifest lock includes commit hash, dependency versions, seeds, policy hash, and config snapshot.

MVP-Extended done:
- MVP-Core done, plus P1 endpoint contract parity complete.
