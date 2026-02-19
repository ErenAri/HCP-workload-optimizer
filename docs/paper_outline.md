# Paper Outline: HPC Workload Optimizer

## 0. Working Metadata

### Candidate title options

1. `Contract-Driven HPC Scheduling Optimization Under Uncertainty: A Systems-First Approach`
2. `From Heuristics to Auditable Decision Intelligence in HPC Queue Management`
3. `Deterministic Replay, Fidelity Gating, and Uncertainty-Aware Backfilling for HPC Workloads`

### Suggested paper type

- Systems engineering paper with empirical evaluation.
- Core contribution is a reproducible control and evaluation framework, not a novel deep-learning architecture.

### Candidate venues (style alignment)

- Systems/performance venues that value reproducibility and policy rigor.
- Industry-facing infrastructure workshops.

## 1. Abstract (Template)

1. Context: HPC clusters rely on static scheduling heuristics and inaccurate user runtime requests.
2. Problem: Policy comparisons are often non-reproducible, under-specified, and weakly validated against observed behavior.
3. Method: Present HPCOpt, a policy-contract-driven system with deterministic replay, executable invariants, uncertainty-aware ML backfilling, and fidelity gating.
4. Evaluation: Run on locked SWF/PWA reference traces with baseline and candidate policy comparisons under fairness/starvation constraints.
5. Results: Report gains and non-gains, including fallback usage and failure modes.
6. Conclusion: Systems-first evaluation reduces over-claiming and improves operational decision credibility.

## 2. Introduction

### 2.1 Motivation

- Queue delays, resource fragmentation, and over-allocation persist in shared clusters.
- Rule-based schedulers are robust but static.
- ML-only framing is insufficient without policy contracts and replay fidelity.

### 2.2 Gap

- Prior applied work often optimizes prediction error but not policy correctness under uncertainty.
- Many evaluations omit deterministic guarantees, constraint enforcement, and provenance-grade artifacts.

### 2.3 Thesis

Reliable scheduling optimization requires a contract-bound systems architecture where ML is advisory, fidelity is mandatory, and recommendations are constraint-gated.

### 2.4 Contributions (Claimed in paper)

1. A formal policy and transition contract for baseline and uncertainty-aware candidate policies.
2. Deterministic replay with executable invariants and adapter boundary schemas.
3. Fidelity gate combining aggregate divergence, distribution similarity, and queue-series correlation.
4. Recommendation engine with hard constraints, fallback accounting, and failure-mode narratives.
5. Reproducibility pack with manifest lock, trace hash provenance, and machine-readable artifacts.

## 3. Problem Formulation

### 3.1 System model

- Discrete-event queueing system over jobs with submission, start, and completion timestamps.
- Resource model in MVP: CPU capacity (`capacity_cpus`), with queue and running sets.

### 3.2 State and transitions

- State vector and transition classes (`job_submit`, `job_start`, `job_complete`).
- Deterministic equal-timestamp ordering: `complete -> submit -> dispatch`.

### 3.3 Objective contract

- Primary: `p95 BSLD`.
- Secondary: `utilization_cpu`.
- Hard constraints:
  - starvation rate bound,
  - fairness deviation bound,
  - Jain index degradation bound.

### 3.4 Decision validity

- Recommendations accepted only if fidelity passes, constraints pass, and primary KPI improves.

## 4. System Design

### 4.1 Architecture

- Ingestion and canonicalization.
- Trace profile layer.
- Runtime quantile modeling.
- Policy replay and Batsim-normalized replay path.
- Fidelity and recommendation layers.

### 4.2 Ownership boundary

- HPCOpt owns policy logic, contract enforcement, and recommendation logic.
- Batsim owns simulation runtime internals when external backend is used.
- Adapter schemas define I/O contract at policy boundary.

### 4.3 Policy suite

- `FIFO_STRICT`.
- `EASY_BACKFILL_BASELINE`.
- `ML_BACKFILL_P50` with uncertainty guard and strict mode.

### 4.4 ML containment principle

- Quantile predictions influence backfill eligibility.
- Explicit fallback chain and fallback-rate telemetry prevent over-attribution.

## 5. Implementation

### 5.1 Language stack

- Python: CLI/API, simulation core, fidelity, metrics, recommendation, artifacts.
- Rust: SWF parser utility and adapter contract parity binaries.

### 5.2 Interfaces

- CLI as primary control plane.
- API as auxiliary interface for runtime and resource-fit predictions.

### 5.3 Artifact contracts

- Simulation report.
- Invariant report.
- Fidelity report.
- Recommendation report.
- Immutable run manifest.

## 6. Experimental Methodology

### 6.1 Datasets

Locked reference suite:

- `CTC-SP2-1996-3.1-cln.swf.gz`
- `SDSC-SP2-1998-4.2-cln.swf.gz`
- `HPC2N-2002-2.2-cln.swf.gz`

All traces hash-locked via `configs/data/reference_suite.yaml`.

### 6.2 Data pipeline

1. ingest SWF to canonical parquet,
2. emit quality report and metadata,
3. build trace profile artifact.

### 6.3 Compared policies

- Baselines: FIFO and EASY.
- Candidate: ML backfill with guard parameter `runtime_guard_k`.

### 6.4 Evaluation protocol

1. baseline replay,
2. baseline fidelity gate,
3. candidate simulation,
4. candidate fidelity check,
5. recommendation generation under constraints.

### 6.5 Metrics

- Queue metrics: mean wait, p95 wait, throughput, makespan.
- Objective metrics: p95 BSLD, utilization, fairness/starvation.
- Fidelity metrics: divergence thresholds + KL/KS + queue correlation.
- Attribution metrics: fallback usage rates.

### 6.6 Sensitivity/ablation plan

- `runtime_guard_k` sweep: `{0.0, 0.5, 1.0, 1.5}`.
- Strict vs non-strict uncertainty mode.
- With-model vs fallback-only candidate behavior.

## 7. Results Section Blueprint

### 7.1 Fidelity first

- Report baseline fidelity pass/fail by trace and policy.
- State explicitly where optimization claims are blocked by fidelity.

### 7.2 Policy outcomes

- Show primary KPI deltas versus EASY baseline.
- Show utilization tradeoffs and constraint checks.

### 7.3 Attribution and robustness

- Report prediction_used vs fallback proportions.
- Include no-improvement narratives on traces without accepted gains.

### 7.4 Failure modes

- Enumerate rejection reasons (`fidelity_failed`, `constraint_violations`, `primary_kpi_not_improved`).

## 8. Performance Characterization (If included)

### 8.1 Reported dimensions

- parse throughput (jobs/sec),
- simulation throughput (events/sec),
- end-to-end runtime,
- memory footprint.

### 8.2 Regression discipline

- Track benchmark snapshots and report median/p95 over repeated runs.
- Separate claims for owned decision module from external simulator internals.

## 9. Threats to Validity

### 9.1 Internal validity

- Metric definition mismatch risk.
- Timestamp/order handling artifacts.
- Trace schema quality and missing memory fields.

### 9.2 External validity

- CPU-centric MVP may not generalize to heterogeneous GPU clusters.
- SWF-era traces may differ from modern AI-heavy workloads.

### 9.3 Construct validity

- Runtime prediction quality does not imply scheduling quality.
- Require policy-level outcomes, not only predictor metrics.

### 9.4 Reproducibility threats

- Environment drift when git metadata is unavailable.
- External simulator version differences.

## 10. Reproducibility and Artifact Appendix

### 10.1 Required artifact set

- run manifests,
- simulation and invariant reports,
- fidelity reports,
- recommendation report,
- exported run bundle.

### 10.2 Command log appendix

Include exact command sequence for each trace run, including config overrides.

### 10.3 Schema references

- `schemas/run_manifest.schema.json`
- `schemas/invariant_report.schema.json`
- `schemas/fidelity_report.schema.json`
- `schemas/adapter_snapshot.schema.json`
- `schemas/adapter_decision.schema.json`

## 11. Related Work (Structure)

1. Classic HPC scheduling heuristics (FIFO, EASY, fair-share methods).
2. ML-based runtime estimation and queue-time prediction.
3. Learning-guided scheduling under uncertainty.
4. Reproducibility and systems artifact evaluation practices.

Positioning statement:
- HPCOpt contributes in contract rigor, deterministic replay, and evaluation discipline at the policy layer.

## 12. Conclusion (Template)

Summarize:

- systems-first framing,
- fidelity-gated optimization claims,
- uncertainty-aware policy gains where present,
- explicit non-gains and failure regimes,
- roadmap to production integration and richer resource models.

## 13. Figures and Tables Checklist

### Figures

1. System architecture diagram.
2. Transition/state machine diagram.
3. Fidelity gate workflow.
4. KPI delta charts by trace and policy.
5. Fallback attribution chart.

### Tables

1. Policy contract summary table.
2. Dataset/reference suite table with hashes.
3. Fidelity threshold table.
4. Primary/secondary KPI outcomes by trace.
5. Constraint and rejection summary table.

## 14. Writing Guardrails

- Do not claim production integration in MVP results.
- Distinguish clearly between implemented functionality and planned extensions.
- Report blocked outcomes and non-improvements explicitly.
- Keep complexity claims scoped to modules owned by this project.

