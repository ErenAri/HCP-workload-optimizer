# Project Charter

## 1. Problem Statement

Shared HPC clusters are commonly governed by rule-based scheduling policies (for example, FIFO and heuristic backfilling).  
These policies are robust and understandable, but they do not adapt to workload uncertainty or changing user behavior.  
Operational consequences include:

- prolonged queue delays for interactive and short jobs,
- over-requested resources that inflate fragmentation,
- idle compute intervals despite non-empty queues,
- reduced throughput under heavy-tail runtime regimes,
- high dispute risk when policy claims are not contract-defined.

HPC Workload Optimizer addresses this gap by constructing a deterministic, policy-contract-driven control layer for simulation-time decision support.

## 2. Primary Objective

Design and validate an advisory scheduling engine that can:

- improve queueing outcomes under uncertainty,
- preserve fairness and starvation constraints,
- quantify fidelity against observed trace behavior,
- emit reproducible, auditable recommendation artifacts.

The project is intentionally systems-first: ML is treated as an uncertainty-aware input to policy decisions, not as autonomous authority.

## 3. Scientific and Engineering Hypotheses

H1. Quantile-aware backfilling reduces tail queueing cost (`p95 BSLD`) relative to static EASY baseline under congestion.

H2. Recommendation quality is only credible if the simulation core first passes fidelity checks against observed traces.

H3. Deterministic replay plus explicit invariants produces stronger industrial credibility than aggregate-only performance reports.

H4. Fallback transparency (`prediction_used`, `requested_fallback`, `actual_fallback`) is required to prevent inflated claims.

## 4. Scope

### In Scope (MVP)

- SWF/PWA trace ingestion and canonicalization.
- Deterministic policy replay for:
  - `FIFO_STRICT`
  - `EASY_BACKFILL_BASELINE`
  - `ML_BACKFILL_P50`
- Runtime quantile modeling (`p10`, `p50`, `p90`) with guard computation.
- Baseline and candidate fidelity reporting.
- Recommendation generation with hard-constraint gating.
- Reproducibility manifesting and artifact export.
- Batsim integration at config/execution/normalization boundary.

### Out of Scope (MVP)

- direct online integration with Slurm/PBS/LSF,
- autonomous policy actuation in production clusters,
- RL scheduling and online learning loops,
- GPU topology-aware optimization as a required feature,
- dashboard-first productization.

## 5. Credibility Criteria

The project is evaluated by systems engineering standards:

- deterministic replay under fixed inputs and seeds,
- explicit policy contract and transition semantics,
- executable invariant enforcement,
- fidelity gate pass/fail logic with distribution checks,
- reproducibility manifest lock and trace hash provenance,
- transparent failure modes and no-improvement narratives.

## 6. Canonical KPI Contract

Primary KPI:

- `p95 BSLD`, where `BSLD_i = (wait_i + runtime_i) / max(runtime_i, 60)`.

Secondary KPI:

- CPU utilization (`utilization_cpu`).

Hard constraints:

- `starved_rate <= 0.02`,
- `fairness_dev - fairness_dev_baseline <= 0.05`,
- `jain_baseline - jain <= 0.03`.

The recommendation engine must reject candidates that violate any hard constraint or fail fidelity gating.

## 7. Expected Deliverables

- canonical datasets and quality reports,
- simulation reports and invariant reports,
- fidelity reports (baseline and candidate),
- recommendation report with failure-mode analysis,
- run manifests with environment and config snapshots,
- exportable run bundles (`json`, `md`),
- documentation and protocol that can be independently replayed.

