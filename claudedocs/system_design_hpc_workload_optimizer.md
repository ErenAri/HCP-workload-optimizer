# HPC Workload Optimizer: System Design

Date: 2026-02-19  
Version: v1.0 (advisory-first architecture)

## 1) Purpose

Build an AI decision layer that improves HPC cluster efficiency and reduces compute waste without replacing the scheduler.

The system must:
- Reduce queue wait time, especially p95 wait time.
- Increase effective GPU and CPU utilization.
- Reduce over-requested resources and fragmentation.
- Preserve fairness and policy compliance.
- Produce actionable, operator-safe recommendations.

## 2) Scope and non-goals

In scope:
- Runtime and resource usage prediction with uncertainty.
- Resource-fit prediction for node and GPU placement quality.
- Trace replay and policy simulation for counterfactual evaluation.
- Recommendation generation with quantified impact and confidence.
- Slurm-aligned integration in advisory mode first.

Out of scope in v1:
- Full scheduler replacement.
- Unbounded autonomous control that applies policy changes automatically.
- Dependence on private, hand-labeled datasets for core functionality.

## 3) Users and decisions

Primary users:
- HPC director.
- Scheduler administrator.
- Platform engineering lead.
- Infra cost owner.

Decisions supported:
- Queue policy tuning (backfill window, class split, fair-share weights).
- Right-sizing guidance by user group or workload class.
- Admission and prioritization recommendations.
- Capacity planning signals (idle pockets, fragmentation hotspots).

## 4) System context

The optimizer runs next to existing schedulers and exposes:
- Offline mode: historical replay and policy simulation.
- Online advisory mode: per-job predictions and policy suggestions.

Integration boundary:
- Scheduler remains source of truth for execution.
- Optimizer supplies scored recommendations and confidence intervals.
- Operators approve or reject changes via existing change control.

## 5) Architecture overview

```text
Trace Sources / Scheduler APIs
  -> Ingestion + Validation
  -> Canonical Event Store
  -> Feature Store (offline + online)
  -> Model Training + Registry
  -> Prediction Services
  -> Policy Simulator
  -> Recommendation Engine
  -> Operator API + Dashboard
  -> Scheduler Config Preview (advisory export)
```

Core services:
- `ingest-service`: parses SWF, Alibaba, Google, and scheduler-native logs.
- `feature-service`: builds time-safe features for training and inference.
- `model-trainer`: trains runtime, wait-time, and resource-fit models.
- `predict-api`: low-latency prediction endpoint for job submissions.
- `simulator`: deterministic replay engine for policy what-if analysis.
- `reco-engine`: translates metric deltas into concrete policy advice.
- `governance-service`: guardrails, approvals, and audit trails.

## 6) Data design

Canonical entities:
- `job_submission`: submit timestamp, user, group, queue, requested resources.
- `job_execution`: start/end timestamps, allocated resources, exit status.
- `job_usage`: observed cpu, memory, gpu utilization statistics.
- `cluster_state_snapshot`: node inventory, queue lengths, utilization, fragmentation.
- `policy_snapshot`: scheduler policy values active at event time.
- `prediction_record`: model version, point estimate, quantiles, confidence.
- `recommendation_record`: recommendation text, expected delta, confidence, constraints.

Canonical resource fields:
- `requested_cpus`, `requested_mem_mb`, `requested_gpus`.
- `allocated_cpus`, `allocated_mem_mb`, `allocated_gpus`.
- `runtime_requested_sec`, `runtime_actual_sec`.
- `wait_time_sec`.
- `gpu_efficiency` (utilization-adjusted work per allocated gpu-hour).

Data contracts:
- Event-time semantics only; ingestion-time never used as model feature.
- All IDs anonymized or hashed for non-production datasets.
- Schema versioning with backward compatibility for replays.

## 7) Feature engineering design

Feature families:
- User behavior: median runtime ratio, variance, over-request tendency.
- Workload shape: job size class, queue class, executable signature.
- Temporal context: hour-of-day, day-of-week, seasonal effects.
- Congestion context: queue depth, active jobs, free resource vectors.
- Topology context: node type inventory, gpu fragmentation score.
- Policy context: fair-share score, backfill parameters, queue weights.

Leakage controls:
- Strict time-aware feature windows.
- No use of future jobs in training examples.
- Rolling backtest framework with expanding training windows.

## 8) Model design

Model set:
- Runtime prediction: gradient boosting with quantile heads (p10, p50, p90).
- Memory prediction: quantile regression for peak memory.
- GPU efficiency prediction: regression by workload and placement context.
- Resource-fit prediction: ranking/classification over candidate placements.
- Queue wait-time prediction: survival model or gradient boosting with censoring-aware features.

Uncertainty requirements:
- Every prediction includes calibrated intervals.
- Calibration measured by interval coverage and pinball loss.
- Recommendations blocked when uncertainty exceeds policy thresholds.

Model governance:
- Model registry with semantic versions.
- Champion/challenger evaluation per cluster and queue segment.
- Automated drift checks on feature distribution and error distribution.
- Fallback baseline model for degraded or drifted states.

## 9) Simulation engine design

Purpose:
- Estimate counterfactual outcomes of policy choices before production changes.

Simulation inputs:
- Historical job trace.
- Cluster topology and resource constraints.
- Scheduler policy set (baseline and alternatives).
- Model predictions and uncertainty bands.

Simulation loop:
- Reconstruct event queue from historical submissions.
- Advance simulated time by scheduling and completion events.
- Apply policy logic to pending jobs at each decision point.
- Resolve placement feasibility with topology and fragmentation constraints.
- Track metrics continuously and at policy boundaries.

Policies compared:
- Baseline FIFO and fair-share.
- Baseline EASY-like backfill.
- ML-assisted backfill.
- Queue class split (short vs long).
- Resource right-sizing and fit-aware placement.

Validation requirement:
- Baseline replay must match historical aggregate metrics within predefined error bands before using counterfactual claims.

## 10) Decision and recommendation engine

Recommendation structure:
- Action: explicit parameter change or policy adjustment.
- Scope: queue, user group, workload class, or cluster partition.
- Expected impact: delta utilization, delta wait time, delta fairness.
- Confidence: probability recommendation is beneficial.
- Risk flags: fairness drift risk, starvation risk, uncertainty risk.
- Rollback hint: precise parameter reversion instructions.

Generation approach:
- Multi-objective scoring over simulation outputs.
- Constraint checks before emission.
- Rule templates for operator clarity and repeatability.

Example output:
- `Reduce GPU request for workload_class=small_train from 4 to 3 default cap; expected +6.8% GPU utilization, -11.2% p95 wait, fairness delta +1.1%.`

## 11) Objective function and guardrails

Primary optimization score:
- Weighted score over utilization, p95 wait time, throughput, and fairness deviation.

Hard constraints:
- No recommendation that violates fairness SLA limits.
- No recommendation that increases starvation beyond threshold.
- No recommendation if uncertainty interval overlaps neutral zone too widely.

Suggested initial constraints:
- Fairness deviation increase <= 5% relative to baseline.
- Starved jobs increase <= 2% absolute.
- p95 wait time must not regress in protected queues.

## 12) Slurm integration design

Integration modes:
- Mode 0: Offline only replay and recommendations.
- Mode 1: Shadow prediction service on job submit, no operator surface.
- Mode 2: Advisory dashboard and config preview export.
- Mode 3: Bounded automation on low-risk queues with auto-rollback.

Slurm touchpoints:
- Submission hook: capture request features and query `predict-api`.
- Completion hook: capture outcomes for retraining dataset.
- Scheduler telemetry pull: queue state, fair-share state, utilization state.
- Config preview export: generated `slurm.conf` and policy snippets for operator review.

Reliability design:
- Prediction timeout budget and default fallback behavior.
- Circuit breaker to baseline scheduler behavior on service degradation.
- End-to-end audit log of every prediction and recommendation consumed.

## 13) APIs

`POST /v1/predict/runtime`
- Input: job request + current cluster snapshot key.
- Output: p10/p50/p90 runtime, memory, gpu efficiency, confidence metadata.

`POST /v1/simulate/policy`
- Input: trace range, policy candidate, model versions.
- Output: metric deltas, confidence intervals, fairness and starvation diagnostics.

`GET /v1/recommendations`
- Input: cluster, queue, lookback window.
- Output: ranked actionable recommendations with expected impact.

`POST /v1/recommendations/{id}/decision`
- Input: approve, reject, defer, reason.
- Output: immutable audit event.

## 14) Observability and operations

SLOs:
- Prediction API availability >= 99.9%.
- P95 prediction latency <= 150 ms for online advisory path.
- Daily training pipeline success >= 99%.

Monitoring:
- Inference latency and error rate.
- Prediction interval coverage drift.
- Simulation reproducibility checks.
- Recommendation acceptance and realized impact tracking.

Incident responses:
- Automatic fallback to baseline policy advice on model drift alert.
- Retraining freeze when data quality checks fail.
- Recommendation suppression when confidence collapses.

## 15) Security, privacy, and compliance

Controls:
- Pseudonymization for user and project identifiers.
- Encryption at rest and in transit.
- Least-privilege service accounts per component.
- Data retention windows and secure deletion policies.

Sensitive data handling:
- Command lines and job names treated as sensitive fields.
- Hash or tokenize identifiers in analytics views.
- On-prem deployment option for regulated environments.

## 16) Evaluation framework

Offline model metrics:
- MAE and MAPE for runtime and memory.
- Pinball loss and interval coverage for uncertainty.
- Ranking quality for resource-fit model.

System-level metrics:
- Utilization percent by resource class.
- Average and p95 wait time.
- Throughput (jobs/day).
- Fairness deviation index.
- Starvation count and starvation duration.
- Estimated cost impact (gpu-hour and cpu-hour savings).

Promotion gates:
- Baseline simulator fidelity gate passed.
- System score improved with constraints satisfied.
- Recommendation confidence above minimum threshold.
- No critical regression in protected queues.

## 17) Rollout plan

Phase 1 (30 days):
- Build parser, canonical schema, feature pipeline, baseline models.
- Deliver deterministic replay and baseline fidelity report.
- Publish first advisory recommendations from offline simulation.

Phase 2 (90 days):
- Add quantile uncertainty and calibration monitoring.
- Add resource-fit and queue wait models.
- Deliver operator dashboard and recommendation audit workflow.

Phase 3 (production pilot):
- Integrate Slurm submission and completion hooks.
- Run shadow mode and compare realized outcomes.
- Enable bounded automation in one low-risk queue.

Phase 4 (advanced):
- Add GPU topology awareness and energy-cost objective.
- Add RL agent only after hard safety constraints and simulator fidelity are proven.

## 18) Definition of done for industrial quality

The system is industrial-grade when:
- It reproduces baseline scheduler behavior within agreed tolerance.
- It demonstrates consistent metric improvement across multiple traces and time windows.
- It emits actionable recommendations tied to exact scheduler parameters.
- It enforces fairness and starvation guardrails by design.
- It provides full auditability of predictions, recommendations, and operator decisions.
- It supports safe rollback and fails open to scheduler baseline behavior.
