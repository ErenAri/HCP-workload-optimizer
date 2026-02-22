# Roadmap and Open Problems

## 1. Execution Strategy

Development is gate-driven:

- G0: contract foundation,
- G1: deterministic simulation core,
- G2: fidelity and constraint enforcement,
- G3: recommendation credibility pack,
- G4: API, observability, and deployment hardening.

This sequencing is designed to prevent interface work from obscuring core validity risks.

## 2. Current Maturity Snapshot

Completed:

- Multi-format ingestion pipeline (SWF, Slurm, PBS/Torque) with canonical parquet output.
- Shadow ingestion daemon for incremental scheduler polling.
- Trace profiling layer.
- Time-safe feature engineering pipeline with chronological cross-validation.
- Runtime quantile training, inference, and hyperparameter tuning.
- Resource-fit modeling (fragmentation classifier + node size regressor).
- Feature importance analysis via permutation importance.
- Model registry with register/promote/archive lifecycle.
- Drift detection (PSI + metric degradation).
- Deterministic simulation for baseline and ML policies.
- Invariant reporting and strict mode behavior.
- Baseline fidelity gate and candidate fidelity report support.
- Stress scenario generation (4 regimes) and automated stress testing.
- Recommendation engine with single-objective and Pareto multi-objective modes.
- Policy sensitivity sweeps over guard coefficient parameter space.
- Full credibility protocol with multi-trace suite orchestration and dossier assembly.
- Benchmark suite with regression gate and history tracking.
- Batsim config/run wrappers with normalization to standard artifacts.
- Manifest and export tooling.
- Adapter contract and cross-language parity tests.
- Production-ready API with file-based auth, Prometheus metrics, and structured logging.
- Docker containerization with pinned base image digests, secrets mount, and GitHub Actions CI/CD.
- JSON Schema validation for all configuration files with `additionalProperties: false` enforcement.
- Artifact retention management with production model protection.
- Modular CLI architecture (6 domain modules + assembler, down from 935-line monolith).
- Coverage enforcement infrastructure (`pytest-cov` with 58% gate in CI).
- CLI test coverage across all 14 command groups (89 total tests).
- Shared test fixtures (`conftest.py`) for API client, trace paths, and stress datasets.
- Rust CI governance: `cargo check` + `clippy --deny warnings` + release build in CI.
- Rust overflow safety: saturating arithmetic across simulation and adapter contract code.
- Rust release profile optimization (LTO, strip, single codegen unit).
- Mandatory cross-language adapter parity test in CI.
- Bandit SAST security scanning in CI.
- File-based API key management with 3-tier loading (file, Docker/K8s mount, legacy env).
- Schema hardening: all 10 schemas locked with `additionalProperties: false`, `policy_id` and `severity` enum constraints.
- Schema validation test suite (automated on every run).

## 3. Short-Horizon Priorities

Priority 1:
- execute full reference-suite experiments with consistent capacities and guard parameters across all ingested trace formats.

Priority 2:
- expand stress scenario coverage with topology-aware and GPU-mixed workloads.

Priority 3:
- tighten benchmark regression acceptance thresholds by workload class.
- enrich credibility dossier with cross-trace comparative visualizations.

## 4. Medium-Horizon Extensions

- richer resource-fit modeling (topology-aware placement features),
- memory proxy model when trace quality supports it,
- broader dataset adapters (Alibaba/Google traces, LSF),
- online model updating with automatic drift-triggered retraining,
- GPU efficiency predictor for heterogeneous clusters,
- alerting integration (PagerDuty/Slack) for drift detection and model staleness.

## 5. Long-Horizon Research Directions

- constrained reinforcement learning for policy adaptation,
- live shadow-mode validation against production scheduler decisions,
- multi-objective policy search including cost and energy dimensions,
- federated drift detection across multi-cluster deployments,
- causal inference for separating policy effect from workload regime change.

## 6. Open Technical Questions

1. Under which workload regimes does uncertainty-aware backfilling remain stable under fairness constraints?
2. How sensitive are recommendation outcomes to queue-series fidelity definitions and sampling cadence?
3. What minimum fidelity level is required before counterfactual gains can be externally trusted?
4. How should GPU- and memory-intensive workloads alter objective weighting and starvation caps?
5. Which contract checks should be promoted from offline diagnostics to runtime admission control in production?
6. What is the optimal drift detection cadence for balancing early detection against false positive rate?
7. How should the Pareto frontier change under different cluster topology constraints?

## 7. Publication Readiness Checklist

For a research-grade release:

- policy contracts finalized and versioned,
- reference-suite hash lock preserved,
- reproducibility manifests complete,
- fidelity and recommendation reports archived for each trace,
- credibility dossier assembled with cross-trace summary,
- failure cases documented alongside successful outcomes,
- stress testing results across all four scenario regimes,
- sensitivity analysis over guard coefficient parameter space,
- complexity and determinism claims bounded to owned modules.
