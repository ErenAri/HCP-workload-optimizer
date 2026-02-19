# Roadmap and Open Problems

## 1. Execution Strategy

Development is gate-driven:

- G0: contract foundation,
- G1: deterministic simulation core,
- G2: fidelity and constraint enforcement,
- G3: recommendation credibility pack,
- G4: optional API expansion.

This sequencing is designed to prevent interface work from obscuring core validity risks.

## 2. Current Maturity Snapshot

Substantial completed elements:

- ingestion and canonicalization pipeline,
- trace profiling layer,
- runtime quantile training and inference,
- deterministic simulation for baseline and ML policy,
- invariant reporting and strict mode behavior,
- baseline fidelity gate and candidate fidelity report support,
- recommendation engine with guardrails and failure narratives,
- Batsim config/run wrappers with normalization to standard artifacts,
- manifest and export tooling,
- adapter contract and cross-language parity tests.

Partially implemented or deferred:

- full features pipeline (`hpcopt features build` scaffold),
- stress execution orchestrator (`hpcopt stress run` scaffold),
- production scheduler integration (shadow mode or online mode),
- GPU-aware resource and energy models.

## 3. Short-Horizon Priorities

Priority 1:
- execute full reference-suite experiments with consistent capacities and guard parameters.

Priority 2:
- produce an aggregate credibility dossier:
  - fidelity outcomes,
  - accepted/blocked recommendation counts,
  - fallback telemetry summary,
  - no-improvement narratives by workload regime.

Priority 3:
- convert performance benchmarking targets into automated regression checks.

## 4. Medium-Horizon Extensions

- richer resource-fit modeling (topology-aware placement features),
- memory proxy model when trace quality supports it,
- broader dataset adapters (Alibaba/Google traces),
- stronger stress protocol automation and reporting.

## 5. Long-Horizon Research Directions

- online model updating and drift detection,
- constrained reinforcement learning for policy adaptation,
- shadow-mode integration with production schedulers,
- multi-objective policy search including cost and energy dimensions.

## 6. Open Technical Questions

1. Under which workload regimes does uncertainty-aware backfilling remain stable under fairness constraints?
2. How sensitive are recommendation outcomes to queue-series fidelity definitions and sampling cadence?
3. What minimum fidelity level is required before counterfactual gains can be externally trusted?
4. How should GPU- and memory-intensive workloads alter objective weighting and starvation caps?
5. Which contract checks should be promoted from offline diagnostics to runtime admission control in production?

## 7. Publication Readiness Checklist

For a research-grade release:

- policy contracts finalized and versioned,
- reference-suite hash lock preserved,
- reproducibility manifests complete,
- fidelity and recommendation reports archived for each trace,
- failure cases documented alongside successful outcomes,
- complexity and determinism claims bounded to owned modules.

