# HPC Workload Optimizer Documentation

This directory contains the formal documentation corpus for the project.  
The intent is to present the system as an engineering artifact, not only as an ML prototype.

## Reading Order

1. `docs/01-project-charter.md`  
   Problem framing, objectives, hypotheses, and non-goals.
2. `docs/02-system-architecture.md`  
   Layered architecture, component boundaries, and ownership model.
3. `docs/03-data-model-and-ingestion.md`  
   SWF/PWA dataset contract, canonical schema, and ingestion pipeline.
4. `docs/04-policy-and-simulation-contract.md`  
   Formal scheduler policies, transition rules, invariants, and adapter contract.
5. `docs/05-ml-runtime-modeling.md`  
   Runtime quantile modeling, uncertainty handling, and fallback semantics.
6. `docs/06-fidelity-objective-and-recommendation.md`  
   Fidelity gate, objective contract, and recommendation decision logic.
7. `docs/07-interfaces-cli-and-api.md`  
   CLI and API surface, including command-level and endpoint-level behavior.
8. `docs/08-reproducibility-and-artifacts.md`  
   Manifest contract, schema references, artifact lifecycle, and validation.
9. `docs/09-experiment-protocol-mvp.md`  
   End-to-end protocol for credible MVP experiments on the reference suite.
10. `docs/10-roadmap-and-open-problems.md`  
   Gate-driven roadmap and research extensions.
11. `docs/paper_outline.md`  
   Publication-oriented manuscript structure, contribution framing, and results checklist.
12. `docs/production-readiness-checklist.md`
    Release gating process and production checklist workflow.

## Operational Documentation

- `docs/ops/slo-and-error-budget.md`
- `docs/ops/ownership-matrix.md`
- `docs/ops/model-acceptance.md`
- `docs/ops/deployment-safety.md`
- `docs/ops/disaster-recovery.md`
- `docs/runbooks/incident-response.md`
- `docs/runbooks/api-latency-degradation.md`
- `docs/runbooks/high-5xx-rate.md`
- `docs/runbooks/model-fallback-spike.md`
- `docs/runbooks/release-rollback.md`
- `docs/security/secrets-management.md`
- `docs/security/vulnerability-management.md`
- `docs/security/access-control.md`
- `docs/mlops/model-lifecycle.md`
- `docs/api/versioning-and-deprecation.md`

## Relation to Existing Design Files

The `design_docs/` directory retains original planning artifacts and policy contracts:

- `design_docs/mvp_design_plan_python_rust_batsim.md`
- `design_docs/policy_spec_baselines_mvp.md`
- `design_docs/mvp_backlog_p0_p1_p2.md`
- `design_docs/systems_first_research_appendix.md`

These files remain authoritative for design history and contract evolution.  
The `docs/` corpus consolidates that material into implementation-aligned technical documentation.
