# Engineering Maturity Program (2026)

This program turns the project into a repeatable, auditable engineering standard for HPC scheduling optimization platforms.

## Program Goal

- Target score: `>= 90/100`
- Program window: `2026-03-01` to `2026-12-15`
- Minimum sustain period: `2 consecutive quarterly reviews`

## Scope

- Reliability and deterministic behavior under production load.
- Security, compliance, and incident readiness.
- Performance at realistic HPC scale.
- Model governance and measurable recommendation quality.
- Ecosystem integration with clear contracts and migration safety.

## Source of Truth

- Epics: `program/engineering-maturity/epics.yaml`
- Milestones and gates: `program/engineering-maturity/milestones.yaml`
- KPI dashboard schema: `schemas/engineering_kpi_dashboard.schema.json`
- Sample dashboard payload: `program/engineering-maturity/kpi-dashboard.sample.json`

## Operating Cadence

- Weekly: update epic status, risks, and blocked actions.
- Biweekly: publish KPI dashboard snapshot for the trailing 28-day window.
- Monthly: run gate review against milestone exit criteria.
- Quarterly: external architecture/reliability review and score recalibration.

## Definition of Done (90+)

- No open Sev1 incidents and no open high-severity vulnerabilities older than 14 days.
- Coverage remains `>= 86%` and all required CI checks are green for 30 consecutive days.
- SLOs are met for API latency and availability in two consecutive monthly windows.
- Benchmark suite shows no regression above configured budget for two releases.
- Reproducibility and fidelity gates pass on the reference suite for every release candidate.
