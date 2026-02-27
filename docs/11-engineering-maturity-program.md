# Engineering Maturity Program

This document operationalizes the roadmap into an execution system with clear ownership, gates, and score-based accountability.

## Program Objective

- Reach and sustain `>= 90/100` engineering score.
- Prove maturity with measurable evidence, not only feature count.
- Keep changes compatible with strict simulation, API, schema, and release contracts.

## Canonical Artifacts

- `program/engineering-maturity/README.md`
- `program/engineering-maturity/epics.yaml`
- `program/engineering-maturity/milestones.yaml`
- `program/engineering-maturity/kpi-dashboard.sample.json`
- `schemas/engineering_kpi_dashboard.schema.json`

## How to Use

1. Select the active milestone in `program/engineering-maturity/milestones.yaml`.
2. Track work through epic issues using the maturity issue templates.
3. Publish KPI snapshots in schema-compliant JSON every two weeks.
4. Run milestone gate review issue with explicit pass/fail evidence.
5. Update score trends and risks before each release candidate.

Current quality gate baseline (M1 hardening):
- tests run with warnings-as-errors policy (`[tool.pytest.ini_options].filterwarnings` ends with `error` and explicit allowlist entries),
- global coverage floor `>= 86%`,
- package coverage floors: `api >= 88%`, `models >= 89%`, `simulate >= 86%`,
- docs consistency and OpenAPI compatibility checks are CI-blocking.

## Scorecard Dimensions

- Reliability (20)
- Security (15)
- Performance (15)
- Quality (15)
- Reproducibility (10)
- Operability (10)
- Governance (10)
- Ecosystem (5)

## Required Governance Loop

- Weekly execution review with epic owners.
- Monthly gate review with milestone criteria.
- Quarterly independent review and remediation closure.

Without this loop, the score cannot be considered valid for maturity claims.
