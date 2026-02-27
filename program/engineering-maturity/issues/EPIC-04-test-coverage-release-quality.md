## Epic ID
EPIC-04

## Owner role
qa-release

## Current status
in_progress

## Target outcome
- Coverage and release quality remain stable with no flaky gate regressions.
- Release manifests and checks remain reproducible.

## Deliverables
- Sustain line coverage at or above 86% across release window.
- Add flaky test tracking and burn-down workflow.
- Tie release checklist to evidence artifacts.

## KPI targets
- `line_coverage_pct`: 86.35 -> maintain `>= 86` by 2026-04-30
- `flaky_test_rate_pct`: target `<= 1`
- `release_checklist_completion_pct`: target `100%`

## Milestone linkage
- Milestone: `M0` Baseline Freeze
- Evidence: `coverage.xml`, `program/engineering-maturity/kpi-snapshots/2026-02-26.json`

## Risks and mitigations
- Risk: new modules lower coverage below floor during rapid delivery.
- Mitigation: add coverage diff check and PR-level test selection policy.
