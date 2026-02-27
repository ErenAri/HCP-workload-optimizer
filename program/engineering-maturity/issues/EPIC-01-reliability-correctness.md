## Epic ID
EPIC-01

## Owner role
simulation-core

## Current status
in_progress

## Target outcome
- Zero Sev1 correctness defects in released builds.
- Deterministic replay remains 100% across supported policies/adapters.

## Deliverables
- Expand property-based invariant tests for scheduler transitions.
- Add release-blocking invariant report artifact in CI.
- Build regression harness for malformed trace edge cases.

## KPI targets
- `test_pass_rate`: 100.0% -> maintain `>= 99.5%` by 2026-04-30
- `escaped_defects_sev1`: 0 -> maintain `0`
- `deterministic_replay_pass_rate`: target `100%`

## Milestone linkage
- Milestone: `M0` Baseline Freeze
- Evidence: `program/engineering-maturity/kpi-snapshots/2026-02-26.json`

## Risks and mitigations
- Risk: invariant blind spots in rare event ordering cases.
- Mitigation: add property tests and event-order fuzz coverage per policy path.
