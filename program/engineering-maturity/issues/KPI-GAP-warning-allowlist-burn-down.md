## Gap ID
KPI-GAP-WARN-ALLOWLIST

## Owner role
qa-release

## Current status
in_progress

## Problem statement
- CI now treats warnings as errors via pytest warning filters (`filterwarnings` includes terminal `error` rule).
- A temporary warning allowlist is still required for LightGBM feature-name warnings.
- This allowlist can mask future regressions if it is not actively burned down.

## Target outcome
- Remove temporary warning allowlist entries and keep test lanes warning-clean.
- Keep warning regressions release-blocking.

## Deliverables
- Isolate warning source in resource-fit prediction path and normalize feature-name handling.
- Remove `LGBMRegressor` and `LGBMClassifier` warning ignores from `pyproject.toml`.
- Add evidence in KPI snapshots that warning count remains zero without allowlist.

## KPI linkage
- `line_coverage_pct`: sustain `>= 86`
- `flaky_test_rate_pct`: sustain `<= 1`
- new local governance metric: `test_warning_count`: target `0`

## Risks and mitigations
- Risk: false confidence from broad warning ignore patterns.
- Mitigation: keep ignore patterns narrow and track explicit removal date in M1 gate review.
