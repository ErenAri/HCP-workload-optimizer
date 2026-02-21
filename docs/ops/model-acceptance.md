# Model Acceptance Policy

## Objective

Define minimum requirements before an ML scheduling policy recommendation is deployable.

## Required Inputs

- Baseline simulation report.
- Candidate simulation report.
- Fidelity report.
- Recommendation report.
- Drift report for current production model.

## Acceptance Rules

1. Fidelity status must be `pass`.
2. Recommendation status must be `accepted`.
3. Constraint contract must pass (fairness/starvation limits).
4. Fallback usage must remain within expected envelope for workload class.
5. No open high-severity reliability or security issues.

## Rejection Rules

- Any required report missing.
- Fidelity `fail`.
- Recommendation `blocked`.
- Unexplained degradation in primary KPI or fairness.

## Change Control

- Promotion requires peer review of all reports.
- Promotion decision recorded in release ticket with report artifact links.
- Rollback trigger documented before deployment.

