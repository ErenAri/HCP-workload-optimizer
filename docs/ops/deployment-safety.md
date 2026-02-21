# Deployment Safety and Rollback

## Strategy

Production deploys use a staged rollout with rollback guardrails.

## Staged Rollout

1. Build image and run CI gates.
2. Deploy to staging.
3. Run verification (`scripts/verify.ps1`) on staging-like workload.
4. Promote to production only when checks pass.

## Canary Guidance

- Start at low traffic percentage.
- Observe:
  - 5xx ratio,
  - p95 latency,
  - fallback rate,
  - model loaded status.
- Pause or roll back on sustained threshold breach.

## Rollback Triggers

- Availability or latency SLO burn alerts.
- Regression in prediction correctness or policy constraints.
- Incident commander decision for unresolved degradation.

## Operational Evidence

- Staging verification workflow run logs.
- Rollback drill log and timestamp.
- Release ticket with go/no-go approvals.

