# Model Lifecycle Policy

## Stages

1. Train candidate model.
2. Evaluate with fidelity + recommendation gates.
3. Register model metadata and metrics.
4. Promote only after approval.
5. Monitor drift and fallback usage.
6. Retrain or rollback as needed.

## Promotion Requirements

- Candidate passes acceptance policy (`docs/ops/model-acceptance.md`).
- No active blocker incidents.
- Release ticket includes artifact links and signoffs.

## Drift and Retraining

- Run drift checks on schedule.
- Trigger retraining when PSI/loss thresholds are exceeded.
- Re-evaluate with the same fidelity/recommendation gates.

## Rollback Criteria

- Sustained SLO degradation linked to current model.
- Fallback spike beyond envelope.
- Constraint regressions detected in production telemetry.

