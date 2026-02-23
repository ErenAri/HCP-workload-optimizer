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

## Model Cards

Starting with v1.2.0, model training automatically generates a `model_card.json` file alongside model artifacts.

### Model Card Contents

The model card documents:

- **Model details**: description, intended use, out-of-scope use, framework (scikit-learn GradientBoostingRegressor), features, target.
- **Dataset characteristics**: row counts (total/train/test), column count, feature statistics (min/max/mean/median/null%).
- **Performance metrics**: quantile metrics (p10/p50/p90), interval coverage, naive baselines, lift vs naive.
- **Fairness/bias evaluation**: per-group statistics (user_id, group_id), runtime mean by group, std across groups.
- **Known limitations**: accuracy caveats, workload stability assumptions, categorical feature generalization, 1-second floor.
- **Ethical considerations**: demographic encoding warning, fairness audit recommendations.

### Generation

Generated automatically at training time by `models/model_card.py:generate_model_card()`.

Output location: `outputs/models/{model_id}/model_card.json`.

The model card replaces or supplements manual model documentation and is part of the formal artifact trail.

