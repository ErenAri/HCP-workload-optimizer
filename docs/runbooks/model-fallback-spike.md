# Model Fallback Spike Runbook

## Trigger

- `X-Fallback-Used=true` rate exceeds expected baseline envelope.

## Checks

1. Confirm model directory configured and readable.
2. Check model staleness and metadata integrity.
3. Confirm runtime predictor cache state.
4. Check for drift alerts or model promotion events.

## Mitigation

1. Restore last known-good model.
2. If unavailable, keep fallback and throttle risky policy rollout.
3. Trigger retraining pipeline if drift threshold breached.

## Exit Criteria

- Fallback rate returns to normal.
- Model health checks stable for 1 hour.

