# High 5xx Rate Runbook

## Trigger

- 5xx error rate exceeds threshold for 5+ minutes.

## Checks

1. Identify endpoint(s) returning 5xx.
2. Inspect structured logs using `X-Trace-ID`.
3. Check upstream dependencies and storage availability.
4. Correlate with recent config/deploy/model changes.

## Mitigation

1. Roll back latest deploy if correlated.
2. Revert risky config changes.
3. Route to fallback policy if model path failing.
4. Increase replicas only if resource saturation is causal.

## Exit Criteria

- 5xx rate below threshold for 30 minutes.
- Root cause identified and tracked.

