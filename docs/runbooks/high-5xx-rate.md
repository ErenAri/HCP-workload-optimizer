# High 5xx Rate Runbook

## Trigger

- 5xx error rate exceeds threshold for 5+ minutes.

## Checks

1. Identify endpoint(s) returning 5xx. Note: 504 responses indicate request timeout (`HPCOPT_REQUEST_TIMEOUT_SEC`, default 30s).
2. Inspect structured logs using `X-Trace-ID`.
3. Check upstream dependencies and storage availability.
4. Check model cache state -- if the predictor failed to load at startup, `model_cache.is_loaded()` will return `False` and all predictions use the fallback path.
5. Correlate with recent config/deploy/model changes.

## Mitigation

1. Roll back latest deploy if correlated.
2. Revert risky config changes.
3. Route to fallback policy if model path failing.
4. Increase replicas only if resource saturation is causal.

## Exit Criteria

- 5xx rate below threshold for 30 minutes.
- Root cause identified and tracked.

