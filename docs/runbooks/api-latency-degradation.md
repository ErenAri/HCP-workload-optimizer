# API Latency Degradation Runbook

## Trigger

- p95 latency breaches SLO for core prediction endpoints.

## Checks

1. Confirm degradation scope by endpoint.
2. Check CPU/memory saturation and queue depth.
3. Check model loading state and fallback rate (`api/model_cache.py` manages the predictor cache).
4. Check request volume spikes and rate limiting behavior (`api/rate_limit.py` manages token buckets).
5. Check for 504 Gateway Timeout responses -- the request timeout (default 30s, configurable via `HPCOPT_REQUEST_TIMEOUT_SEC`) may be too aggressive for the current load.

## Mitigation

1. Scale deployment replicas.
2. Enable temporary request shedding/rate reduction if needed (override via `HPCOPT_RATE_LIMIT` env var).
3. Increase request timeout if 504s are legitimate slow predictions (`HPCOPT_REQUEST_TIMEOUT_SEC`).
4. Roll back recent deploy if regression introduced.
5. Force fallback path if model inference is bottlenecked.

## Validation

- p95 returns below threshold for 30 minutes.
- 5xx rate returns to baseline.

