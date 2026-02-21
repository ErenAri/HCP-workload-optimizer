# API Latency Degradation Runbook

## Trigger

- p95 latency breaches SLO for core prediction endpoints.

## Checks

1. Confirm degradation scope by endpoint.
2. Check CPU/memory saturation and queue depth.
3. Check model loading state and fallback rate.
4. Check request volume spikes and rate limiting behavior.

## Mitigation

1. Scale deployment replicas.
2. Enable temporary request shedding/rate reduction if needed.
3. Roll back recent deploy if regression introduced.
4. Force fallback path if model inference is bottlenecked.

## Validation

- p95 returns below threshold for 30 minutes.
- 5xx rate returns to baseline.

