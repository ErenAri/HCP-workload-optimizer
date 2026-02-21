# Release Rollback Runbook

## Trigger

- New release breaches SLO or causes functional regression.

## Preconditions

- Previous image tag and rollout command available.
- On-call approver identified.

## Steps

1. Pause rollout progression.
2. Roll back deployment to previous stable image.
3. Validate:
   - `/health`, `/ready`, `/v1/system/status`
   - 5xx and p95 latency recovery
4. Record incident and rollback timestamp.
5. Block further promotion until root cause is resolved.

## Success Criteria

- Service restored within rollback SLO target.
- No recurring alert after 30-minute observation window.

