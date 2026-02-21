# Incident Response Runbook

## Trigger

Use this runbook when production availability, latency, or correctness is degraded.

## Steps

1. Declare incident and assign commander.
2. Freeze active deploys.
3. Collect current status:
   - `/health`, `/ready`, `/v1/system/status`
   - error rate / latency panels
   - recent deploy/model promotion history
4. Classify severity:
   - Sev1: major outage or severe SLO breach
   - Sev2: partial degradation
5. Mitigate:
   - rollback latest deploy if correlated
   - switch to safe policy/fallback if model issue
6. Communicate status every 15 minutes.
7. Close incident after stability window.
8. Open postmortem with timeline and action items.

