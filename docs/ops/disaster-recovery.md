# Disaster Recovery Plan

## Scope

Covers restoration of:

- API service configuration,
- model artifacts,
- run manifests and reports,
- operational configs and dashboards.

## Targets

- RTO (recovery time objective): **4 hours**
- RPO (recovery point objective): **24 hours**

## Backup Plan

- Daily backup of:
  - `outputs/models/`
  - `outputs/reports/`
  - `configs/`
- Backup integrity verified daily.

## Restore Procedure

1. Provision clean staging namespace.
2. Restore latest backup snapshot.
3. Run validation:
   - `python scripts/production_readiness_gate.py --mode validate`
   - `powershell -ExecutionPolicy Bypass -File scripts/verify.ps1 -SkipCorrectness -SkipLoad`
4. Validate API status endpoints and key prediction path.

## Drill Cadence

- Quarterly DR drill.
- Record:
  - start/end timestamps,
  - actual RTO/RPO,
  - failures and remediations.

