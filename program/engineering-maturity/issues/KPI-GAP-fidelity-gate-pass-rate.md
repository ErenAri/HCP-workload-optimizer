## KPI metric
fidelity_gate_pass_rate_pct

## Current value
0.0

## Target value
>= 98.0

## Severity
critical

## Impact
Recommendation and promotion credibility is blocked because fidelity evidence currently fails in generated reports.

## Probable root cause
Fidelity configuration and/or calibration is not aligned with current generated trace behavior; baseline reports in `outputs/**/*fidelity*_report.json` are failing.

## Remediation plan
- Owner: ml-systems, due 2026-03-15: produce trace-by-trace failure breakdown and candidate threshold adjustments.
- Owner: simulation-core, due 2026-03-22: verify invariant and queue-series assumptions against failing reports.
- Owner: qa-release, due 2026-03-29: add KPI trend check in milestone review workflow.
