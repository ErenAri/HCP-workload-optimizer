## Epic ID
EPIC-03

## Owner role
security

## Current status
in_progress

## Target outcome
- No open high-severity vulnerabilities older than 14 days.
- Repeatable security evidence generated on every release path.

## Deliverables
- Export machine-readable security alert counts for KPI snapshots.
- Define and enforce release gate for unresolved high/critical findings.
- Document security escalation ownership in ops runbooks.

## KPI targets
- `open_high_sev_vulns`: 0 older than 14 days
- `secret_rotation_sla_days`: `<= 90`
- `security_gate_pass_rate`: `100%`

## Milestone linkage
- Milestone: `M0` Baseline Freeze
- Evidence: `program/engineering-maturity/kpi-snapshots/2026-02-26.json`

## Risks and mitigations
- Risk: GitHub security API availability/scope mismatch blocks automation.
- Mitigation: add fallback ingestion path from CI scan artifacts to snapshot pipeline.
