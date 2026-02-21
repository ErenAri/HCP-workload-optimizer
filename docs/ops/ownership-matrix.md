# Ownership Matrix

## On-Call and Escalation

- Primary team: `platform-team`
- Secondary team: `ml-systems-team`
- Escalation target: `engineering-manager`

## Control Ownership

| Control | Primary Owner | Backup Owner | Review Cadence |
|---|---|---|---|
| Correctness gates (lint/type/tests) | Platform | ML Systems | Per PR |
| Benchmark regression | ML Systems | Platform | Weekly |
| API load SLO | Platform | SRE | Weekly |
| Fidelity and recommendation gate | ML Systems | Platform | Per release |
| Observability and runbooks | SRE | Platform | Monthly |
| Security controls | Security | Platform | Weekly |
| Deployment safety and rollback | SRE | Platform | Per release |
| Model ops (drift/retrain) | ML Systems | SRE | Weekly |
| Disaster recovery | SRE | Platform | Quarterly drill |
| API compatibility | Platform | ML Systems | Per PR / release |

## Release Approval Roles

- Technical signoff: Platform + ML Systems.
- Operational signoff: SRE.
- Security signoff: Security owner.
- Final go/no-go: Engineering manager.

