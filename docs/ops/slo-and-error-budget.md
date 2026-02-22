# SLO and Error Budget

## Service Scope

This document defines production SLOs for the HPCOpt API service (`hpcopt.api.app`) running on Kubernetes.

## Availability SLO

- Target: **99.5%** monthly availability.
- Measurement window: rolling 30 days.
- Availability formula: successful requests / total requests.
- Successful request: HTTP status `<500`.

## Latency SLOs

- `GET /health`, `GET /ready`, `GET /v1/system/status`: p95 `< 300ms`
- `POST /v1/runtime/predict`, `POST /v1/resource-fit/predict`: p95 `< 1.5s`

## Error Budget

- Monthly availability budget: `0.5%` downtime/error.
- Policy when budget burn exceeds threshold:
  - freeze non-critical feature releases,
  - prioritize reliability bug fixes,
  - require rollback readiness for any deploy.

## Alerting Policy

- Fast burn alert: budget burn rate > 2x over 1h.
- Slow burn alert: budget burn rate > 1x over 6h.
- High-severity alert for:
  - sustained 5xx spike,
  - prediction latency p95 breach,
  - fallback spike beyond baseline envelope.

## Reporting

- Weekly reliability review:
  - SLO attainment,
  - top incidents,
  - corrective actions,
  - open risk items.

## Alerting Destinations

Production alerts are routed via Prometheus Alertmanager (see [`k8s/alertmanager-config.yaml`](../../k8s/alertmanager-config.yaml)):

| Severity   | Destination           | Repeat Interval |
|------------|-----------------------|-----------------|
| `critical` | PagerDuty on-call     | 1 hour          |
| `warning`  | Slack `#hpcopt-alerts` | 4 hours         |
| `info`     | Slack `#hpcopt-alerts` | 24 hours        |

### Setup

1. Create a PagerDuty service and obtain an integration key.
2. Create a Slack incoming webhook for `#hpcopt-alerts`.
3. Replace the placeholder values in `k8s/alertmanager-config.yaml`.
4. Apply: `kubectl apply -f k8s/alertmanager-config.yaml`
