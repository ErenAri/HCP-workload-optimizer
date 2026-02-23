# Horizontal Scaling Guide

## Kubernetes Resilience Features

HPCOpt ships with three resilience-focused K8s manifests for production hardening:

### PodDisruptionBudget (`k8s/pdb.yaml`)

- Ensures at least 1 pod is always available during voluntary disruptions (node drain, cluster upgrades).
- Protects against accidental eviction of all replicas.
- Spec: `minAvailable: 1`.

### NetworkPolicy (`k8s/network-policy.yaml`)

- Restricts ingress to only authorized namespaces:
  - `ingress-nginx` namespace (external traffic via ingress controller),
  - `monitoring` namespace (Prometheus scraping).
- Blocks pod-to-pod traffic from other namespaces.
- Ingress on TCP port 8080.

### Deployment Lifecycle (`k8s/deployment.yaml`)

- **preStop hook**: `sleep 5s` before SIGTERM to allow in-flight connection draining.
- **Readiness probe**: returns `503` when degraded (disk low, shutting down), removing the pod from Service endpoints.
- **Graceful shutdown**: drains in-flight requests over the termination grace period.

---

## Current Architecture

HPCOpt runs as a **single-instance** API server by default. Key constraints:

| Component          | Scaling Behavior                        |
|--------------------|-----------------------------------------|
| Rate limiter       | In-process (per-instance token bucket)  |
| Model cache        | In-process (loaded per instance)        |
| Model registry     | File-based JSONL with advisory locking  |
| API keys           | File-mounted (reads on every request)   |

## Multi-Instance Deployment

HPCOpt **can** be deployed with multiple replicas. Each replica:
- Loads its own copy of the ML model from the shared PVC
- Maintains its own rate-limiting buckets
- Reads API keys from the same mounted secret/file

### What Works

- **Read-only prediction endpoints** (`/v1/runtime/predict`, `/v1/resource-fit/predict`) scale horizontally without coordination.
- **Health/readiness probes** are stateless and work per-instance.
- **Prometheus metrics** are scraped per-pod via the ServiceMonitor.

### Known Limitations

1. **Rate limiting is per-instance**: With N replicas, effective rate limit is N × configured limit. For strict global rate limiting, use a Redis-backed limiter (see Future Path below).
2. **Model registry writes are single-writer**: The JSONL model registry uses advisory file locks. Only one writer should be active at a time. Use a K8s StatefulSet with `replicas: 1` for the registry writer, or migrate to PostgreSQL (see `persistent-state.md`).
3. **Log-level changes are per-instance**: The `POST /v1/admin/log-level` endpoint only affects the receiving pod.

## HPA Configuration

See [`k8s/hpa.yaml`](../../k8s/hpa.yaml) for the Horizontal Pod Autoscaler configuration:
- **Min replicas**: 2 (for availability)
- **Max replicas**: 8
- **Scale-up trigger**: CPU > 70% sustained for 60s
- **Scale-down**: Conservative (1 pod per 2 minutes, 5-minute stabilization)

## Future Path: Distributed Components

| Component           | Current                | Target                          |
|---------------------|------------------------|---------------------------------|
| Rate limiter        | In-process token bucket | Redis-backed sliding window    |
| Model cache         | Per-instance memory     | Shared Redis/NFS cache         |
| Model registry      | JSONL + file lock       | PostgreSQL with row-level lock |
| Session state       | None (stateless)        | Redis (if needed)              |
