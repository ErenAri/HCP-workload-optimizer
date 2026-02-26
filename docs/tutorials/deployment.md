# Deployment Guide

## Docker

```bash
docker build -t hpcopt .
docker run -p 8080:8080 \
    -e HPCOPT_API_KEY=your-secure-key \
    -v $(pwd)/outputs:/app/outputs \
    hpcopt
```

## Kubernetes

Apply the included manifests:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/hpa.yaml
```

### ConfigMap

The `k8s/configmap.yaml` sets environment variables for the API:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hpcopt-config
data:
  HPCOPT_LOG_LEVEL: "INFO"
  HPCOPT_MODEL_DIR: "/app/outputs/models/latest"
```

### Horizontal Pod Autoscaler

The `k8s/hpa.yaml` scales based on CPU utilization:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HPCOPT_API_KEY` | (required) | API authentication key |
| `HPCOPT_LOG_LEVEL` | `INFO` | Logging level |
| `HPCOPT_MODEL_DIR` | `outputs/models` | Path to trained models |
| `HPCOPT_FEEDBACK_DIR` | `outputs/feedback` | Prediction feedback storage |

## Health Checks

```bash
# API health
curl http://localhost:8080/health

# Metrics (Prometheus)
curl http://localhost:8080/metrics
```

## Production Checklist

See [Production Readiness Checklist](../production-readiness-checklist.md) for the full list.

Key items:

- [ ] API key configured and rotated
- [ ] TLS enabled (reverse proxy)
- [ ] Model artifacts verified (hash check)
- [ ] Feedback loop connected
- [ ] Prometheus alerting on drift_detected
- [ ] Log aggregation configured
- [ ] Backup strategy for model artifacts
