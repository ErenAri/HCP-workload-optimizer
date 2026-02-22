# Logging Operations Guide

## Log Levels by Environment

| Environment | Level   | Rationale                               |
|-------------|---------|---------------------------------------- |
| `dev`       | `DEBUG` | Full verbosity for local development    |
| `staging`   | `INFO`  | Moderate verbosity for integration      |
| `prod`      | `WARNING` | Minimal noise; errors + warnings only |

Set via `HPCOPT_LOG_LEVEL` env var or `configs/environments/<env>.yaml`.

The runtime log level can be changed dynamically via:

```bash
curl -X POST http://localhost:8080/v1/admin/log-level \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"level": "DEBUG"}'
```

## Structured Logging Format

All log messages are emitted as structured JSON with:
- `timestamp` (ISO 8601)
- `level`
- `trace_id` (correlation ID)
- `message`
- `module` / `function`

## Docker Log Rotation

Configure the Docker JSON file logging driver in `docker-compose.yaml` or daemon config:

```yaml
services:
  hpcopt:
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
```

Or system-wide in `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  }
}
```

## Kubernetes Log Management

In K8s, logs are typically managed by the node-level logging agent (e.g., Fluentd, Fluent Bit, or the default kubelet rotation).

**Default kubelet rotation** (usually sufficient):
- Logs stored in `/var/log/pods/`
- Rotated when container log file exceeds 10MB (configurable via `--container-log-max-size`)
- Retains 5 rotated files by default (`--container-log-max-files`)

**For centralized logging**, deploy a DaemonSet (Fluent Bit → Elasticsearch/Loki):

```yaml
# See k8s/ directory for ServiceMonitor which feeds Prometheus.
# For log aggregation, deploy Fluent Bit with this output config snippet:
# [OUTPUT]
#     Name  es
#     Match *
#     Host  elasticsearch.logging.svc.cluster.local
#     Port  9200
#     Index hpcopt-logs
```

## Volume Management

- **Model artifacts**: Mount as read-only PVC (`hpcopt-models` in `k8s/deployment.yaml`)
- **Temporary files**: Use `emptyDir` ephemeral volumes (cleaned on pod restart)
- **Disk health**: The `/health` endpoint checks free disk space (threshold: `HPCOPT_HEALTH_MIN_DISK_FREE_GB`)
