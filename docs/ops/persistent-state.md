# Persistent State & Model Registry

## Current Design

The model registry uses **JSONL files** with advisory file locking for state management:

```
outputs/registry/
├── runtime_registry.jsonl       # Runtime quantile model entries
├── resource_fit_registry.jsonl  # Resource fit model entries (future)
└── promotion_log.jsonl          # Audit trail of model promotions
```

### Write Safety

- **Single-writer assumption**: Advisory file locks (`fcntl.flock` on Linux, `msvcrt.locking` on Windows) prevent concurrent writes from the same host.
- **Append-only**: Registry entries are appended, never modified. Promotion/archival is a new entry with updated status.
- **Atomic reads**: Not guaranteed. A concurrent read during a write may see a partial line.

### Risks Under Multi-Instance

| Risk                     | Severity | Mitigation                        |
|--------------------------|----------|-----------------------------------|
| Concurrent writes        | High     | Advisory locks don't work across pods |
| Partial line reads       | Medium   | Retry + JSON parse validation     |
| Split-brain registry     | High     | Each pod sees its own copy        |

## Recommended Workaround (K8s)

For K8s deployments with multiple API replicas, use a **StatefulSet with `replicas: 1`** for the registry writer process:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hpcopt-registry-writer
spec:
  replicas: 1           # CRITICAL: single writer
  serviceName: hpcopt-registry
  template:
    spec:
      containers:
        - name: registry
          command: ["hpcopt", "serve", "--registry-only"]
          volumeMounts:
            - name: registry-data
              mountPath: /app/outputs/registry
  volumeClaimTemplates:
    - metadata:
        name: registry-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

## Migration Path: PostgreSQL Backend

For production deployments requiring horizontal writes and transactional safety:

1. **Schema**: Two tables — `model_entries` (immutable rows) and `model_status` (current promotion state).
2. **Locking**: PostgreSQL row-level locks (`SELECT ... FOR UPDATE`) replace advisory file locks.
3. **Migration**: Read existing JSONL entries → insert into PostgreSQL → update config to use SQL backend.
4. **Connection pooling**: Use PgBouncer or SQLAlchemy's built-in pool with `pool_size=5, max_overflow=10`.

### Alternative: etcd

For environments where PostgreSQL is not available:
- etcd provides strongly consistent key-value storage
- Use etcd leases for model lock management
- Suitable for small registry sizes (< 10K entries)
