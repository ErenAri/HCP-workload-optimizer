# Distributed Tracing with OpenTelemetry

## Overview

HPCOpt includes optional OpenTelemetry instrumentation for distributed tracing of API requests. Tracing is enabled by installing the `[tracing]` extra and configuring an OTLP collector endpoint.

## Setup

### 1. Install the Tracing Extra

```bash
pip install hpc-workload-optimizer[tracing]
```

This adds:
- `opentelemetry-api`
- `opentelemetry-sdk`
- `opentelemetry-instrumentation-fastapi`

### 2. Configure Environment Variables

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=hpcopt
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1    # sample 10% of traces in prod
```

### 3. Deploy an OTLP Collector

See [`k8s/otel-collector.yaml`](../../k8s/otel-collector.yaml) for a K8s sidecar deployment, or run locally:

```bash
docker run -d -p 4317:4317 -p 4318:4318 \
  otel/opentelemetry-collector:latest \
  --config /etc/otel/collector.yaml
```

## Architecture

```
┌──────────────┐     gRPC (4317)     ┌──────────────────┐     ┌─────────┐
│  HPCOpt API  │ ──────────────────► │  OTLP Collector  │ ──► │  Jaeger │
│  (FastAPI)   │                     │  (sidecar/daemon) │     │  / Tempo│
└──────────────┘                     └──────────────────┘     └─────────┘
```

## What Gets Traced

- Every HTTP request (path, method, status, duration)
- Correlation ID propagation via `X-Trace-ID` / `X-Correlation-ID` headers
- Model loading and prediction latency
- Rate limiting decisions

## Sampling Strategy

| Environment | Sampler                      | Rate  |
|-------------|------------------------------|-------|
| `dev`       | `always_on`                  | 100%  |
| `staging`   | `parentbased_traceidratio`   | 50%   |
| `prod`      | `parentbased_traceidratio`   | 10%   |
