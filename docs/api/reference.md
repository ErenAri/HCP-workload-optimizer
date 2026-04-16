# HPC Workload Optimizer API

**OpenAPI version:** `3.1.0`  •  **Spec version:** `0.1.0`

_This page is auto-generated from `schemas/openapi_baseline.json` by_
_`scripts/generate_api_reference.py`. Do not edit by hand._

Systems-first API for runtime/resource-fit predictions and HPC advisory.

## Endpoints

### `/health`

#### `GET` — Health

Health check with model staleness, disk, memory, config validation.

**Responses:**

- **200** — Successful Response

### `/metrics`

#### `GET` — Metrics

Prometheus metrics endpoint.

**Responses:**

- **200** — Successful Response

### `/ready`

#### `GET` — Ready

Kubernetes readiness probe.

**Responses:**

- **200** — Successful Response

### `/v1/resource-fit/predict`

#### `POST` — Predict Resource Fit

**Request body:**

- `application/json`
  - schema ref: `ResourceFitRequest`

**Responses:**

- **200** — Successful Response
  - `application/json` -> schema ref: `ResourceFitResponse`
- **422** — Validation Error
  - `application/json` -> schema ref: `HTTPValidationError`

### `/v1/runtime/predict`

#### `POST` — Predict Runtime

**Request body:**

- `application/json`
  - schema ref: `RuntimePredictRequest`

**Responses:**

- **200** — Successful Response
  - `application/json` -> schema ref: `RuntimePredictResponse`
- **422** — Validation Error
  - `application/json` -> schema ref: `HTTPValidationError`

### `/v1/system/status`

#### `GET` — System Status

**Responses:**

- **200** — Successful Response

## Schemas

### `HTTPValidationError`

- **type:** `object`
- **title:** `HTTPValidationError`
- **properties:**
  - `detail`
    - **type:** `array`
    - **title:** `Detail`
    - **items:**
      - **$ref:** `ValidationError`

### `ResourceFitRequest`

- **type:** `object`
- **title:** `ResourceFitRequest`
- **properties:**
  - `candidate_node_cpus` *(required)*
    - **type:** `array`
    - **title:** `Candidate Node Cpus`
    - **items:**
      - **type:** `integer`
  - `queue_depth_jobs`
    - **title:** `Queue Depth Jobs`
  - `requested_cpus` *(required)*
    - **type:** `integer`
    - **title:** `Requested Cpus`
    - **minimum:** `1.0`

### `ResourceFitResponse`

- **type:** `object`
- **title:** `ResourceFitResponse`
- **properties:**
  - `fragmentation_risk` *(required)*
    - **type:** `string`
    - **title:** `Fragmentation Risk`
    - **enum:** `['low', 'medium', 'high']`
  - `notes` *(required)*
    - **type:** `array`
    - **title:** `Notes`
    - **items:**
      - **type:** `string`
  - `recommendation` *(required)*
    - **type:** `object`
    - **title:** `Recommendation`

### `RuntimePredictRequest`

- **type:** `object`
- **title:** `RuntimePredictRequest`
- **properties:**
  - `group_id`
    - **title:** `Group Id`
  - `partition_id`
    - **title:** `Partition Id`
  - `queue_depth_jobs`
    - **title:** `Queue Depth Jobs`
  - `queue_id`
    - **title:** `Queue Id`
  - `requested_cpus` *(required)*
    - **type:** `integer`
    - **title:** `Requested Cpus`
    - **minimum:** `1.0`
  - `requested_mem`
    - **title:** `Requested Mem`
  - `requested_runtime_sec`
    - **title:** `Requested Runtime Sec`
  - `runtime_guard_k`
    - **type:** `number`
    - **title:** `Runtime Guard K`
    - **default:** `0.5`
    - **minimum:** `0.0`
    - **maximum:** `2.0`
  - `user_id`
    - **description:** `User id when available`
    - **title:** `User Id`

### `RuntimePredictResponse`

- **type:** `object`
- **title:** `RuntimePredictResponse`
- **properties:**
  - `fallback_used` *(required)*
    - **type:** `boolean`
    - **title:** `Fallback Used`
  - `notes` *(required)*
    - **type:** `array`
    - **title:** `Notes`
    - **items:**
      - **type:** `string`
  - `predictor_version` *(required)*
    - **type:** `string`
    - **title:** `Predictor Version`
  - `runtime_guard_sec` *(required)*
    - **type:** `integer`
    - **title:** `Runtime Guard Sec`
  - `runtime_p50_sec` *(required)*
    - **type:** `integer`
    - **title:** `Runtime P50 Sec`
  - `runtime_p90_sec` *(required)*
    - **type:** `integer`
    - **title:** `Runtime P90 Sec`

### `ValidationError`

- **type:** `object`
- **title:** `ValidationError`
- **properties:**
  - `ctx`
    - **type:** `object`
    - **title:** `Context`
  - `input`
    - **title:** `Input`
  - `loc` *(required)*
    - **type:** `array`
    - **title:** `Location`
    - **items:**
  - `msg` *(required)*
    - **type:** `string`
    - **title:** `Message`
  - `type` *(required)*
    - **type:** `string`
    - **title:** `Error Type`
