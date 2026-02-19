# Data Model and Ingestion

## 1. Dataset Strategy

The MVP is intentionally constrained to SWF/PWA traces to maximize comparability with established scheduling literature.

Reference suite (locked in `configs/data/reference_suite.yaml`):

- `CTC-SP2-1996-3.1-cln.swf.gz`
- `SDSC-SP2-1998-4.2-cln.swf.gz`
- `HPC2N-2002-2.2-cln.swf.gz`

Each reference entry stores filename, source URL, and SHA-256 hash.

## 2. SWF Parsing Contract

Source parser:
- `python/hpcopt/ingest/swf.py` (canonical pipeline),
- `rust/swf-parser/src/main.rs` (high-speed statistics utility).

SWF format assumptions:
- 18-field standard workload line format,
- comments prefixed by `;` or `#`,
- blank and malformed lines tracked in quality report.

## 3. Canonical Job Schema (Current Implementation)

The ingestion pipeline emits parquet with at least:

- `job_id`
- `submit_ts`
- `start_ts`
- `end_ts`
- `wait_sec`
- `runtime_actual_sec`
- `runtime_requested_sec` (nullable)
- `allocated_cpus`
- `requested_cpus`
- `requested_mem` (nullable)
- `status`
- `user_id` (nullable)
- `group_id` (nullable)
- `queue_id` (nullable)
- `partition_id` (nullable)
- `runtime_overrequest_ratio` (nullable)

Derived timestamps:

- `start_ts = submit_ts + max(wait_time, 0)`
- `end_ts = start_ts + max(run_time, 0)`

Fallback rules:

- `requested_cpus` falls back to `allocated_cpus` when missing,
- `runtime_overrequest_ratio` computed only when requested and actual runtime are valid.

## 4. Quality Report Contract

Ingestion emits `*_quality_report.json` containing:

- line-level counters (`total_lines`, `comment_lines`, `blank_lines`, `malformed_lines`),
- parsed row count,
- null rates (for example, `requested_mem_null_rate`),
- fallback usage counts (`requested_cpu_fallback_rows`),
- dataset-level row count and paths.

This report is required for data quality transparency and downstream interpretation.

## 5. Dataset Metadata Contract

Ingestion emits `*.metadata.json` next to parquet with:

- dataset SHA-256,
- source trace filename and SHA-256,
- dataset ID and row count.

If the source file matches reference-suite definitions, the metadata includes suite membership details.

## 6. Reference Suite Locking

Command:

```bash
hpcopt data lock-reference-suite \
  --config configs/data/reference_suite.yaml \
  --raw-dir data/raw
```

Behavior:

- computes hashes for configured filenames found in `data/raw`,
- updates hash fields in the YAML file when missing or stale,
- emits lock report at `outputs/reports/reference_suite_lock_report.json`,
- optionally fails on missing files with `--strict-missing`.

## 7. Trace Profiling Layer

Command:

```bash
hpcopt profile trace --dataset <parquet_path>
```

Output report sections:

- job size distribution,
- runtime heavy-tail profile,
- over-request distribution,
- congestion regime (queue length),
- user skew metrics (`top_user_share`, HHI).

The profile layer is critical for interpretation of both wins and non-wins in policy evaluation.

## 8. Data Integrity Principles

- Canonical pipeline must remain null-safe when memory fields are absent.
- All derived metrics must avoid future leakage.
- Reference-suite provenance must be hash-verifiable.
- Every stage should emit machine-readable reports for independent audit.

