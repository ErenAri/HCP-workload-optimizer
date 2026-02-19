# Fidelity, Objective Contract, and Recommendation Logic

## 1. Why Fidelity Precedes Optimization Claims

Counterfactual policy gains are only meaningful if the simulator reproduces core behavior of observed traces under baseline policies.  
Therefore fidelity is a gate, not a descriptive metric.

Implementation:
- `python/hpcopt/simulate/fidelity.py`
- config: `configs/simulation/fidelity_gate.yaml`

## 2. Baseline Fidelity Procedure

Command:

```bash
hpcopt simulate fidelity-gate --trace <canonical_trace.parquet>
```

Policy set evaluated against observed trace:

- `FIFO_STRICT`
- `EASY_BACKFILL_BASELINE`

Observed references are reconstructed from trace columns (`submit_ts`, `start_ts`, `end_ts`).

## 3. Fidelity Metrics

Aggregate divergence metrics:

- `mean_wait_sec`
- `p95_wait_sec`
- `throughput`
- `makespan_sec`

Distribution metrics:

- wait-time KL divergence,
- slowdown KS statistic,
- queue-length Pearson correlation (fixed cadence with right-continuous hold).

## 4. Gate Thresholds

Default thresholds:

- max single aggregate divergence: `0.20`
- max dual aggregate divergence count threshold: two or more metrics above `0.15`
- wait KL max: `0.20`
- slowdown KS max: `0.15`
- queue correlation min: `0.85`
- queue cadence: 60 seconds

If thresholds are violated, fidelity status is `fail`.

## 5. Objective Contract Metrics

Computation modules:

- `python/hpcopt/simulate/metrics.py`
- `python/hpcopt/simulate/objective.py`

Core objective values:

- `p95_bsld`
- `utilization_cpu`
- fairness metrics:
  - `fairness_dev`
  - `jain`
- starvation metrics:
  - `starved_rate`
  - `starved_jobs`

## 6. Constraint Evaluation

Candidate runs are evaluated against baseline with:

- `starved_rate <= 0.02`
- `fairness_dev_delta <= 0.05`
- `jain_delta <= 0.03`

Constraint violation blocks recommendation acceptance.

## 7. Weighted Analysis Score

For diagnostic ranking:

```text
score = w1 * delta_p95_bsld + w2 * delta_utilization - w3 * fairness_penalty
```

Default weights:
- `w1 = 1.0`
- `w2 = 0.3`
- `w3 = 2.0`

Note:
- acceptance still requires fidelity pass, hard constraints pass, and primary KPI improvement.

## 8. Recommendation Engine Workflow

Command:

```bash
hpcopt recommend generate \
  --baseline-report <baseline_sim_report.json> \
  --candidate-report <candidate_sim_report.json> \
  --fidelity-report <optional_fidelity_report.json>
```

Decision logic:

1. load baseline objective metrics,
2. evaluate each candidate score and constraints,
3. apply fidelity guardrail if provided,
4. require primary improvement (`delta_p95_bsld > 0`) for acceptance,
5. emit accepted recommendation or blocked outcome with reasons.

## 9. Failure-Mode Outputs

Recommendation report includes:

- rejection reasons per candidate,
- explicit failure-mode list,
- no-improvement narrative when no candidate is accepted.

This is required to avoid selective reporting of favorable runs only.

## 10. Batsim Candidate Fidelity

When Batsim run normalization is enabled, candidate fidelity can be emitted automatically:

```bash
hpcopt simulate batsim-run --config <run_config.json> --no-dry-run --use-wsl
```

If trace parquet is available in config, a candidate fidelity report is generated and attached to the run report.

