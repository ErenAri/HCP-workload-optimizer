# MVP Policy Spec Contract (Batsim)

Date: 2026-02-19  
Status: Evaluation contract for simulation comparability

## 1) Purpose

Define exact scheduling policy behavior for MVP experiments so results are reproducible and not disputable.

This contract governs:
- Baseline policies used in replay and comparison.
- Optional fair-share approximation behavior.
- ML-assisted candidate policy behavior.
- Baseline fidelity gate pass/fail logic.

## 1.1) Definitions

Core time/resource definitions:
- `wait_i = start_ts_i - submit_ts_i`.
- `runtime_i = end_ts_i - start_ts_i`.
- `slowdown_i = (wait_i + runtime_i) / max(runtime_i, tau_sec)`.
- `BSLD_i = slowdown_i` with `tau_sec = 60`.
- `makespan = max(end_ts) - min(submit_ts)`.
- Evaluation window `W = [window_start_ts, window_end_ts)`.
- Offline replay default: `window_start_ts = min(submit_ts)` and `window_end_ts = max(end_ts)`.
- `evaluation_duration = window_end_ts - window_start_ts`.
- `completed_jobs_in_window = count(job_i where end_ts_i in W)`.
- `throughput = completed_jobs_in_window / evaluation_duration`.
- Unless explicitly configured, jobs outside `W` are excluded from throughput numerator and denominator construction.

Queue definitions:
- `queue_len_jobs(t) = |Q(t)|`.
- `queue_len_cpu_demand(t) = sum(requested_cpus_j for j in Q(t))`.

Fairness/starvation definitions:
- `starved_i = 1(wait_i > starvation_wait_cap_sec)`.
- `starved_rate = sum(starved_i) / total_jobs`.
- `share_u = cpu_sec_u / sum(cpu_sec_k)`.
- `fairness_dev = 0.5 * sum_u |share_u - target_u|`.
- `jain = (sum_u share_u)^2 / (N_active_users * sum_u share_u^2)`.

Evaluation windows:
- Offline replay default: whole evaluated trace window (`W` above).
- Online/rolling diagnostics default: trailing `rolling_window_sec = 604800` (7 days).
- A run must record its selected window mode in the manifest.

## 2) Global simulation model

- Scheduler decisions are event-driven (at job submit and job completion events).
- Resource model for SWF MVP is processor-count only (no GPU/topology assumptions in baseline contract).
- Job execution duration in simulation uses `runtime_actual_sec` as ground truth.
- Runtime estimates are used only for reservation and backfill decisions.
- Deterministic tie-breaking is always: `submit_ts ASC`, then `job_id ASC`.
- If `runtime_requested_sec` is missing or non-positive, fallback estimate is `runtime_actual_sec`.

### 2.1 Engine state vector

State is explicitly modeled as:

```text
State = {
  clock_ts,
  running_jobs,
  queued_jobs,
  reserved_slots,
  event_queue,
  free_resources,
  completed_jobs,
  accounting_counters
}
```

### 2.2 Allowed state transitions

Only these transitions are allowed:
- `job_submit`: append job to `queued_jobs`, update `event_queue`.
- `job_start`: remove job from `queued_jobs`, allocate resources, insert into `running_jobs`.
- `job_complete`: remove job from `running_jobs`, release resources, append to `completed_jobs`.

Scheduling decisions occur immediately after `job_submit` and `job_complete` events.  
No other transition type may mutate queue or resource state.

For equal timestamps, event processing order is:
1. `job_complete`
2. `job_submit`
3. Scheduler dispatch decisions resulting in `job_start`

### 2.3 Formal transition specification

Each transition below is normative and must be implemented with explicit precondition and postcondition checks.

`job_submit(job)`
- Preconditions:
- `job.job_id` is not present in `queued_jobs`, `running_jobs`, or `completed_jobs`.
- `job.submit_ts >= clock_ts` or event is scheduled in `event_queue` for future processing.
- Required fields (`submit_ts`, `requested_cpus`) are parse-valid.
- Postconditions:
- `job` exists in `queued_jobs`.
- `event_queue` no longer contains an unconsumed duplicate submit event for `job`.
- `accounting_counters.submitted += 1`.
- State delta:
- `queued_jobs := queued_jobs U {job}`.
- `event_queue := event_queue - {current submit event}`.

`job_start(job, alloc)`
- Preconditions:
- `job` exists in `queued_jobs`.
- `job.submit_ts <= clock_ts`.
- `alloc.cpus <= free_resources.cpus`.
- Policy-specific reservation constraints are satisfied (including EASY head-of-line protection).
- Postconditions:
- `job` does not exist in `queued_jobs`.
- `job` exists in `running_jobs` with start timestamp `clock_ts`.
- `free_resources.cpus` decreased by `alloc.cpus`.
- `event_queue` contains completion event for `job` at `clock_ts + runtime_used`.
- State delta:
- `queued_jobs := queued_jobs - {job}`.
- `running_jobs := running_jobs U {job(start_ts=clock_ts, alloc)}`.
- `free_resources.cpus := free_resources.cpus - alloc.cpus`.
- `event_queue := event_queue U {job_complete(job) at completion_ts}`.

`job_complete(job)`
- Preconditions:
- `job` exists in `running_jobs`.
- Completion timestamp equals `clock_ts` for consumed completion event.
- Postconditions:
- `job` does not exist in `running_jobs`.
- `job` exists in `completed_jobs`.
- `free_resources.cpus` increased by job allocation.
- `accounting_counters.completed += 1`.
- State delta:
- `running_jobs := running_jobs - {job}`.
- `completed_jobs := completed_jobs U {job(end_ts=clock_ts)}`.
- `free_resources.cpus := free_resources.cpus + job.alloc.cpus`.
- `event_queue := event_queue - {current completion event}`.

### 2.4 Scheduler Invariants (MVP)

- No job starts before its `submit_ts`.
- Total allocated resources never exceed cluster capacity.
- A job cannot exist in both `queued_jobs` and `running_jobs`.
- Simulation clock is monotonic non-decreasing.
- Deterministic ordering is preserved for identical inputs and seed.
- Under EASY policies, head-of-line reservation protection is never violated.
- Recommendation runs marked successful must not breach configured starvation and fairness thresholds.

### 2.5 Invariant execution interface

Each processed event step must emit an invariant status record:

```text
InvariantStatus = {
  step_index,
  event_type,
  clock_ts,
  passed,
  failed_invariants[],
  severity,
  state_hash
}
```

Strict mode behavior:
- In strict mode, `passed = false` must terminate the run immediately.
- Non-strict mode may continue for diagnostics, but run cannot be marked successful.

### 2.6 Queue-length series definition (fidelity contract)

Observed queue series and simulated queue series must be constructed using identical rules.

Queue definitions:
- `queue_len_jobs(t) = |Q(t)|`.
- `queue_len_cpu_demand(t) = sum(requested_cpus_j for j in Q(t))`.

Sampling rules:
- Primary fidelity sampling is event-driven at every event boundary.
- At each timestamp, apply transition ordering first (`job_complete -> job_submit -> job_start`) and then sample queue state.
- Optional fixed-cadence output uses 60-second buckets with right-continuous hold interpolation.

Correlation computation contract:
- Queue-length correlation metric for gate decisions is computed on fixed 60-second cadence only.
- Series alignment window is `[min(submit_ts), max(end_ts)]` across observed and simulated runs.
- Interpolation is right-continuous hold.
- Both aligned series are z-score normalized before Pearson correlation.
- If either series has zero variance:
- correlation is `1.0` when both normalized series are identical constants,
- otherwise correlation is `0.0`.

Fidelity default:
- Queue-length correlation uses `queue_len_jobs(t)` unless explicitly configured otherwise.

### 2.7 Fairness and starvation constraint definitions

Starvation:
- Per-job starvation indicator: `starved_i = 1(wait_i > starvation_wait_cap_sec)`.
- Default cap: `starvation_wait_cap_sec = 172800` (48 hours).
- Cluster starvation rate: `starved_rate = sum(starved_i) / total_jobs`.

Fairness:
- Per-user delivered share over evaluation window:
- `share_u = cpu_sec_u / sum(cpu_sec_k)`.
- Equal-share target for MVP:
- `target_u = 1 / N_active_users`.
- Fairness deviation:
- `fairness_dev = 0.5 * sum_u |share_u - target_u|`.
- Jain fairness index:
- `jain = (sum_u share_u)^2 / (N_active_users * sum_u share_u^2)`.

Constraint thresholds:
- `starved_rate <= 0.02`.
- `fairness_dev - fairness_dev_baseline <= 0.05`.
- `jain_baseline - jain <= 0.03`.
- Constraint window must match the selected run evaluation window (section 1.1).
- `fairness_dev_baseline` and `jain_baseline` must be computed under the same window mode and identical alignment window `W` as the candidate run.

## 3) Policy definitions

### 3.1 `FIFO_STRICT`

Rules:
- Queue ordered by submit time.
- No backfilling.
- No fair-share adjustment.
- When resources are available, schedule the first feasible job in FIFO order.

Estimate source:
- Not used for ordering.

### 3.2 `EASY_BACKFILL_BASELINE`

Rules:
- Preserve FIFO queue order and protect head-of-line job.
- Compute reservation start time for head-of-line job using current running jobs and estimated remaining times.
- Any queued job may backfill only when it is feasible now.
- Any queued job may backfill only when its estimated completion time is less than or equal to the head reservation time.
- Among eligible backfill jobs, choose FIFO order.
- Recompute reservation whenever submit/completion events occur.

Estimate source:
- `runtime_requested_sec` (with fallback rule above).

### 3.3 `FIFO_FAIRSHARE_APPROX` (optional scenario)

Rules:
- Base order starts from FIFO.
- For each scheduling decision, compute user overuse on trailing 7-day window:
- `usage_u = sum(allocated_cpu_sec)` for completed jobs by user `u`.
- `share_u = usage_u / sum(usage_k)` across users with pending jobs.
- `target_u = 1 / N_active_users`.
- `overuse_u = max(0, share_u - target_u)`.
- Priority key: `overuse_u ASC`, then `submit_ts ASC`, then `job_id ASC`.
- No backfill in this policy.

Estimate source:
- Not used for ordering.

### 3.4 `ML_BACKFILL_P50` (candidate policy)

Rules:
- Follows EASY behavior and constraints.
- Backfill eligibility uses uncertainty-adjusted runtime:
- `runtime_guard = p50 + runtime_guard_k * (p90 - p50)`.
- `runtime_guard_k` is a first-class policy parameter with default `0.5`.
- Recommended sensitivity sweep for analysis: `runtime_guard_k in {0.0, 0.5, 1.0, 1.5}`.
- Head-of-line reservation uses predicted runtime `p90` for conservative protection.
- Backfill is allowed only if `completion(runtime_guard) <= T_h` (head reservation time).
- Strict uncertainty mode requires `completion(p90) <= T_h`.
- If prediction is unavailable for a job, fallback to `runtime_requested_sec` and then fallback rule.

Estimate source:
- Prediction quantiles (`p50`, `p90`) from model artifact tied to run manifest.

Fallback accounting (mandatory):
- Track and report `prediction_used_count`.
- Track and report `requested_fallback_count`.
- Track and report `actual_fallback_count`.
- Report percentages over total scheduled jobs for each category.

## 4) Baseline fidelity gate

Goal:
- Validate simulator realism before using counterfactual claims.

Procedure:
1. Replay historical trace with `FIFO_STRICT`.
2. Replay historical trace with `EASY_BACKFILL_BASELINE`.
3. Compare simulated aggregates to observed aggregates from trace.
4. Build observed queue series from trace events (`submit_ts`, `start_ts`) using the queue-series rules in section 2.6.
5. Build simulated queue series with the same sampling contract.

Core metrics:
- Mean wait time.
- p95 wait time.
- Throughput.
- Makespan.
- Wait-time distribution KL divergence (simulation vs observed).
- Slowdown distribution shape similarity (KS statistic).
- Queue-length time-series correlation.

Pass criteria:
- No single core metric diverges by more than 20%.
- No two or more core metrics diverge by more than 15%.
- Wait-time KL divergence must be <= 0.20.
- Slowdown distribution KS statistic must be <= 0.15.
- Queue-length time-series Pearson correlation must be >= 0.85.

Failure handling:
- Block recommendation generation for that trace/run.
- Mark run as `fidelity_failed` and require model/policy claim suppression.

## 5) Reporting requirements

Every simulation report must include:
- Policy ID and policy parameters.
- Estimate source (`requested`, `predicted p50/p90`, fallback rate).
- Random seed and run ID.
- Fidelity gate result and divergence table.
- Fallback accounting percentages for prediction and fallback paths.
- Failure mode report containing policies rejected due to constraint violations.
- Failure mode report containing traces with no primary KPI improvement.
- Failure mode report containing workload patterns where ML policy degrades primary KPI.
- No-improvement narrative for each trace where `ML_BACKFILL_P50` does not beat `EASY_BACKFILL_BASELINE`.
