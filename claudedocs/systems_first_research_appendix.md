# Systems-First Research Appendix

Date: 2026-02-19  
Applies to: `claudedocs/mvp_design_plan_python_rust_batsim.md`, `claudedocs/policy_spec_baselines_mvp.md`

## 1) Formal model summary

Let system state be:

```text
S = (t, Q, R, V, E, F, C, A)
```

Where:
- `t`: simulation clock.
- `Q`: queued jobs.
- `R`: running jobs.
- `V`: reservation structure.
- `E`: event priority queue.
- `F`: free resource vector.
- `C`: completed jobs.
- `A`: accounting counters and telemetry.

Transition relation:
- `S --submit(j)--> S'`
- `S --start(j,a)--> S'`
- `S --complete(j)--> S'`

No other transition is permitted to mutate `Q`, `R`, `V`, or `F`.

## 2) Transition obligations

For each transition, implementation must enforce:
- Preconditions.
- Postconditions.
- Explicit state delta.

Reference contract:
- `claudedocs/policy_spec_baselines_mvp.md` section 2.3.

## 3) Complexity analysis (expected)

Assume:
- `n` jobs total.
- `|Q|` queued jobs at a decision step.
- priority data structures implemented as heaps/ordered maps.

Scope boundary:
- These complexity targets apply to our scheduler decision module (policy logic + candidate selection structures), including custom Batsim scheduler components we own.
- They do not claim asymptotic bounds for Batsim internal engine code paths outside our module.

Expected costs:
- Event dequeue/enqueue: `O(log n)` per operation.
- FIFO scheduling check: `O(1)` candidate selection, feasibility check dominated by resource lookup.
- EASY candidate scan (naive): `O(|Q|)` per decision.
- EASY candidate scan (indexed/prunable): `O(log |Q| + k)` where `k` is feasible candidate checks.
- Total replay complexity:
- naive EASY scan: `O(events * |Q|)`
- indexed implementation target: near `O(events log n)` average regime.

Memory complexity:
- `O(n)` for job records + event queue + telemetry artifacts.

## 4) Reservation correctness proof sketch

Claim:
- In `EASY_BACKFILL_BASELINE`, head-of-line job reservation is not delayed by backfilled jobs.

Sketch:
1. Compute reservation time `T_h` for head-of-line job under current running set and estimates.
2. Allow backfill job `b` only if feasible now and `completion(runtime_guard_b) <= T_h` (strict mode: `completion(p90_b) <= T_h`).
3. Starting `b` cannot consume resources beyond `T_h` by rule 2.
4. Therefore resource availability required for head-of-line job at `T_h` remains unchanged.
5. Hence head-of-line reservation protection is preserved.

Proof obligations:
- Correct `T_h` computation.
- Deterministic event ordering for simultaneous timestamps.
- Accurate resource accounting at each transition.

## 5) Determinism proof argument

Given:
- Identical input trace.
- Identical policy/config hash.
- Identical model artifact hash.
- Identical seed(s).
- Deterministic event ordering (`complete -> submit -> dispatch`) at equal timestamps.

Then:
- Transition sequence is identical.
- State sequence `{S_0, S_1, ...}` is identical.
- Output artifacts and metrics are reproducible up to numeric serialization tolerances.

## 6) Counterexample scenario catalog

Purpose:
- Prevent over-claiming and document regimes where gains may fail.

Required scenarios:
- Heavy-tail domination:
- Few very long jobs dominate occupancy; short-job prioritization has limited impact.
- Low congestion regime:
- Queue remains shallow; backfill optimization yields minimal benefit.
- Prediction uncertainty regime:
- High fallback rate or wide intervals reduce ML contribution.
- User skew regime:
- One user dominates submissions; fairness constraints limit possible gains.
- Burst shock regime:
- Sudden submission spikes may temporarily degrade p95 BSLD.

For each scenario report:
- Trigger condition.
- Observed degradation signal.
- Which constraint blocked aggressive policy.
- Mitigation candidates for next iteration.

### 6.1 Executable scenario generators

All catalog scenarios should be runnable via generator tooling.

CLI pattern:
- `hpcopt stress gen --scenario <name> [--param value ...]`
- `hpcopt stress run --scenario <name> --policy <yaml> --model <version>`

Examples:
- `hpcopt stress gen --scenario heavy_tail --alpha 1.2 --n-jobs 5000`
- `hpcopt stress gen --scenario low_congestion --target-util 0.35`
- `hpcopt stress gen --scenario user_skew --top-user-share 0.65`
- `hpcopt stress gen --scenario burst_shock --burst-factor 4 --burst-duration-sec 1800`

Expected degrade signatures (contract):
- `heavy_tail`: elevated p95 BSLD with limited utilization gains.
- `low_congestion`: near-zero delta between ML and EASY.
- `user_skew`: fairness penalty growth and reduced feasible optimization space.
- `burst_shock`: transient queue spike and increased starvation risk until backlog drains.

## 7) Artifact checklist for publication-grade runs

Each published run should include:
- `run_manifest.json` (immutable).
- `fidelity_report.json`.
- `invariant_report.json`.
- `benchmark_report.json`.
- `failure_mode_report.json`.
- `no_improvement_narratives.md`.

Each artifact must reference:
- `git_commit`.
- policy hash.
- model hash/version.
- seed(s).
- environment fingerprint (OS/CPU/RAM/toolchain versions).
