"""Run RL policy search on CTC-SP2 trace."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    from hpcopt.simulate.rl_env import SchedulingEnv, grid_search_policy

    dataset = Path("outputs/benchmark/curated/ctc_sp2.json")
    if not dataset.exists():
        print("ERROR: " + str(dataset) + " not found")
        return

    with open(dataset) as f:
        jobs = json.load(f)

    df = pd.DataFrame(jobs)
    print("Loaded " + str(len(df)) + " jobs")

    env = SchedulingEnv(
        trace_df=df,
        capacity_cpus=512,
        decision_interval=100,
    )

    print("\n=== Grid Search Policy Optimization ===")
    print("Searching over backfill_threshold x priority_boost_short")

    t0 = time.perf_counter()
    best_action, best_result, all_results = grid_search_policy(
        env,
        bf_thresholds=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
        short_boosts=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    elapsed = time.perf_counter() - t0

    print("\n=== RESULTS ===")
    print("Search time: " + format(elapsed, ".1f") + "s (" + str(len(all_results)) + " configs)")
    print()
    print("Best policy found:")
    print("  backfill_threshold:    " + format(best_action.backfill_threshold, ".2f"))
    print("  priority_boost_short:  " + format(best_action.priority_boost_short, ".2f"))
    print("  p95_bsld:  " + format(best_result.p95_bsld, ".2f"))
    print("  utilization: " + format(best_result.utilization, ".1%"))
    print("  mean_wait: " + format(best_result.mean_wait_sec, ",.0f") + "s")

    # Compare with baselines
    print("\n=== Baseline Comparison ===")
    fifo = env.run_episode(
        __import__("hpcopt.simulate.rl_env", fromlist=["SchedulingAction"]).SchedulingAction(
            backfill_threshold=0.0, priority_boost_short=0.0
        )
    )
    print("FIFO:           p95_bsld=" + format(fifo.p95_bsld, ".2f") + ", util=" + format(fifo.utilization, ".1%"))

    easy = env.run_episode(
        __import__("hpcopt.simulate.rl_env", fromlist=["SchedulingAction"]).SchedulingAction(
            backfill_threshold=1.0, priority_boost_short=0.0
        )
    )
    print("EASY_BACKFILL:  p95_bsld=" + format(easy.p95_bsld, ".2f") + ", util=" + format(easy.utilization, ".1%"))
    print("RL-OPTIMIZED:   p95_bsld=" + format(best_result.p95_bsld, ".2f") + ", util=" + format(best_result.utilization, ".1%"))

    if easy.p95_bsld > 0:
        improvement = (1 - best_result.p95_bsld / easy.p95_bsld) * 100
        print("RL vs EASY improvement: " + format(improvement, "+.1f") + "%")

    # Save results
    out_dir = Path("outputs/benchmark/rl_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "grid_search_results.json", "w") as f:
        json.dump({
            "best_action": {
                "backfill_threshold": best_action.backfill_threshold,
                "priority_boost_short": best_action.priority_boost_short,
            },
            "best_metrics": {
                "p95_bsld": best_result.p95_bsld,
                "utilization": best_result.utilization,
                "mean_wait_sec": best_result.mean_wait_sec,
            },
            "baselines": {
                "fifo": {"p95_bsld": fifo.p95_bsld, "utilization": fifo.utilization},
                "easy_backfill": {"p95_bsld": easy.p95_bsld, "utilization": easy.utilization},
            },
            "all_results": all_results,
            "search_time_sec": elapsed,
        }, f, indent=2)
    print("\nSaved to: " + str(out_dir / "grid_search_results.json"))


if __name__ == "__main__":
    main()
