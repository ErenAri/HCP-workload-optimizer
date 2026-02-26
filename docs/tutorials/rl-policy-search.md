# RL Policy Search

The RL policy search module finds optimal scheduling parameters by running simulations with different configurations and selecting the best one based on p95 Bounded Slowdown.

## How It Works

The RL environment models scheduling as a decision problem:

- **State**: Queue depth, utilization, wait times, completion progress
- **Action**: Backfill threshold, short-job priority boost, starvation cap
- **Reward**: Negative BSLD + utilization bonus

## Grid Search

The most practical approach for this 3-parameter action space:

```python
from pathlib import Path
import pandas as pd
from hpcopt.simulate.rl_env import SchedulingEnv, grid_search_policy

# Load trace
df = pd.read_json("outputs/curated/ctc_sp2.json")

# Create environment
env = SchedulingEnv(trace_df=df, capacity_cpus=512)

# Search
best_action, best_result, all_results = grid_search_policy(
    env,
    bf_thresholds=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0],
    short_boosts=[0.0, 0.25, 0.5, 0.75, 1.0],
)

print(f"Best p95_BSLD: {best_result.p95_bsld:.2f}")
print(f"Backfill threshold: {best_action.backfill_threshold:.2f}")
print(f"Short-job boost: {best_action.priority_boost_short:.2f}")
```

## Random Search

For broader exploration with more configurations:

```python
from hpcopt.simulate.rl_env import random_search_policy

best_action, best_result = random_search_policy(
    env, n_trials=200, seed=42
)
```

## Results on CTC-SP2

| Policy | p95 BSLD | vs EASY_BACKFILL |
|---|---|---|
| FIFO | 173.13 | — |
| EASY_BACKFILL | 3.82 | baseline |
| **RL-Optimized** | **2.85** | **+25.5%** |

The optimal policy found: full backfill (`threshold=1.0`) with strong short-job priority boost (`boost=1.0`).

## Running the Script

```bash
python scripts/rl_policy_search.py
```

Results are saved to `outputs/benchmark/rl_search/grid_search_results.json`.
