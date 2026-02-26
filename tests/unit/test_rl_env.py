"""Tests for the RL scheduling environment and policy search."""
from __future__ import annotations

import pandas as pd

from hpcopt.simulate.rl_env import SchedulingEnv, grid_search_policy, random_search_policy


def _make_trace(n_jobs: int = 20) -> pd.DataFrame:
    """Create a small synthetic trace for testing."""
    rows = []
    for i in range(n_jobs):
        rows.append(
            {
                "job_id": i,
                "submit_ts": i * 10,
                "runtime_actual_sec": 50 + (i % 5) * 20,
                "requested_cpus": 1 + (i % 4),
                "runtime_requested_sec": 100 + (i % 3) * 50,
                "user_id": i % 3,
            }
        )
    return pd.DataFrame(rows)


def test_scheduling_env_reset() -> None:
    """Environment resets to initial state."""
    df = _make_trace()
    env = SchedulingEnv(trace_df=df, capacity_cpus=16)
    obs = env.reset()
    assert hasattr(obs, "queue_depth")
    assert hasattr(obs, "utilization")
    assert obs.capacity_cpus == 16


def test_scheduling_env_step() -> None:
    """Environment steps without error."""
    df = _make_trace()
    env = SchedulingEnv(trace_df=df, capacity_cpus=16)
    env.reset()
    action = {"backfill_threshold": 0.5, "priority_boost_short": 0.0, "starvation_cap": 100}
    obs, reward, done, info = env.step(action)
    assert hasattr(obs, "queue_depth")
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_scheduling_env_full_episode() -> None:
    """Running a full episode terminates."""
    df = _make_trace(10)
    env = SchedulingEnv(trace_df=df, capacity_cpus=16)
    env.reset()
    action = {"backfill_threshold": 1.0, "priority_boost_short": 0.5, "starvation_cap": 50}
    for _ in range(200):
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done is True
    assert "p95_bsld" in info
    assert "utilization" in info


def test_grid_search_policy() -> None:
    """Grid search completes and finds a best policy."""
    df = _make_trace(10)
    env = SchedulingEnv(trace_df=df, capacity_cpus=16)
    best_action, best_result, all_results = grid_search_policy(
        env,
        bf_thresholds=[0.0, 1.0],
        short_boosts=[0.0, 1.0],
    )
    assert best_action is not None
    assert best_result is not None
    assert best_result.p95_bsld >= 0
    assert len(all_results) == 4  # 2x2 grid


def test_random_search_policy() -> None:
    """Random search completes and finds a best policy."""
    df = _make_trace(10)
    env = SchedulingEnv(trace_df=df, capacity_cpus=16)
    best_action, best_result = random_search_policy(env, n_trials=5, seed=42)
    assert best_action is not None
    assert best_result is not None
    assert best_result.p95_bsld >= 0
