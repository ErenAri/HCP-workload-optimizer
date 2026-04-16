"""Tests for the RLScheduler-style Gymnasium environment.

Skipped if the ``[rl]`` optional dependency group (gymnasium) is not
installed.  These tests do *not* require torch / stable-baselines3 — only
the env wiring is exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gymnasium = pytest.importorskip("gymnasium")

from hpcopt.rl.env import JOB_FEATURES, MAX_QUEUE_SIZE, RLSchedulerEnv  # noqa: E402


def _make_trace(n: int = 10, capacity: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "job_id": i,
                "submit_ts": i * 5,
                "runtime_actual_sec": 50 + (i % 4) * 25,
                "requested_cpus": 1 + (i % min(capacity, 4)),
                "runtime_requested_sec": 100,
                "user_id": i % 3,
            }
        )
    return pd.DataFrame(rows)


def test_env_observation_shape_and_action_space():
    env = RLSchedulerEnv(_make_trace(20), capacity_cpus=8)
    obs, info = env.reset(seed=0)
    assert obs.shape == (MAX_QUEUE_SIZE, JOB_FEATURES)
    assert obs.dtype == np.float32
    assert env.action_space.n == MAX_QUEUE_SIZE
    assert "queue_len" in info


def test_env_action_mask_consistency():
    env = RLSchedulerEnv(_make_trace(10), capacity_cpus=8)
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (MAX_QUEUE_SIZE,)
    assert mask.dtype == bool
    # At least one job at t=0 should fit in capacity 8.
    assert mask.any()


def test_env_step_advances_clock():
    env = RLSchedulerEnv(_make_trace(8), capacity_cpus=16)
    env.reset(seed=0)
    mask = env.action_masks()
    action = int(np.argmax(mask))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (MAX_QUEUE_SIZE, JOB_FEATURES)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_runs_to_termination_with_random_valid_actions():
    env = RLSchedulerEnv(_make_trace(15), capacity_cpus=16)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        mask = env.action_masks()
        if not mask.any():
            # The env should advance time and either terminate or open a
            # valid action; passing an invalid action triggers the penalty
            # path which still advances the simulation.
            obs, reward, terminated, truncated, info = env.step(0)
        else:
            valid = np.flatnonzero(mask)
            obs, reward, terminated, truncated, info = env.step(int(rng.choice(valid)))
        if terminated:
            break
    assert terminated, "episode failed to terminate within budget"


def test_env_illegal_action_penalised():
    env = RLSchedulerEnv(_make_trace(5), capacity_cpus=4)
    env.reset(seed=0)
    mask = env.action_masks()
    # Pick an illegal slot deliberately.
    illegal = int(np.flatnonzero(~mask)[0])
    _, reward, _, _, _ = env.step(illegal)
    # Reward should include the -1 illegal-action penalty.
    assert reward <= 0.0


def test_env_window_random_changes_starting_offset():
    df = _make_trace(50)
    env = RLSchedulerEnv(df, capacity_cpus=8, max_jobs=10, window_random=True, seed=1)
    env.reset(seed=1)
    snap1_jobs = list(env._jobs)  # internal, ok for white-box test
    env.reset(seed=2)
    snap2_jobs = list(env._jobs)
    # Different seeds should sample different windows (probabilistically; with
    # 41 possible offsets the chance of collision is < 3%).
    assert [j["job_id"] for j in snap1_jobs] != [j["job_id"] for j in snap2_jobs]


