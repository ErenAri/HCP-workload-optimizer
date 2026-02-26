"""Reinforcement learning environment for HPC scheduling policy search.

Provides a Gymnasium-compatible environment where an RL agent learns
to make scheduling decisions (job ordering, backfill thresholds) by
interacting with the discrete-event simulator.

The agent observes cluster state (queue depth, utilization, job features)
and outputs scheduling parameters that control the backfill policy.

Usage:
    from hpcopt.simulate.rl_env import SchedulingEnv, train_policy

    env = SchedulingEnv(trace_df=df, capacity_cpus=512)
    policy = train_policy(env, episodes=200)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SchedulingAction:
    """Action space for the RL agent.

    backfill_threshold: fraction of capacity below which backfill is attempted
        (0.0 = no backfill, 1.0 = always backfill)
    priority_boost_short: bonus priority for short jobs (< median runtime)
        (0.0 = no boost, 1.0 = max boost)
    starvation_cap_factor: multiplier on default starvation wait cap
        (0.5 = aggressive, 2.0 = permissive)
    """

    backfill_threshold: float = 0.8
    priority_boost_short: float = 0.0
    starvation_cap_factor: float = 1.0


@dataclass
class SchedulingState:
    """Observation space for the RL agent."""

    clock_ts: int = 0
    queue_depth: int = 0
    running_count: int = 0
    free_cpus: int = 0
    capacity_cpus: int = 0
    utilization: float = 0.0
    avg_wait_sec: float = 0.0
    p95_wait_sec: float = 0.0
    jobs_completed: int = 0
    jobs_total: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to normalized feature array for neural network input."""
        cap = max(self.capacity_cpus, 1)
        total = max(self.jobs_total, 1)
        return np.array(
            [
                self.queue_depth / total,
                self.running_count / total,
                self.free_cpus / cap,
                self.utilization,
                min(self.avg_wait_sec / 86400.0, 10.0),  # normalize to days, cap at 10
                min(self.p95_wait_sec / 86400.0, 10.0),
                self.jobs_completed / total,
            ],
            dtype=np.float32,
        )


@dataclass
class EpisodeResult:
    """Result of one RL training episode."""

    p95_bsld: float
    utilization: float
    mean_wait_sec: float
    makespan_sec: int
    reward: float
    actions_taken: list[SchedulingAction] = field(default_factory=list)


class SchedulingEnv:
    """Gymnasium-like environment for scheduling policy search.

    The environment runs a discrete-event simulation on a job trace.
    At each decision point (when the queue changes), the agent observes
    cluster state and outputs scheduling parameters.
    """

    def __init__(
        self,
        trace_df: pd.DataFrame,
        capacity_cpus: int,
        decision_interval: int = 50,
    ):
        """
        Args:
            trace_df: Job trace with submit_ts, runtime_actual_sec, requested_cpus
            capacity_cpus: Total cluster CPU capacity
            decision_interval: How many job events between RL decisions
        """
        self.capacity_cpus = capacity_cpus
        self.decision_interval = decision_interval

        # Prepare jobs sorted by submit time
        required = ["job_id", "submit_ts", "runtime_actual_sec", "requested_cpus"]
        df = trace_df[required].copy()
        df = df.sort_values("submit_ts").reset_index(drop=True)
        self.jobs = df.to_dict("records")
        self.n_jobs = len(self.jobs)

        # State
        self._reset_state()

    def _reset_state(self) -> None:
        self.clock = 0
        self.free_cpus = self.capacity_cpus
        self.queued: list[dict] = []
        self.running: list[dict] = []
        self.completed: list[dict] = []
        self.job_idx = 0
        self.step_count = 0
        self.wait_times: list[float] = []
        self.cpu_time_used = 0
        self.total_time = 0

    def reset(self) -> SchedulingState:
        """Reset environment to initial state."""
        self._reset_state()
        return self._observe()

    def _observe(self) -> SchedulingState:
        waits = self.wait_times if self.wait_times else [0.0]
        return SchedulingState(
            clock_ts=self.clock,
            queue_depth=len(self.queued),
            running_count=len(self.running),
            free_cpus=self.free_cpus,
            capacity_cpus=self.capacity_cpus,
            utilization=1.0 - (self.free_cpus / self.capacity_cpus) if self.capacity_cpus > 0 else 0,
            avg_wait_sec=float(np.mean(waits)),
            p95_wait_sec=float(np.percentile(waits, 95)) if len(waits) >= 2 else 0,
            jobs_completed=len(self.completed),
            jobs_total=self.n_jobs,
        )

    def step(self, action: SchedulingAction) -> tuple[SchedulingState, float, bool]:
        """Execute one decision step.

        Returns:
            (observation, reward, done)
        """
        events_processed = 0

        while events_processed < self.decision_interval:
            # Admit new jobs that have arrived
            while self.job_idx < self.n_jobs and self.jobs[self.job_idx]["submit_ts"] <= self.clock:
                job = self.jobs[self.job_idx].copy()
                job["queue_enter_ts"] = self.clock
                self.queued.append(job)
                self.job_idx += 1

            # Dispatch from queue using RL action parameters
            self._dispatch(action)

            # Advance clock to next event
            next_ts = self._next_event_time()
            if next_ts is None:
                break

            elapsed = max(next_ts - self.clock, 0)
            busy_cpus = self.capacity_cpus - self.free_cpus
            self.cpu_time_used += busy_cpus * elapsed
            self.total_time += self.capacity_cpus * elapsed
            self.clock = next_ts

            # Complete finished jobs
            still_running = []
            for rj in self.running:
                if rj["end_ts"] <= self.clock:
                    wait = rj["start_ts"] - rj["submit_ts"]
                    self.wait_times.append(float(wait))
                    self.completed.append(rj)
                    self.free_cpus += rj["requested_cpus"]
                else:
                    still_running.append(rj)
            self.running = still_running

            events_processed += 1
            self.step_count += 1

        done = (len(self.completed) == self.n_jobs) or (
            self.job_idx >= self.n_jobs and len(self.queued) == 0 and len(self.running) == 0
        )

        reward = self._compute_reward()
        obs = self._observe()
        return obs, reward, done

    def _dispatch(self, action: SchedulingAction) -> None:
        """Dispatch jobs from queue based on RL action parameters."""
        if not self.queued:
            return

        # Sort queue: apply priority boost for short jobs
        if action.priority_boost_short > 0:
            median_rt = float(np.median([j["runtime_actual_sec"] for j in self.queued]))
            self.queued.sort(
                key=lambda j: (
                    j["submit_ts"]
                    - action.priority_boost_short * 3600 * (1.0 if j["runtime_actual_sec"] < median_rt else 0.0)
                )
            )

        threshold_cpus = int(action.backfill_threshold * self.capacity_cpus)

        # Greedy head-of-queue dispatch
        dispatched_indices = []
        for i, job in enumerate(self.queued):
            if job["requested_cpus"] <= self.free_cpus:
                self._start_job(job)
                dispatched_indices.append(i)
                if self.free_cpus <= 0:
                    break
            elif i == 0:
                # Head of queue blocked — try backfill
                break

        # Remove dispatched
        for i in reversed(dispatched_indices):
            self.queued.pop(i)

        # Backfill: try remaining jobs if capacity above threshold
        if self.free_cpus > 0 and self.free_cpus >= (self.capacity_cpus - threshold_cpus):
            backfilled = []
            for i, job in enumerate(self.queued):
                if job["requested_cpus"] <= self.free_cpus:
                    self._start_job(job)
                    backfilled.append(i)
                    if self.free_cpus <= 0:
                        break
            for i in reversed(backfilled):
                self.queued.pop(i)

    def _start_job(self, job: dict) -> None:
        job["start_ts"] = self.clock
        job["end_ts"] = self.clock + job["runtime_actual_sec"]
        self.free_cpus -= job["requested_cpus"]
        self.running.append(job)

    def _next_event_time(self) -> int | None:
        candidates = []
        if self.running:
            candidates.append(min(rj["end_ts"] for rj in self.running))
        if self.job_idx < self.n_jobs:
            candidates.append(self.jobs[self.job_idx]["submit_ts"])
        return min(candidates) if candidates else None

    def _compute_reward(self) -> float:
        """Reward = negative BSLD (minimize slowdown) + utilization bonus."""
        if not self.wait_times:
            return 0.0

        # BSLD component
        bslds = []
        for rj in self.completed[-self.decision_interval :]:
            wait = rj["start_ts"] - rj["submit_ts"]
            runtime = max(rj["runtime_actual_sec"], 10)
            bslds.append(max(1.0, wait / runtime))

        mean_bsld = float(np.mean(bslds)) if bslds else 1.0

        # Utilization component
        util = self.cpu_time_used / max(self.total_time, 1)

        # Combined reward: low BSLD + high utilization
        reward = -math.log(mean_bsld + 1) + 0.5 * util
        return reward

    def run_episode(self, action: SchedulingAction) -> EpisodeResult:
        """Run full episode with a fixed action (for parameter sweep)."""
        self.reset()
        total_reward = 0.0

        while True:
            _, reward, done = self.step(action)
            total_reward += reward
            if done:
                break

        # Final metrics
        bslds = []
        for rj in self.completed:
            wait = rj["start_ts"] - rj["submit_ts"]
            runtime = max(rj["runtime_actual_sec"], 10)
            bslds.append(max(1.0, wait / runtime))

        waits = [rj["start_ts"] - rj["submit_ts"] for rj in self.completed]
        util = self.cpu_time_used / max(self.total_time, 1)

        return EpisodeResult(
            p95_bsld=float(np.percentile(bslds, 95)) if bslds else 0,
            utilization=float(util),
            mean_wait_sec=float(np.mean(waits)) if waits else 0,
            makespan_sec=self.clock,
            reward=total_reward,
        )


def random_search_policy(
    env: SchedulingEnv,
    n_trials: int = 100,
    seed: int = 42,
) -> tuple[SchedulingAction, EpisodeResult]:
    """Find best scheduling parameters via random search.

    More practical than gradient-based RL for this low-dimensional
    action space (3 continuous parameters).

    Args:
        env: Scheduling environment
        n_trials: Number of random parameter configurations to try
        seed: Random seed

    Returns:
        (best_action, best_result)
    """
    rng = np.random.default_rng(seed)
    best_action = SchedulingAction()
    best_result: EpisodeResult | None = None

    logger.info("Starting random policy search (%d trials)", n_trials)

    for trial in range(n_trials):
        action = SchedulingAction(
            backfill_threshold=float(rng.uniform(0.0, 1.0)),
            priority_boost_short=float(rng.uniform(0.0, 1.0)),
            starvation_cap_factor=float(rng.uniform(0.5, 3.0)),
        )

        result = env.run_episode(action)

        if best_result is None or result.p95_bsld < best_result.p95_bsld:
            best_action = action
            best_result = result
            logger.info(
                "Trial %d/%d: NEW BEST p95_bsld=%.2f, util=%.1f%%, bf_thresh=%.2f, short_boost=%.2f",
                trial + 1,
                n_trials,
                result.p95_bsld,
                result.utilization * 100,
                action.backfill_threshold,
                action.priority_boost_short,
            )

    assert best_result is not None
    return best_action, best_result


def grid_search_policy(
    env: SchedulingEnv,
    bf_thresholds: list[float] | None = None,
    short_boosts: list[float] | None = None,
) -> tuple[SchedulingAction, EpisodeResult, list[dict[str, Any]]]:
    """Systematic grid search over scheduling parameters.

    Args:
        env: Scheduling environment
        bf_thresholds: Backfill threshold values to try
        short_boosts: Short-job priority boost values to try

    Returns:
        (best_action, best_result, all_results)
    """
    if bf_thresholds is None:
        bf_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if short_boosts is None:
        short_boosts = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_results: list[dict[str, Any]] = []
    best_action = SchedulingAction()
    best_result: EpisodeResult | None = None

    total = len(bf_thresholds) * len(short_boosts)
    logger.info("Starting grid search (%d configurations)", total)

    for i, bf in enumerate(bf_thresholds):
        for j, sb in enumerate(short_boosts):
            action = SchedulingAction(
                backfill_threshold=bf,
                priority_boost_short=sb,
            )
            result = env.run_episode(action)

            all_results.append(
                {
                    "backfill_threshold": bf,
                    "priority_boost_short": sb,
                    "p95_bsld": result.p95_bsld,
                    "utilization": result.utilization,
                    "mean_wait_sec": result.mean_wait_sec,
                    "reward": result.reward,
                }
            )

            if best_result is None or result.p95_bsld < best_result.p95_bsld:
                best_action = action
                best_result = result

    assert best_result is not None
    logger.info(
        "Grid search best: p95_bsld=%.2f, util=%.1f%%, bf_thresh=%.2f, short_boost=%.2f",
        best_result.p95_bsld,
        best_result.utilization * 100,
        best_action.backfill_threshold,
        best_action.priority_boost_short,
    )

    return best_action, best_result, all_results
