"""Inference glue: trained RL policy → simulator dispatcher.

Provides ``RLPolicy`` which wraps a saved ``MaskablePPO`` model and exposes
``predict_action(snapshot) -> int`` returning the index into the snapshot's
sorted queued-jobs tuple.  The companion ``choose_rl_trained`` adapter
dispatcher consumes that to emit a single ``DispatchDecision`` per call.

The simulator's main loop calls the dispatcher repeatedly until the agent
chooses a no-op (no feasible action) or the queue empties — matching the
RLScheduler protocol where each agent step picks at most one job.

If ``stable_baselines3`` is unavailable the module still imports; calling
``RLPolicy.load`` raises ``ImportError`` with installation guidance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hpcopt.simulate.adapter import (
    DispatchDecision,
    SchedulerDecision,
    SchedulerStateSnapshot,
)

logger = logging.getLogger(__name__)


class RLPolicy:
    """Loaded MaskablePPO policy ready to act on simulator snapshots."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str | Path) -> RLPolicy:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RLPolicy.load requires sb3-contrib. "
                "Install with: pip install 'hpc-workload-optimizer[rl]'"
            ) from exc
        model = MaskablePPO.load(str(path))
        return cls(model)

    def predict_action(self, obs: np.ndarray, action_masks: np.ndarray) -> int:
        action, _ = self._model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        return int(action)


def _build_obs_and_mask(
    snapshot: SchedulerStateSnapshot,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Construct observation matrix + mask from a SchedulerStateSnapshot.

    Returns ``(obs, mask, queue_list)`` where ``queue_list`` is the FIFO-
    sorted list of queued jobs aligned with the obs rows.
    """
    # Lazy import keeps the simulator's core import-light.
    from hpcopt.rl.env import MAX_QUEUE_SIZE, _encode_jobs

    queue = sorted(snapshot.queued_jobs, key=lambda j: (j.submit_ts, j.job_id))
    queue_dicts = [
        {
            "job_id": j.job_id,
            "submit_ts": j.submit_ts,
            "requested_cpus": j.requested_cpus,
            "runtime_actual_sec": j.runtime_estimate_sec,
            "runtime_estimate_sec": j.runtime_estimate_sec,
        }
        for j in queue
    ]
    obs = _encode_jobs(
        queued=queue_dicts,
        clock_ts=snapshot.clock_ts,
        capacity_cpus=snapshot.capacity_cpus,
        free_cpus=snapshot.free_cpus,
    )
    mask = np.zeros(MAX_QUEUE_SIZE, dtype=bool)
    for i, j in enumerate(queue[:MAX_QUEUE_SIZE]):
        if j.requested_cpus <= snapshot.free_cpus:
            mask[i] = True
    return obs, mask, queue


def choose_rl_trained(
    snapshot: SchedulerStateSnapshot,
    policy: RLPolicy | None,
) -> SchedulerDecision:
    """Adapter dispatcher for the ``RL_TRAINED`` policy.

    Repeatedly queries the model until either the queue is empty, no
    queued job fits, or the model picks a masked slot (which we treat as
    "stop dispatching this tick").
    """
    decisions: list[DispatchDecision] = []
    if policy is None:
        # No model loaded → fall back to FIFO behaviour for safety.
        logger.warning("RL_TRAINED invoked without a loaded policy; falling back to FIFO.")
        from hpcopt.simulate.adapter import choose_fifo_strict
        fb = choose_fifo_strict(snapshot)
        return SchedulerDecision(
            policy_id="RL_TRAINED",
            reservation_ts=fb.reservation_ts,
            decisions=tuple(
                DispatchDecision(
                    job_id=d.job_id,
                    requested_cpus=d.requested_cpus,
                    runtime_estimate_sec=d.runtime_estimate_sec,
                    estimated_completion_ts=d.estimated_completion_ts,
                    reason="rl_fallback_fifo",
                )
                for d in fb.decisions
            ),
        )

    available = snapshot.free_cpus
    # Build a mutable working snapshot so successive picks see the
    # post-dispatch state.
    queued_remaining: list = list(sorted(snapshot.queued_jobs, key=lambda j: (j.submit_ts, j.job_id)))

    for _ in range(len(queued_remaining)):
        # Build snapshot-shaped scratch and encode.
        from hpcopt.simulate.adapter import SchedulerStateSnapshot as _S
        scratch = _S(
            clock_ts=snapshot.clock_ts,
            capacity_cpus=snapshot.capacity_cpus,
            free_cpus=available,
            queued_jobs=tuple(queued_remaining),
            running_jobs=snapshot.running_jobs,
        )
        obs, mask, queue = _build_obs_and_mask(scratch)
        if not mask.any():
            break
        action = policy.predict_action(obs, mask)
        if action >= len(queue) or not mask[action]:
            break
        chosen = queue[action]
        decisions.append(
            DispatchDecision(
                job_id=chosen.job_id,
                requested_cpus=chosen.requested_cpus,
                runtime_estimate_sec=chosen.runtime_estimate_sec,
                estimated_completion_ts=snapshot.clock_ts + chosen.runtime_estimate_sec,
                reason="rl_trained_pick",
            )
        )
        available -= chosen.requested_cpus
        queued_remaining = [j for j in queued_remaining if j.job_id != chosen.job_id]

    return SchedulerDecision(
        policy_id="RL_TRAINED",
        reservation_ts=None,
        decisions=tuple(decisions),
    )
