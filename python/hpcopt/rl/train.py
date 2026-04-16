"""PPO training loop for the RLScheduler-style environment.

Uses ``sb3_contrib.MaskablePPO`` so the agent's softmax is restricted to
feasible queue slots only.  Hyperparameter defaults follow the
RLScheduler reference implementation (https://github.com/DIR-LAB/RLScheduler):
clip_range=0.2, lr=3e-4, n_steps=4096, batch_size=256, gamma=1.0, gae_lambda=0.97.

Heavy deps (``torch``, ``stable_baselines3``, ``sb3_contrib``) are imported
lazily; install ``hpc-workload-optimizer[rl]``.

Example
-------
.. code-block:: python

    from hpcopt.rl.train import train_ppo
    from hpcopt.ingest.swf import ingest_swf

    res = ingest_swf("data/SDSC-SP2.swf", out_dir="data/parquet",
                     dataset_id="sdsc", report_dir="reports")
    df = pd.read_parquet(res.parquet_path)
    train_ppo(df, capacity_cpus=128, total_timesteps=100_000,
              save_path="models/rl/sdsc_ppo.zip")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def train_ppo(
    trace_df: pd.DataFrame,
    capacity_cpus: int,
    total_timesteps: int = 100_000,
    save_path: str | Path = "models/rl/ppo_policy.zip",
    *,
    learning_rate: float = 3e-4,
    n_steps: int = 4096,
    batch_size: int = 256,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    gamma: float = 1.0,
    gae_lambda: float = 0.97,
    ent_coef: float = 0.0,
    max_jobs_per_episode: int = 256,
    window_random: bool = True,
    seed: int = 42,
    verbose: int = 1,
) -> Any:
    """Train a MaskablePPO agent on ``trace_df``.

    Returns the trained ``MaskablePPO`` instance and writes it to ``save_path``.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "train_ppo requires sb3-contrib. Install with: "
            "pip install 'hpc-workload-optimizer[rl]'"
        ) from exc

    from hpcopt.rl.env import RLSchedulerEnv
    from hpcopt.rl.features import KernelFeaturesExtractor

    env = RLSchedulerEnv(
        trace_df=trace_df,
        capacity_cpus=capacity_cpus,
        max_jobs=max_jobs_per_episode,
        window_random=window_random,
        seed=seed,
    )

    policy_kwargs = {
        "features_extractor_class": KernelFeaturesExtractor,
        "features_extractor_kwargs": {"kernel_hidden": 32},
        "net_arch": [],  # the kernel already produces per-slot logits
    }

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        seed=seed,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
    )

    logger.info("Starting PPO training: total_timesteps=%d", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("Saved trained model to %s", save_path)

    return model
