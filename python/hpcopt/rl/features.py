"""Kernel-based features extractor for the RLScheduler-style policy.

Mirrors the "shared kernel" trick from Zhang et al. (SC'20): the same
small MLP is applied independently to each of the ``MAX_QUEUE_SIZE``
queued-job feature vectors, producing one logit per slot.  The resulting
policy is permutation-equivariant and order-invariant — critical for a
Discrete-over-queue-slots action space.

This module imports ``torch`` and ``stable_baselines3`` lazily; install
``hpc-workload-optimizer[rl]`` to use it.
"""

from __future__ import annotations

try:
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from torch import nn
except ImportError as exc:  # pragma: no cover - surfaced at import time
    raise ImportError(
        "hpcopt.rl.features requires torch + stable-baselines3. "
        "Install with: pip install 'hpc-workload-optimizer[rl]'"
    ) from exc

from hpcopt.rl.env import JOB_FEATURES, MAX_QUEUE_SIZE


class KernelFeaturesExtractor(BaseFeaturesExtractor):
    """Shared per-slot MLP producing one logit per queue position.

    The output has shape ``(batch, MAX_QUEUE_SIZE)`` so that downstream
    ``MaskableActorCriticPolicy`` can treat the features themselves as
    action logits via a final identity head.  In practice we let SB3 add
    its own action head; the kernel just produces a permutation-equivariant
    embedding of the queue.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        kernel_hidden: int = 32,
        features_dim: int = MAX_QUEUE_SIZE,
    ):
        if observation_space.shape != (MAX_QUEUE_SIZE, JOB_FEATURES):
            raise ValueError(
                f"KernelFeaturesExtractor expects obs shape "
                f"({MAX_QUEUE_SIZE},{JOB_FEATURES}); got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=features_dim)
        self.kernel = nn.Sequential(
            nn.Linear(JOB_FEATURES, kernel_hidden),
            nn.Tanh(),
            nn.Linear(kernel_hidden, kernel_hidden // 2),
            nn.Tanh(),
            nn.Linear(kernel_hidden // 2, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, MAX_QUEUE_SIZE, JOB_FEATURES) — SB3 may flatten,
        # so reshape defensively.
        x = observations
        if x.dim() == 2 and x.shape[-1] == MAX_QUEUE_SIZE * JOB_FEATURES:
            x = x.view(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
        # Apply kernel across slots: (B,128,8) -> (B,128,1) -> (B,128)
        return self.kernel(x).squeeze(-1)
