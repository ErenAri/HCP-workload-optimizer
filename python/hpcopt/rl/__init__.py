"""Reinforcement learning subpackage for HPC scheduling.

Implements an RLScheduler-style (Zhang et al., SC'20) Gymnasium environment,
a kernel-based stable-baselines3 policy, and inference glue that lets a
trained model act as the ``RL_TRAINED`` policy in the discrete-event
simulator.

Heavy dependencies (``gymnasium``, ``stable-baselines3``, ``sb3-contrib``,
``torch``) are optional and live behind the ``[rl]`` extra in
``pyproject.toml``.  Importing submodules will surface clear ImportErrors
if the extras are missing.
"""

__all__ = []  # public API exposed via submodules
