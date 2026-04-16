"""RL_TRAINED policy fallback path — runs without gymnasium / sb3.

Validates that requesting ``RL_TRAINED`` with no loaded model degrades
gracefully to FIFO behaviour rather than crashing the simulator.  This
keeps the policy registered as 'safe to enumerate' even on systems
without the ``[rl]`` extras.
"""

from __future__ import annotations

import pandas as pd
from hpcopt.simulate.core import SUPPORTED_POLICIES, run_simulation_from_trace


def _make_trace(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "job_id": i,
                "submit_ts": i * 5,
                "runtime_actual_sec": 50 + (i % 4) * 25,
                "requested_cpus": 1 + (i % 4),
            }
        )
    return pd.DataFrame(rows)


def test_rl_trained_registered_in_supported_policies():
    assert "RL_TRAINED" in SUPPORTED_POLICIES


def test_rl_trained_falls_back_to_fifo_without_policy():
    trace = _make_trace(8)
    result = run_simulation_from_trace(
        trace_df=trace,
        policy_id="RL_TRAINED",
        capacity_cpus=8,
        run_id="rl_fallback",
        strict_invariants=True,
        policy_context=None,
    )
    assert len(result.jobs_df) == len(trace)
    assert result.invariant_report["violations"] == []
