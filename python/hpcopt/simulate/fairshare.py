"""Fairshare priority computation for FAIRSHARE_BACKFILL.

Implements an exponentially-decayed per-user usage tracker patterned after
Slurm's ``PriorityType=priority/multifactor`` fairshare component (Slurm
docs, "Multifactor Priority Plugin").  For each job submission we compute a
priority score ``= -decayed_usage_cpu_seconds(user, t_submit)``.  Higher
score → dispatched earlier.

Decay model
~~~~~~~~~~~
Usage decays with half-life ``H``: ``usage(t) = usage(t0) * 0.5 ** ((t-t0)/H)``.
Each completed job contributes ``cpu_seconds`` of usage, "billed" at the
job's submit time.  The default half-life of 7 days (604800s) matches the
Slurm default.

This module is intentionally trace-level: priorities are precomputed from
the trace by ``compute_fairshare_priorities`` and attached to each job via
``attach_runtime_estimates``.  The dispatcher then sorts the queue by
descending priority.  Keeping the computation outside the inner sim loop
preserves determinism and matches the offline-trace evaluation idiom used
by the rest of the simulator.
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_HALF_LIFE_SEC = 7 * 24 * 3600  # 7 days, Slurm default


def _decayed(usage: float, last_ts: int, now_ts: int, half_life: float) -> float:
    if usage <= 0.0 or now_ts <= last_ts or half_life <= 0:
        return usage
    decay = math.pow(0.5, (now_ts - last_ts) / half_life)
    return usage * decay


def compute_fairshare_priorities(
    jobs_df: pd.DataFrame,
    half_life_sec: float = DEFAULT_HALF_LIFE_SEC,
) -> pd.Series:
    """Return a ``pd.Series`` of fairshare priority scores indexed like ``jobs_df``.

    Higher score → earlier dispatch.  Score is ``-decayed_usage`` so that the
    user with the lowest recent usage wins.

    Causality: a job's score reflects only usage from jobs submitted *before*
    its own ``submit_ts`` and that have *completed* by that time.  This
    mirrors what an online scheduler would know.
    """
    if jobs_df.empty:
        return pd.Series([], dtype=float)

    df = jobs_df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = -1
    df["user_id"] = df["user_id"].fillna(-1)
    df["runtime_actual_sec"] = df["runtime_actual_sec"].fillna(0).astype(int)
    df["requested_cpus"] = df["requested_cpus"].fillna(1).astype(int)
    df["submit_ts"] = df["submit_ts"].astype(int)

    # Build a chronological event list of completions: (completion_ts, user, cpu_sec)
    completion_events: list[tuple[int, object, float]] = []
    for row in df.itertuples(index=False):
        # Without start_ts in the trace, approximate completion as
        # submit + runtime; this is the lowest-information predictor an
        # online scheduler could use, and is enough for relative ordering.
        c_ts = int(row.submit_ts) + int(row.runtime_actual_sec)
        cpu_sec = float(row.requested_cpus) * float(row.runtime_actual_sec)
        completion_events.append((c_ts, row.user_id, cpu_sec))
    completion_events.sort(key=lambda x: x[0])

    # Per-user (decayed_usage, last_decay_ts).
    state: dict[object, tuple[float, int]] = {}
    ev_idx = 0
    n_events = len(completion_events)

    # Process jobs in submission order; for each, fold in any completion
    # events whose c_ts <= submit_ts before reading the score.
    order = df["submit_ts"].argsort(kind="stable").tolist()
    scores = [0.0] * len(df)

    for pos in order:
        sub_ts = int(df.iloc[pos]["submit_ts"])
        user = df.iloc[pos]["user_id"]
        # Apply all completion events that happened by sub_ts.
        while ev_idx < n_events and completion_events[ev_idx][0] <= sub_ts:
            c_ts, c_user, c_cpu = completion_events[ev_idx]
            usage, last_ts = state.get(c_user, (0.0, c_ts))
            usage = _decayed(usage, last_ts, c_ts, half_life_sec)
            usage += c_cpu
            state[c_user] = (usage, c_ts)
            ev_idx += 1
        # Read this user's decayed usage as of sub_ts.
        usage, last_ts = state.get(user, (0.0, sub_ts))
        usage_now = _decayed(usage, last_ts, sub_ts, half_life_sec)
        scores[pos] = -usage_now

    return pd.Series(scores, index=df.index, dtype=float)
