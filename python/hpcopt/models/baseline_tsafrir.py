"""Tsafrir / Etsion / Feitelson user-history runtime predictor.

Reference:
    D. Tsafrir, Y. Etsion, D. G. Feitelson,
    "Backfilling Using System-Generated Predictions Rather Than User Runtime Estimates",
    IEEE TPDS 18(6), 2007.

Algorithm (Section 4 of the paper):
    predict(user) = mean of the user's last TWO actually-completed runtimes,
    clamped above by the user-supplied wall-time request:

        predict = min( (r_{n-1} + r_n) / 2, user_estimate )

    Cold start:
        - 0 completed jobs by this user  ->  predict = user_estimate
        - 1 completed job by this user   ->  predict = min(r_1, user_estimate)

    Ordering: "last two completed" is by completion timestamp (end_ts),
    not submit timestamp.

Notes:
    * Per-user only. The paper found per-(user, executable) did not help
      (Section 4.3); we follow that finding.
    * The wall-time clamp is critical: without it, predictions above the
      user's request are useless because the scheduler still kills the job
      at user_estimate (Section 3, Section 6).
    * Section 5 of the paper describes an online "extension on overshoot"
      mechanism (double the prediction up to the user wall-time when a
      running job exceeds its current prediction). That is a *runtime*
      correction applied while a job is running, not a submit-time
      prediction. The minimum viable baseline implemented here covers
      submit-time prediction only; the overshoot-extension mechanism is
      tracked as a follow-up because it requires changes to the simulation
      core's running-job state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TsafrirPrediction:
    """Result of a single Tsafrir prediction."""

    runtime_estimate_sec: int
    completed_history_count: int  # how many completed jobs by this user fed the prediction (0, 1, or 2)
    clamped_by_user_estimate: bool


def predict_one(
    user_id: int | None,
    user_estimate_sec: int,
    history_runtimes_sec: list[int],
) -> TsafrirPrediction:
    """Compute a single Tsafrir prediction.

    Args:
        user_id: User identifier (used only for traceability; not consumed here).
        user_estimate_sec: User-supplied wall-time request, in seconds. Must be > 0.
            This is also the upper bound on the returned prediction.
        history_runtimes_sec: This user's most recent completed-job runtimes,
            ordered MOST-RECENT FIRST. The first two entries are used.

    Returns:
        TsafrirPrediction with the clamped runtime estimate and history metadata.
    """
    if user_estimate_sec <= 0:
        raise ValueError("user_estimate_sec must be > 0")

    # Filter out non-positive runtimes defensively.
    history = [int(r) for r in history_runtimes_sec if int(r) > 0]
    n = len(history)

    if n == 0:
        return TsafrirPrediction(
            runtime_estimate_sec=int(user_estimate_sec),
            completed_history_count=0,
            clamped_by_user_estimate=False,
        )

    if n == 1:
        raw = history[0]
    else:
        # Average of the two most recent completions.
        raw = (history[0] + history[1]) // 2
        if raw < 1:
            raw = 1

    clamped = raw > user_estimate_sec
    estimate = min(raw, int(user_estimate_sec))
    return TsafrirPrediction(
        runtime_estimate_sec=int(max(1, estimate)),
        completed_history_count=min(n, 2),
        clamped_by_user_estimate=clamped,
    )


def compute_tsafrir_estimates(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute Tsafrir runtime estimates for every job in a trace.

    Walks the trace in (submit_ts, job_id) order, maintaining a per-user
    list of "completed" runtimes. A job's contribution to its user's
    history becomes visible only at its actual completion timestamp
    (submit_ts + runtime_actual_sec), not at submit time -- this preserves
    online causality.

    Required columns on jobs_df:
        - job_id (int)
        - submit_ts (int)
        - runtime_actual_sec (int)
        - runtime_estimate_sec (int)  -- the user-supplied wall-time request
            (or its fallback value, as already produced by coerce_trace_df).
        - user_id (nullable; None values are treated as a single shared "anonymous" pool).

    Returns:
        A copy of jobs_df with three additional columns:
            - tsafrir_runtime_sec (int)
            - tsafrir_history_count (int 0..2)
            - tsafrir_clamped (bool)
    """
    required = {"job_id", "submit_ts", "runtime_actual_sec", "runtime_estimate_sec"}
    missing = required - set(jobs_df.columns)
    if missing:
        raise ValueError(f"compute_tsafrir_estimates: missing columns {sorted(missing)}")

    df = jobs_df.copy().reset_index(drop=True)

    # Build a list of (event_ts, event_type, user_id, job_idx) events.
    # event_type 0 = "completion is now visible to subsequent submits"
    # event_type 1 = "submit"
    # We process completions before submits at the same ts so that a job
    # finishing at exactly the same instant another job submits is visible.
    submits: list[tuple[int, int, object, int]] = []
    completions: list[tuple[int, int, object, int]] = []

    for idx, row in df.iterrows():
        submit_ts = int(row["submit_ts"])
        runtime_actual = int(row["runtime_actual_sec"])
        end_ts = submit_ts + runtime_actual
        user_id = row.get("user_id")
        # Treat NaN / None user ids as a sentinel "anonymous" group.
        if user_id is None or (isinstance(user_id, float) and pd.isna(user_id)):
            user_key: object = "__anon__"
        else:
            try:
                user_key = int(user_id)
            except (TypeError, ValueError):
                user_key = str(user_id)

        submits.append((submit_ts, 1, user_key, int(idx)))
        completions.append((end_ts, 0, user_key, int(idx)))

    events = sorted(submits + completions, key=lambda e: (e[0], e[1], e[3]))

    # Per-user history of recent completed runtimes, MOST RECENT FIRST,
    # capped at 2 (we never need more than 2).
    history: dict[object, list[int]] = {}

    out_estimate = [0] * len(df)
    out_history_count = [0] * len(df)
    out_clamped = [False] * len(df)

    for ts, event_type, user_key, idx in events:
        if event_type == 0:
            # Completion: prepend this job's actual runtime to the user's history.
            actual = int(df.at[idx, "runtime_actual_sec"])
            if actual <= 0:
                continue
            buf = history.setdefault(user_key, [])
            buf.insert(0, actual)
            if len(buf) > 2:
                del buf[2:]
        else:
            # Submit: read the current history (excluding this job, since its
            # completion event has not fired yet for the same ts unless event_type 0
            # was processed first -- which the sort above guarantees).
            user_estimate = int(df.at[idx, "runtime_estimate_sec"])
            if user_estimate <= 0:
                # Fall back to actual runtime to keep the estimate positive;
                # this matches the coerce_trace_df behavior for missing requests.
                user_estimate = max(1, int(df.at[idx, "runtime_actual_sec"]))
            buf = history.get(user_key, [])
            pred = predict_one(
                user_id=user_key if isinstance(user_key, int) else None,
                user_estimate_sec=user_estimate,
                history_runtimes_sec=buf,
            )
            out_estimate[idx] = pred.runtime_estimate_sec
            out_history_count[idx] = pred.completed_history_count
            out_clamped[idx] = pred.clamped_by_user_estimate

    df["tsafrir_runtime_sec"] = out_estimate
    df["tsafrir_history_count"] = out_history_count
    df["tsafrir_clamped"] = out_clamped
    return df
