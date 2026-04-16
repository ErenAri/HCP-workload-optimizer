"""Time-ordered free-CPU availability profile for Conservative Backfill.

Implements the classic data structure described by Mu'alem & Feitelson
(IEEE TPDS 12(6), 2001, "Utilization, Predictability, Workloads, and User
Runtime Estimates in Scheduling the IBM SP2 with Backfilling") and used by
Batsim's ``conservative_bf``, AccaSim, and the canonical Pyss reference
simulator.

Representation
--------------
``self.events`` is a list of ``(time, free_cpus_after)`` tuples in strictly
increasing time order. Between consecutive events the number of free CPUs is
constant. ``free_at(t)`` returns the free count at time ``t`` (latest event
with ``time <= t``).

Operations
~~~~~~~~~~
* ``free_at(t)``                              — O(log n) via bisect.
* ``find_earliest_window(req, runtime, after)`` — earliest start time at
  which ``req`` CPUs are continuously free for ``runtime`` seconds.
* ``insert(start, duration, req)``            — decrement free count across
  ``[start, start+duration)``; splits events as needed.

This module is deliberately dependency-free (pure stdlib) so the Rust port
in ``rust/sim-runner/`` can mirror the algorithm with confidence.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass
class AvailabilityProfile:
    """Free-CPU timeline anchored at ``base_ts``."""

    base_ts: int
    initial_free: int
    # (time, free_after_this_time) sorted by time ascending; first entry's
    # time is base_ts and free_after is initial_free.
    events: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_snapshot(
        cls,
        clock_ts: int,
        free_cpus: int,
        running: list[tuple[int, int]],
    ) -> AvailabilityProfile:
        """Build a profile from the current free count and running jobs.

        ``running`` is a list of ``(end_ts, allocated_cpus)``.  Each running
        job releases its CPUs at ``end_ts``.  Events at the same time are
        merged.
        """
        prof = cls(base_ts=clock_ts, initial_free=int(free_cpus))
        prof.events = [(int(clock_ts), int(free_cpus))]
        # Group releases by end_ts (in case of ties).
        releases: dict[int, int] = {}
        for end_ts, cpus in running:
            t = max(int(end_ts), int(clock_ts))
            releases[t] = releases.get(t, 0) + int(cpus)
        cur = int(free_cpus)
        for t in sorted(releases.keys()):
            cur += releases[t]
            if prof.events and prof.events[-1][0] == t:
                prof.events[-1] = (t, cur)
            else:
                prof.events.append((t, cur))
        return prof

    # ── helpers ────────────────────────────────────────────────────

    def _times(self) -> list[int]:
        return [t for t, _ in self.events]

    def free_at(self, t: int) -> int:
        if t < self.base_ts:
            return self.initial_free
        times = self._times()
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return self.initial_free
        return self.events[idx][1]

    def _ensure_breakpoint(self, t: int) -> int:
        """Insert an event at ``t`` if not present; return its index."""
        if t < self.base_ts:
            t = self.base_ts
        times = self._times()
        idx = bisect.bisect_left(times, t)
        if idx < len(self.events) and self.events[idx][0] == t:
            return idx
        # Carry forward the free count from the previous event.
        prev_free = self.events[idx - 1][1] if idx > 0 else self.initial_free
        self.events.insert(idx, (t, prev_free))
        return idx

    # ── public API ─────────────────────────────────────────────────

    def find_earliest_window(self, req: int, runtime: int, after: int) -> int:
        """Return the earliest start ``s >= after`` such that ``free_at(t) >= req``
        for all ``t`` in ``[s, s+runtime)``.

        Algorithm: candidate start times are ``after`` and every event time
        ``> after``.  For each candidate, walk subsequent events until either
        we cover ``runtime`` seconds or hit an event with ``free < req``.
        """
        if runtime <= 0:
            return max(after, self.base_ts)
        candidates: list[int] = [max(after, self.base_ts)]
        for t, _ in self.events:
            if t > candidates[0]:
                candidates.append(t)
        # Deduplicate while preserving order.
        seen: set[int] = set()
        uniq: list[int] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)

        for start in uniq:
            end = start + runtime
            if self._window_fits(start, end, req):
                return start
        # Should be unreachable when the profile is unbounded above and
        # eventually all CPUs are free; fall back to a sentinel.
        return 10**18

    def _window_fits(self, start: int, end: int, req: int) -> bool:
        if self.free_at(start) < req:
            return False
        times = self._times()
        # Iterate every event in (start, end); each marks a free-count change.
        idx = bisect.bisect_right(times, start)
        while idx < len(self.events) and self.events[idx][0] < end:
            if self.events[idx][1] < req:
                return False
            idx += 1
        return True

    def insert(self, start: int, duration: int, req: int) -> None:
        """Decrement free count across ``[start, start+duration)`` by ``req``."""
        if req <= 0 or duration <= 0:
            return
        end = start + duration
        self._ensure_breakpoint(start)
        self._ensure_breakpoint(end)
        times = self._times()
        i = bisect.bisect_left(times, start)
        j = bisect.bisect_left(times, end)
        for k in range(i, j):
            t, free = self.events[k]
            self.events[k] = (t, free - req)
