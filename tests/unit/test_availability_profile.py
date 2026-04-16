"""Unit tests for the AvailabilityProfile data structure."""

from __future__ import annotations

from hpcopt.simulate.availability_profile import AvailabilityProfile


def test_empty_profile_returns_initial_free():
    p = AvailabilityProfile.from_snapshot(clock_ts=100, free_cpus=64, running=[])
    assert p.free_at(100) == 64
    assert p.free_at(99) == 64  # before base_ts
    assert p.free_at(1_000_000) == 64


def test_profile_releases_running_jobs():
    p = AvailabilityProfile.from_snapshot(
        clock_ts=0, free_cpus=10, running=[(50, 4), (100, 2)]
    )
    assert p.free_at(0) == 10
    assert p.free_at(49) == 10
    assert p.free_at(50) == 14
    assert p.free_at(99) == 14
    assert p.free_at(100) == 16


def test_find_window_now_when_enough_free():
    p = AvailabilityProfile.from_snapshot(clock_ts=0, free_cpus=10, running=[])
    assert p.find_earliest_window(req=4, runtime=100, after=0) == 0


def test_find_window_waits_for_release():
    # Only 2 free now; 4 free at t=50, 6 at t=100. Need 5 cpus.
    p = AvailabilityProfile.from_snapshot(
        clock_ts=0, free_cpus=2, running=[(50, 2), (100, 2)]
    )
    assert p.find_earliest_window(req=5, runtime=10, after=0) == 100


def test_insert_blocks_subsequent_dispatch():
    p = AvailabilityProfile.from_snapshot(clock_ts=0, free_cpus=10, running=[])
    # Reserve 8 cpus from t=0..50 → only 2 free in that window.
    p.insert(start=0, duration=50, req=8)
    assert p.free_at(0) == 2
    assert p.free_at(49) == 2
    assert p.free_at(50) == 10
    # A new request for 5 cpus over 30s cannot start now; must wait 50.
    assert p.find_earliest_window(req=5, runtime=30, after=0) == 50


def test_insert_with_release_in_middle():
    # 4 free now, +2 at t=20 → 6 free; +2 at t=80 → 8 free.
    p = AvailabilityProfile.from_snapshot(
        clock_ts=0, free_cpus=4, running=[(20, 2), (80, 2)]
    )
    # Need 5 cpus for 50s. Cannot start at 0 (only 4). At 20 we get 6 free
    # for 60s (until t=80) — fits.
    assert p.find_earliest_window(req=5, runtime=50, after=0) == 20


def test_insert_does_not_affect_outside_window():
    p = AvailabilityProfile.from_snapshot(clock_ts=0, free_cpus=10, running=[])
    p.insert(start=100, duration=50, req=5)
    assert p.free_at(0) == 10
    assert p.free_at(99) == 10
    assert p.free_at(100) == 5
    assert p.free_at(149) == 5
    assert p.free_at(150) == 10
