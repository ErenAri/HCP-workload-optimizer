from hpcopt.simulate.adapter import (
    AdapterEvent,
    choose_easy_backfill,
    choose_ml_backfill_p50,
    order_events,
    parse_state_snapshot,
)


def test_adapter_snapshot_requires_fields() -> None:
    payload = {
        "clock_ts": 100,
        "capacity_cpus": 32,
        # free_cpus is intentionally missing.
        "queued_jobs": [],
        "running_jobs": [],
    }
    try:
        parse_state_snapshot(payload)
    except ValueError as exc:
        assert "free_cpus" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing snapshot fields.")


def test_equal_timestamp_event_ordering_contract() -> None:
    events = [
        AdapterEvent(ts=42, event_type="dispatch", job_id=9),
        AdapterEvent(ts=42, event_type="job_submit", job_id=7),
        AdapterEvent(ts=42, event_type="job_complete", job_id=8),
    ]
    ordered = order_events(events)
    assert [event.event_type for event in ordered] == [
        "job_complete",
        "job_submit",
        "dispatch",
    ]


def test_easy_decision_is_deterministic_for_identical_inputs() -> None:
    snapshot = parse_state_snapshot(
        {
            "clock_ts": 10,
            "capacity_cpus": 16,
            "free_cpus": 8,
            "queued_jobs": [
                {"job_id": 2, "submit_ts": 5, "requested_cpus": 4, "runtime_estimate_sec": 20},
                {"job_id": 1, "submit_ts": 5, "requested_cpus": 4, "runtime_estimate_sec": 10},
            ],
            "running_jobs": [],
        }
    )
    decision_a = choose_easy_backfill(snapshot)
    decision_b = choose_easy_backfill(snapshot)
    assert decision_a.decisions == decision_b.decisions
    assert [decision.job_id for decision in decision_a.decisions] == [1, 2]


def test_easy_reservation_enforcement_blocks_unsafe_backfill() -> None:
    snapshot = parse_state_snapshot(
        {
            "clock_ts": 0,
            "capacity_cpus": 8,
            "free_cpus": 2,
            "queued_jobs": [
                {"job_id": 100, "submit_ts": 0, "requested_cpus": 8, "runtime_estimate_sec": 30},
                {"job_id": 101, "submit_ts": 1, "requested_cpus": 2, "runtime_estimate_sec": 20},
            ],
            "running_jobs": [
                {"job_id": 200, "end_ts": 10, "allocated_cpus": 6},
            ],
        }
    )
    decision = choose_easy_backfill(snapshot)
    # HoL reservation is at t=10; job 101 would complete at t=20, so it must be blocked.
    assert decision.reservation_ts == 10
    assert [item.job_id for item in decision.decisions] == []


def test_ml_backfill_strict_mode_uses_p90_gate() -> None:
    snapshot = parse_state_snapshot(
        {
            "clock_ts": 0,
            "capacity_cpus": 8,
            "free_cpus": 2,
            "queued_jobs": [
                {
                    "job_id": 100,
                    "submit_ts": 0,
                    "requested_cpus": 8,
                    "runtime_estimate_sec": 30,
                    "runtime_p90_sec": 30,
                    "runtime_guard_sec": 30,
                    "estimate_source": "prediction",
                },
                {
                    "job_id": 101,
                    "submit_ts": 1,
                    "requested_cpus": 2,
                    "runtime_estimate_sec": 5,
                    "runtime_p90_sec": 20,
                    "runtime_guard_sec": 7,
                    "estimate_source": "prediction",
                },
            ],
            "running_jobs": [
                {"job_id": 200, "end_ts": 10, "allocated_cpus": 6},
            ],
        }
    )
    loose = choose_ml_backfill_p50(snapshot, strict_uncertainty_mode=False)
    strict = choose_ml_backfill_p50(snapshot, strict_uncertainty_mode=True)
    assert [item.job_id for item in loose.decisions] == [101]
    assert [item.job_id for item in strict.decisions] == []
