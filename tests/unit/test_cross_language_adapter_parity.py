import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hpcopt.simulate.adapter import (
    SchedulerDecision,
    choose_easy_backfill,
    choose_fifo_strict,
    choose_ml_backfill_p50,
    parse_state_snapshot,
)


def _decision_to_dict(decision: SchedulerDecision) -> dict:
    return {
        "policy_id": decision.policy_id,
        "reservation_ts": decision.reservation_ts,
        "decisions": [
            {
                "job_id": item.job_id,
                "requested_cpus": item.requested_cpus,
                "runtime_estimate_sec": item.runtime_estimate_sec,
                "estimated_completion_ts": item.estimated_completion_ts,
                "reason": item.reason,
            }
            for item in decision.decisions
        ],
    }


def _run_rust_contract(fixture: Path, policy_id: str, strict_uncertainty_mode: bool = False) -> dict:
    if shutil.which("cargo") is None:
        pytest.fail("cargo is required for cross-language parity test; install Rust toolchain")
    root = Path(__file__).resolve().parents[2]
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "sim-runner",
        "--bin",
        "adapter_contract",
        "--",
        "--input",
        str(fixture),
        "--policy",
        policy_id,
    ]
    if strict_uncertainty_mode:
        cmd.append("--strict-uncertainty-mode")

    proc = subprocess.run(
        cmd,
        cwd=root / "rust",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"rust adapter_contract failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(proc.stdout)


@pytest.mark.slow
def test_cross_language_fifo_easy_ml_contract_parity() -> None:
    fixture = Path("tests/fixtures/adapter_snapshot_case.json").resolve()
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    snapshot = parse_state_snapshot(payload)

    expected_fifo = _decision_to_dict(choose_fifo_strict(snapshot))
    expected_easy = _decision_to_dict(choose_easy_backfill(snapshot))
    expected_ml = _decision_to_dict(choose_ml_backfill_p50(snapshot, strict_uncertainty_mode=False))
    expected_ml_strict = _decision_to_dict(choose_ml_backfill_p50(snapshot, strict_uncertainty_mode=True))

    assert _run_rust_contract(fixture, "FIFO_STRICT") == expected_fifo
    assert _run_rust_contract(fixture, "EASY_BACKFILL_BASELINE") == expected_easy
    assert _run_rust_contract(fixture, "ML_BACKFILL_P50") == expected_ml
    assert _run_rust_contract(fixture, "ML_BACKFILL_P50", strict_uncertainty_mode=True) == expected_ml_strict
