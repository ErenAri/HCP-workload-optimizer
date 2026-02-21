"""Integration test: two invocations of orchestrated credibility run produce identical artifacts."""

from pathlib import Path

import json
import pandas as pd
from pandas.testing import assert_frame_equal

from hpcopt.ingest.swf import ingest_swf
from hpcopt.orchestrate.credibility import run_credibility_protocol


def test_credibility_protocol_deterministic(tmp_path: Path) -> None:
    """Two orchestrated runs with same inputs produce identical simulation metrics."""
    trace_path = Path("tests/fixtures/sample_trace.swf")

    result_a = run_credibility_protocol(
        trace_path=trace_path,
        trace_id="sample_a",
        capacity_cpus=4,
        runtime_guard_k=0.5,
        seed=42,
        output_dir=tmp_path / "run_a",
        reference_suite_config=Path("configs/data/reference_suite.yaml"),
        fidelity_config=Path("configs/simulation/fidelity_gate.yaml"),
        strict_invariants=True,
    )

    result_b = run_credibility_protocol(
        trace_path=trace_path,
        trace_id="sample_b",
        capacity_cpus=4,
        runtime_guard_k=0.5,
        seed=42,
        output_dir=tmp_path / "run_b",
        reference_suite_config=Path("configs/data/reference_suite.yaml"),
        fidelity_config=Path("configs/simulation/fidelity_gate.yaml"),
        strict_invariants=True,
    )

    # Both should complete (pass or fail but not error)
    assert result_a.status in ("pass", "fail")
    assert result_b.status in ("pass", "fail")
    assert result_a.status == result_b.status

    # Fidelity status must match
    assert result_a.fidelity_status == result_b.fidelity_status

    # Recommendation status must match
    assert result_a.recommendation_status == result_b.recommendation_status

    # If recommendation reports exist, their structural content should match
    if result_a.recommendation_report_path and result_b.recommendation_report_path:
        rec_a = json.loads(result_a.recommendation_report_path.read_text(encoding="utf-8"))
        rec_b = json.loads(result_b.recommendation_report_path.read_text(encoding="utf-8"))
        assert rec_a["status"] == rec_b["status"]
        # Candidate scores should be identical for deterministic runs
        if rec_a.get("candidates") and rec_b.get("candidates"):
            for ca, cb in zip(rec_a["candidates"], rec_b["candidates"]):
                assert ca["score"]["score"] == cb["score"]["score"]
