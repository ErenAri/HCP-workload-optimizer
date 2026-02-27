from pathlib import Path

import pandas as pd
from hpcopt.ingest.swf import ingest_swf
from hpcopt.simulate.fidelity import run_baseline_fidelity_gate


def test_fidelity_gate_generates_policy_reports(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    trace_df = pd.read_parquet(ingest.dataset_path)
    result = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=4,
        out_path=tmp_path / "reports" / "fidelity.json",
        run_id="fidelity_fixture",
        config_path=None,
        strict_invariants=False,
    )
    assert result.report_path.exists()
    assert result.status in {"pass", "fail"}
    assert "FIFO_STRICT" in result.report["policy_reports"]
    assert "EASY_BACKFILL_BASELINE" in result.report["policy_reports"]


def test_fidelity_gate_small_fixture_skips_distribution_noise(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    trace_df = pd.read_parquet(ingest.dataset_path)
    result = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=4,
        out_path=tmp_path / "reports" / "fidelity_small_fixture.json",
        run_id="fidelity_small_fixture",
        config_path=None,
        strict_invariants=False,
    )

    assert result.status == "pass"
    for report in result.report["policy_reports"].values():
        checks = report["distribution_checks"]
        assert checks["skipped_small_sample"] is True
        assert checks["sample_size"] < checks["min_sample_size_required"]
        assert "wait_kl_exceeded" not in report["fail_reasons"]
        assert "slowdown_ks_exceeded" not in report["fail_reasons"]
