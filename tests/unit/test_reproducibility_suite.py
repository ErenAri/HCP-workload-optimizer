import json
from pathlib import Path

import pandas as pd
from hpcopt.artifacts.manifest import build_manifest
from hpcopt.features.pipeline import build_feature_dataset
from hpcopt.ingest.swf import ingest_swf
from hpcopt.simulate.core import run_simulation_from_trace
from hpcopt.simulate.fidelity import run_baseline_fidelity_gate
from hpcopt.simulate.stress import generate_stress_scenario
from pandas.testing import assert_frame_equal


class _DeterministicPredictor:
    def predict_one(self, features: dict) -> dict[str, float]:
        requested = float(features.get("runtime_requested_sec") or 100.0)
        return {
            "p10": max(1.0, requested * 0.4),
            "p50": max(1.0, requested * 0.8),
            "p90": max(1.0, requested * 1.1),
        }


def test_simulation_replay_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    trace_df = pd.read_parquet(ingest.dataset_path)
    predictor = _DeterministicPredictor()

    run_a = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id="ML_BACKFILL_P50",
        capacity_cpus=4,
        run_id="deterministic_a",
        strict_invariants=True,
        runtime_predictor=predictor,
        runtime_guard_k=0.5,
        strict_uncertainty_mode=False,
    )
    run_b = run_simulation_from_trace(
        trace_df=trace_df,
        policy_id="ML_BACKFILL_P50",
        capacity_cpus=4,
        run_id="deterministic_b",
        strict_invariants=True,
        runtime_predictor=predictor,
        runtime_guard_k=0.5,
        strict_uncertainty_mode=False,
    )

    assert_frame_equal(run_a.jobs_df, run_b.jobs_df, check_like=False)
    assert_frame_equal(run_a.queue_series_df, run_b.queue_series_df, check_like=False)
    assert run_a.metrics == run_b.metrics
    assert run_a.objective_metrics == run_b.objective_metrics
    assert run_a.fallback_accounting == run_b.fallback_accounting
    assert run_a.invariant_report["step_count"] == run_b.invariant_report["step_count"]
    assert run_a.invariant_report["violations"] == run_b.invariant_report["violations"]


def test_stress_generation_is_seed_reproducible(tmp_path: Path) -> None:
    left = generate_stress_scenario(
        scenario="user_skew",
        out_dir=tmp_path / "left",
        n_jobs=250,
        seed=77,
        params={"top_user_share": 0.6},
    )
    right = generate_stress_scenario(
        scenario="user_skew",
        out_dir=tmp_path / "right",
        n_jobs=250,
        seed=77,
        params={"top_user_share": 0.6},
    )

    left_df = pd.read_parquet(left.dataset_path)
    right_df = pd.read_parquet(right.dataset_path)
    assert_frame_equal(left_df, right_df, check_like=False)


def test_fidelity_gate_is_deterministic(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    trace_df = pd.read_parquet(ingest.dataset_path)

    result_a = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=4,
        out_path=tmp_path / "fidelity_a.json",
        run_id="fidelity_a",
        strict_invariants=True,
    )
    result_b = run_baseline_fidelity_gate(
        trace_df=trace_df,
        capacity_cpus=4,
        out_path=tmp_path / "fidelity_b.json",
        run_id="fidelity_b",
        strict_invariants=True,
    )

    assert result_a.status == result_b.status

    report_a = json.loads(result_a.report_path.read_text(encoding="utf-8"))
    report_b = json.loads(result_b.report_path.read_text(encoding="utf-8"))
    # Structural status must match even if timestamps differ
    assert report_a["status"] == report_b["status"]


def test_feature_pipeline_is_deterministic(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )

    result_a = build_feature_dataset(
        dataset_path=ingest.dataset_path,
        out_dir=tmp_path / "features_a",
        report_dir=tmp_path / "reports_a",
        dataset_id="feat_a",
    )
    result_b = build_feature_dataset(
        dataset_path=ingest.dataset_path,
        out_dir=tmp_path / "features_b",
        report_dir=tmp_path / "reports_b",
        dataset_id="feat_b",
    )

    df_a = pd.read_parquet(result_a.feature_dataset_path)
    df_b = pd.read_parquet(result_b.feature_dataset_path)
    assert_frame_equal(df_a, df_b, check_like=False)
    assert result_a.row_count == result_b.row_count
    assert result_a.fold_count == result_b.fold_count


def test_manifest_includes_enhanced_fields(tmp_path: Path) -> None:
    dummy_input = tmp_path / "input.txt"
    dummy_input.write_text("test", encoding="utf-8")
    dummy_output = tmp_path / "output.txt"
    dummy_output.write_text("result", encoding="utf-8")

    manifest = build_manifest(
        command="test-command",
        inputs=[dummy_input],
        outputs=[dummy_output],
        params={"key": "value"},
        seeds=[42],
    )

    # Enhanced manifest fields (A3.1)
    assert "pip_freeze_snapshot" in manifest
    assert isinstance(manifest["pip_freeze_snapshot"], list)
    assert "os_fingerprint" in manifest
    assert isinstance(manifest["os_fingerprint"], dict)
    assert "system" in manifest["os_fingerprint"]
    assert "model_hash" in manifest
    assert "git_commit" in manifest
