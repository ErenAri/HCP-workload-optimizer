from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import hpcopt.orchestrate.credibility as credibility_mod


def _patch_credibility_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    trace_df = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 1,
                "runtime_actual_sec": 10,
                "requested_cpus": 2,
                "runtime_requested_sec": 12,
                "user_id": 1,
                "group_id": 1,
                "queue_id": 1,
                "partition_id": 1,
            }
        ]
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_parquet(dataset_path, index=False)
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps({"source_trace_filename": "sample_trace.swf", "source_trace_sha256": "abc"}),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    feature_path = tmp_path / "features.parquet"
    trace_df.to_parquet(feature_path, index=False)
    fidelity_report_path = tmp_path / "fidelity.json"
    fidelity_report_path.write_text("{}", encoding="utf-8")
    recommendation_path = tmp_path / "recommendation.json"
    recommendation_path.write_text(json.dumps({"status": "accepted"}), encoding="utf-8")

    monkeypatch.setattr(
        credibility_mod,
        "ingest_swf",
        lambda **_kwargs: SimpleNamespace(
            dataset_path=dataset_path,
            dataset_metadata_path=metadata_path,
        ),
    )
    monkeypatch.setattr(credibility_mod, "assert_reference_trace_hash_match", lambda **_kwargs: None)
    monkeypatch.setattr(
        credibility_mod,
        "build_trace_profile",
        lambda **_kwargs: SimpleNamespace(profile_path=profile_path),
    )
    monkeypatch.setattr(
        credibility_mod,
        "build_feature_dataset",
        lambda **_kwargs: SimpleNamespace(feature_dataset_path=feature_path),
    )

    def _fake_train_runtime_quantile_models(**kwargs):
        model_dir = Path(kwargs["out_dir"]) / str(kwargs["model_id"])
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = model_dir / "metrics.json"
        metadata_out = model_dir / "metadata.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "quantiles": {
                        "p10": {"pinball_loss": 1.0},
                        "p50": {"pinball_loss": 1.0},
                        "p90": {"pinball_loss": 1.0},
                    }
                }
            ),
            encoding="utf-8",
        )
        metadata_out.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            model_dir=model_dir,
            metrics_path=metrics_path,
            metadata_path=metadata_out,
        )

    monkeypatch.setattr(credibility_mod, "train_runtime_quantile_models", _fake_train_runtime_quantile_models)

    class _StubRuntimeQuantilePredictor:
        def __init__(self, _model_dir: Path):
            self.model_dir = _model_dir

        def predict_one(self, _features: dict[str, object]) -> dict[str, float]:
            return {"p10": 4.0, "p50": 8.0, "p90": 12.0}

    monkeypatch.setattr(credibility_mod, "RuntimeQuantilePredictor", _StubRuntimeQuantilePredictor)

    def _fake_simulation(**_kwargs):
        jobs = pd.DataFrame(
            [
                {
                    "job_id": 1,
                    "submit_ts": 1,
                    "start_ts": 1,
                    "end_ts": 2,
                    "runtime_actual_sec": 1,
                    "requested_cpus": 1,
                }
            ]
        )
        return SimpleNamespace(
            jobs_df=jobs,
            metrics={"p95_bsld": 1.0},
            objective_metrics={
                "p95_bsld": 1.0,
                "mean_wait_sec": 1.0,
                "utilization": 0.75,
                "fairness_dev": 0.01,
                "starvation_rate": 0.0,
                "jain_fairness": 0.99,
            },
            fallback_accounting={
                "total_scheduled_jobs": 1,
                "prediction_used_count": 1,
                "requested_fallback_count": 0,
                "actual_fallback_count": 0,
            },
            invariant_report={"run_id": "x", "strict_mode": True, "step_count": 1, "violations": []},
            queue_series_df=jobs,
        )

    monkeypatch.setattr(credibility_mod, "run_simulation_from_trace", _fake_simulation)
    monkeypatch.setattr(
        credibility_mod,
        "run_baseline_fidelity_gate",
        lambda **_kwargs: SimpleNamespace(report_path=fidelity_report_path, status="pass"),
    )
    monkeypatch.setattr(
        credibility_mod,
        "generate_recommendation_report",
        lambda **_kwargs: SimpleNamespace(report_path=recommendation_path, payload={"status": "accepted"}),
    )
    monkeypatch.setattr(credibility_mod, "build_manifest", lambda **_kwargs: {"run_id": "cred-manifest"})
    monkeypatch.setattr(credibility_mod, "write_manifest", lambda *_args, **_kwargs: None)

    trace_input = tmp_path / "trace.swf"
    trace_input.write_text("; synthetic", encoding="utf-8")
    return trace_input, dataset_path


def test_credibility_candidate_report_single_predictor_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path, _dataset_path = _patch_credibility_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(credibility_mod, "_HAS_LIGHTGBM", False)

    result = credibility_mod.run_credibility_protocol(
        trace_path=trace_path,
        trace_id="trace_single",
        capacity_cpus=8,
        output_dir=tmp_path / "out",
    )
    assert result.status == "pass"
    assert result.candidate_report_path is not None

    payload = json.loads(result.candidate_report_path.read_text(encoding="utf-8"))
    summary = payload["predictor_ensemble"]
    assert summary["type"] == "single"
    assert "model_dir" in summary


def test_credibility_candidate_report_ensemble_predictor_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path, _dataset_path = _patch_credibility_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(credibility_mod, "_HAS_LIGHTGBM", True)

    class _StubEnsemble:
        summary = {
            "type": "ensemble",
            "n_models": 2,
            "members": [{"name": "a", "weight": 0.6}, {"name": "b", "weight": 0.4}],
        }

    monkeypatch.setattr(
        credibility_mod.EnsemblePredictor,
        "from_model_dirs",
        lambda *_args, **_kwargs: _StubEnsemble(),
    )

    result = credibility_mod.run_credibility_protocol(
        trace_path=trace_path,
        trace_id="trace_ensemble",
        capacity_cpus=8,
        output_dir=tmp_path / "out",
    )
    assert result.status == "pass"
    assert result.candidate_report_path is not None

    payload = json.loads(result.candidate_report_path.read_text(encoding="utf-8"))
    summary = payload["predictor_ensemble"]
    assert summary["type"] == "ensemble"
    assert summary["n_models"] == 2
