from pathlib import Path

import json
import pandas as pd

from hpcopt.features.pipeline import build_chronological_splits, build_feature_dataset
from hpcopt.ingest.swf import ingest_swf


def test_feature_build_outputs_expected_artifacts_and_columns(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    result = build_feature_dataset(
        dataset_path=ingest.dataset_path,
        out_dir=tmp_path / "features",
        report_dir=tmp_path / "reports",
        dataset_id="sample_trace",
        n_folds=2,
        train_fraction=0.6,
        val_fraction=0.2,
    )
    assert result.feature_dataset_path.exists()
    assert result.split_manifest_path.exists()
    assert result.feature_report_path.exists()
    assert result.row_count == 3
    assert result.fold_count >= 1

    features = pd.read_parquet(result.feature_dataset_path)
    assert set(features.columns) >= {
        "job_id",
        "submit_ts",
        "runtime_actual_sec",
        "queue_congestion_at_submit_jobs",
        "user_runtime_median_lookback",
        "user_runtime_var_lookback",
        "user_job_count_lookback",
        "user_overrequest_mean_lookback",
        "user_behavior_pattern",
        "job_size_class",
        "submit_hour",
        "submit_dow",
    }


def test_feature_lookback_is_time_safe_for_same_user(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 10,
                "start_ts": 10,
                "end_ts": 20,
                "runtime_actual_sec": 10,
                "runtime_requested_sec": 20,
                "requested_cpus": 4,
                "requested_mem": None,
                "user_id": 7,
            },
            {
                "job_id": 2,
                "submit_ts": 20,
                "start_ts": 22,
                "end_ts": 122,
                "runtime_actual_sec": 100,
                "runtime_requested_sec": 120,
                "requested_cpus": 4,
                "requested_mem": None,
                "user_id": 7,
            },
            {
                "job_id": 3,
                "submit_ts": 30,
                "start_ts": 30,
                "end_ts": 40,
                "runtime_actual_sec": 10,
                "runtime_requested_sec": 10,
                "requested_cpus": 2,
                "requested_mem": None,
                "user_id": 8,
            },
        ]
    )
    dataset = tmp_path / "trace.parquet"
    trace.to_parquet(dataset, index=False)
    result = build_feature_dataset(
        dataset_path=dataset,
        out_dir=tmp_path / "features",
        report_dir=tmp_path / "reports",
        dataset_id="trace",
    )
    features = pd.read_parquet(result.feature_dataset_path).sort_values("job_id")

    job2 = features.loc[features["job_id"] == 2].iloc[0]
    # The second job of user 7 must only see the first job runtime in lookback.
    assert float(job2["user_runtime_median_lookback"]) == 10.0
    assert int(job2["user_job_count_lookback"]) == 1


def test_chronological_split_contract_and_manifest(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "job_id": list(range(1, 21)),
            "submit_ts": [i * 10 for i in range(20)],
            "start_ts": [i * 10 for i in range(20)],
            "end_ts": [(i * 10) + 5 for i in range(20)],
            "runtime_actual_sec": [5.0] * 20,
            "runtime_requested_sec": [8.0] * 20,
            "requested_cpus": [2] * 20,
            "requested_mem": [None] * 20,
            "user_id": [1] * 20,
        }
    )
    dataset = tmp_path / "split_trace.parquet"
    df.to_parquet(dataset, index=False)
    result = build_feature_dataset(
        dataset_path=dataset,
        out_dir=tmp_path / "features",
        report_dir=tmp_path / "reports",
        dataset_id="split_trace",
        n_folds=3,
        train_fraction=0.6,
        val_fraction=0.2,
    )

    split_payload = json.loads(result.split_manifest_path.read_text(encoding="utf-8"))
    assert split_payload["n_folds_generated"] >= 1
    for fold in split_payload["folds"]:
        assert fold["chronology_ok"] is True
        assert fold["train"]["row_count"] > 0
        assert fold["validation"]["row_count"] > 0
        assert fold["test"]["row_count"] > 0
        assert fold["train"]["submit_ts_max"] <= fold["validation"]["submit_ts_min"]
        assert fold["validation"]["submit_ts_max"] <= fold["test"]["submit_ts_min"]

    # Direct split helper also enforces chronology.
    feature_df = pd.read_parquet(result.feature_dataset_path)
    folds = build_chronological_splits(feature_df, n_folds=2, train_fraction=0.6, val_fraction=0.2)
    assert len(folds) >= 1
    assert all(bool(fold["chronology_ok"]) for fold in folds)

