from pathlib import Path

import pandas as pd
from hpcopt.ingest.swf import ingest_swf


def test_ingest_swf_emits_canonical_dataset_and_quality_report(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/sample_trace.swf")
    out_dir = tmp_path / "curated"
    report_dir = tmp_path / "reports"

    result = ingest_swf(
        input_path=fixture,
        out_dir=out_dir,
        dataset_id="sample_trace",
        report_dir=report_dir,
    )

    assert result.row_count == 3
    assert result.dataset_path.exists()
    assert result.quality_report_path.exists()
    assert result.dataset_metadata_path.exists()

    df = pd.read_parquet(result.dataset_path)
    assert set(df.columns) >= {
        "job_id",
        "submit_ts",
        "start_ts",
        "end_ts",
        "wait_sec",
        "runtime_actual_sec",
        "runtime_requested_sec",
        "allocated_cpus",
        "requested_cpus",
        "requested_mem",
        "user_id",
        "queue_id",
        "runtime_overrequest_ratio",
    }
    assert int(df["requested_mem"].isna().sum()) == 2
    assert float(df.loc[df["job_id"] == 1, "runtime_overrequest_ratio"].iloc[0]) == 2.0
