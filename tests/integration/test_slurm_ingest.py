"""Integration tests for Slurm sacct ingestion."""

from pathlib import Path

import pandas as pd


def test_slurm_ingest_parsable2(tmp_path: Path) -> None:
    """Test parsing sacct --parsable2 output format."""
    from hpcopt.ingest.slurm import ingest_slurm

    # Create a sample sacct output
    sacct_content = (
        "JobID|Submit|Start|End|Elapsed|AllocCPUS|ReqCPUS|ReqMem|User|Group|Partition|State\n"
        "1001|2024-01-15T10:00:00|2024-01-15T10:00:30|2024-01-15T10:05:30|00:05:00|4|4|8000M|user1|grp1|normal|COMPLETED\n"
        "1002|2024-01-15T10:01:00|2024-01-15T10:02:00|2024-01-15T10:12:00|00:10:00|8|8|16000M|user2|grp1|normal|COMPLETED\n"
        "1003|2024-01-15T10:02:00|2024-01-15T10:03:00|2024-01-15T10:08:00|00:05:00|2|2|4000M|user1|grp1|gpu|COMPLETED\n"
        "1004_1|2024-01-15T10:03:00|2024-01-15T10:04:00|2024-01-15T10:09:00|00:05:00|1|1|2000M|user3|grp2|normal|COMPLETED\n"
        "1004_2|2024-01-15T10:03:00|2024-01-15T10:04:30|2024-01-15T10:09:30|00:05:00|1|1|2000M|user3|grp2|normal|COMPLETED\n"
    )
    input_file = tmp_path / "sacct_output.txt"
    input_file.write_text(sacct_content, encoding="utf-8")

    result = ingest_slurm(
        input_path=input_file,
        out_dir=tmp_path / "curated",
        dataset_id="slurm_test",
        report_dir=tmp_path / "reports",
    )

    assert result.dataset_path.exists()
    assert result.row_count >= 3  # At least 3 non-array jobs + 2 array tasks

    df = pd.read_parquet(result.dataset_path)
    assert "job_id" in df.columns
    assert "submit_ts" in df.columns
    assert "runtime_actual_sec" in df.columns
    assert "requested_cpus" in df.columns
    assert len(df) >= 3
