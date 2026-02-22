"""CLI tests for ingest commands (swf, slurm, pbs)."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hpcopt.cli.main import app


def test_ingest_swf_cli(tmp_path: Path, sample_trace_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "curated"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "ingest", "swf",
            "--input", str(sample_trace_path),
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--dataset-id", "cli_test_swf",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "cli_test_swf.parquet").exists()
    assert "Dataset:" in result.output
    assert "Rows:" in result.output


def test_ingest_slurm_cli(tmp_path: Path) -> None:
    # Create a minimal sacct --parsable2 fixture
    sacct_content = (
        "JobID|Submit|Start|End|Elapsed|AllocCPUS|ReqCPUS|ReqMem|User|Group|Partition|State\n"
        "1001|2024-01-01T00:00:00|2024-01-01T00:01:00|2024-01-01T00:11:00|00:10:00|4|4|4000Mc|alice|grp1|batch|COMPLETED\n"
        "1002|2024-01-01T00:05:00|2024-01-01T00:06:00|2024-01-01T00:16:00|00:10:00|8|8|8000Mc|bob|grp1|batch|COMPLETED\n"
    )
    input_path = tmp_path / "sacct_dump.txt"
    input_path.write_text(sacct_content, encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "curated"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "ingest", "slurm",
            "--input", str(input_path),
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--dataset-id", "cli_test_slurm",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dataset:" in result.output


def test_ingest_pbs_cli(tmp_path: Path) -> None:
    # Create a minimal PBS accounting log fixture
    pbs_content = (
        "01/01/2024 00:00:00;E;1001.server;user=alice group=grp1 queue=batch "
        "Resource_List.ncpus=4 Resource_List.walltime=00:10:00 "
        "resources_used.walltime=00:08:00 ctime=1704067200 "
        "qtime=1704067200 etime=1704067200 start=1704067260 end=1704067740\n"
        "01/01/2024 00:05:00;E;1002.server;user=bob group=grp1 queue=batch "
        "Resource_List.ncpus=8 Resource_List.walltime=00:15:00 "
        "resources_used.walltime=00:10:00 ctime=1704067500 "
        "qtime=1704067500 etime=1704067500 start=1704067560 end=1704068160\n"
    )
    input_path = tmp_path / "pbs_acct.log"
    input_path.write_text(pbs_content, encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "curated"
    report_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "ingest", "pbs",
            "--input", str(input_path),
            "--out", str(out_dir),
            "--report-out", str(report_dir),
            "--dataset-id", "cli_test_pbs",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dataset:" in result.output
