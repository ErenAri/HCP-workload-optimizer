"""Tests for PBS/Torque accounting log ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest
from hpcopt.ingest.pbs import (
    _canonical_job_id,
    _parse_mem_kb,
    _parse_ncpus_from_nodes,
    _parse_pbs_timestamp,
    _parse_walltime,
    ingest_pbs,
)

# --- Helper parsing tests ---


def test_parse_walltime_hhmmss() -> None:
    assert _parse_walltime("01:30:00") == 5400
    assert _parse_walltime("00:05:30") == 330


def test_parse_walltime_ddhhmmss() -> None:
    assert _parse_walltime("1:02:03:04") == 86400 + 7200 + 180 + 4


def test_parse_walltime_mmss() -> None:
    assert _parse_walltime("05:30") == 330


def test_parse_walltime_empty() -> None:
    assert _parse_walltime("") is None
    assert _parse_walltime("   ") is None


def test_parse_pbs_timestamp_epoch() -> None:
    assert _parse_pbs_timestamp("1704067200") == 1704067200


def test_parse_pbs_timestamp_formatted() -> None:
    ts = _parse_pbs_timestamp("01/01/2024 00:00:00")
    assert ts is not None
    assert ts > 0


def test_parse_pbs_timestamp_empty() -> None:
    assert _parse_pbs_timestamp("") is None


def test_parse_mem_kb() -> None:
    assert _parse_mem_kb("4096kb") is not None
    assert _parse_mem_kb("4gb") == 4 * 1024
    assert _parse_mem_kb("1024mb") == 1024
    assert _parse_mem_kb("") is None
    assert _parse_mem_kb("0") == 0


def test_parse_ncpus_from_nodes_ppn() -> None:
    assert _parse_ncpus_from_nodes("1:ppn=8") == 8
    assert _parse_ncpus_from_nodes("2:ppn=4") == 8


def test_parse_ncpus_from_nodes_simple() -> None:
    assert _parse_ncpus_from_nodes("1") == 1


def test_parse_ncpus_from_nodes_prefix() -> None:
    assert _parse_ncpus_from_nodes("nodes=1:ppn=4") == 4


def test_parse_ncpus_from_nodes_multi() -> None:
    assert _parse_ncpus_from_nodes("1:ppn=4+1:ppn=8") == 12


def test_canonical_job_id_strip_server() -> None:
    assert _canonical_job_id("12345.pbs-server") == "12345"


def test_canonical_job_id_array_job() -> None:
    assert _canonical_job_id("12345[0].pbs-server") == "12345_0"


def test_canonical_job_id_plain() -> None:
    assert _canonical_job_id("12345") == "12345"


# --- Integration tests ---


def test_ingest_pbs_basic(tmp_path: Path) -> None:
    pbs_content = (
        "01/01/2024 00:00:00;E;1001.server;user=alice group=grp1 queue=batch "
        "Resource_List.ncpus=4 Resource_List.walltime=00:10:00 "
        "resources_used.walltime=00:08:00 ctime=1704067200 "
        "qtime=1704067200 etime=1704067200 start=1704067260 end=1704067740\n"
    )
    input_path = tmp_path / "pbs.log"
    input_path.write_text(pbs_content, encoding="utf-8")

    result = ingest_pbs(
        input_path=input_path,
        out_dir=tmp_path / "out",
        dataset_id="pbs_test",
        report_dir=tmp_path / "reports",
    )
    assert result.row_count == 1
    assert result.dataset_path.exists()


def test_ingest_pbs_non_exit_records_skipped(tmp_path: Path) -> None:
    content = (
        "01/01/2024 00:00:00;Q;1001.server;user=alice queue=batch\n"
        "01/01/2024 00:00:00;S;1001.server;user=alice queue=batch\n"
        "01/01/2024 00:10:00;E;1001.server;user=alice group=grp1 queue=batch "
        "Resource_List.ncpus=4 resources_used.walltime=00:08:00 ctime=1704067200 "
        "start=1704067260 end=1704067740\n"
    )
    input_path = tmp_path / "pbs2.log"
    input_path.write_text(content, encoding="utf-8")

    result = ingest_pbs(
        input_path=input_path,
        out_dir=tmp_path / "out2",
        dataset_id="pbs_skip",
        report_dir=tmp_path / "rep2",
    )
    assert result.row_count == 1


def test_ingest_pbs_empty_file_raises(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.log"
    input_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No parsable PBS"):
        ingest_pbs(
            input_path=input_path,
            out_dir=tmp_path / "out",
            dataset_id="empty",
            report_dir=tmp_path / "rep",
        )


def test_ingest_pbs_malformed_lines(tmp_path: Path) -> None:
    content = (
        "this is not a valid PBS line\n"
        "01/01/2024 00:10:00;E;1001.server;user=alice group=grp1 queue=batch "
        "Resource_List.ncpus=4 resources_used.walltime=00:08:00 ctime=1704067200 "
        "start=1704067260 end=1704067740\n"
    )
    input_path = tmp_path / "mixed.log"
    input_path.write_text(content, encoding="utf-8")

    result = ingest_pbs(
        input_path=input_path,
        out_dir=tmp_path / "out",
        dataset_id="mixed",
        report_dir=tmp_path / "rep",
    )
    assert result.row_count == 1
