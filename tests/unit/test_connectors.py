"""Tests for Slurm and PBS live connectors."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from hpcopt.integrations.slurm_connector import (
    SlurmConnector,
    SlurmJob,
    _parse_slurm_datetime,
    _parse_slurm_time,
)
from hpcopt.integrations.pbs_connector import (
    PBSConnector,
    PBSJob,
    _parse_pbs_datetime,
    _parse_pbs_walltime,
)


# ── Slurm time/date parsing ────────────────────────────────────


def test_parse_slurm_time_hhmmss() -> None:
    assert _parse_slurm_time("01:30:00") == 5400


def test_parse_slurm_time_with_days() -> None:
    assert _parse_slurm_time("1-02:00:00") == 93600


def test_parse_slurm_time_empty() -> None:
    assert _parse_slurm_time("") == 0
    assert _parse_slurm_time("Unknown") == 0


def test_parse_slurm_time_mmss() -> None:
    assert _parse_slurm_time("30:00") == 1800


def test_parse_slurm_datetime_valid() -> None:
    ts = _parse_slurm_datetime("2024-01-15T10:30:00")
    assert ts is not None
    assert ts > 0


def test_parse_slurm_datetime_invalid() -> None:
    assert _parse_slurm_datetime("") is None
    assert _parse_slurm_datetime("Unknown") is None


# ── PBS time/date parsing ───────────────────────────────────────


def test_parse_pbs_walltime_hhmmss() -> None:
    assert _parse_pbs_walltime("02:30:00") == 9000


def test_parse_pbs_walltime_mmss() -> None:
    assert _parse_pbs_walltime("30:00") == 1800


def test_parse_pbs_walltime_empty() -> None:
    assert _parse_pbs_walltime("") == 0


def test_parse_pbs_datetime_iso() -> None:
    ts = _parse_pbs_datetime("2024-01-15T10:30:00")
    assert ts is not None
    assert ts > 0


def test_parse_pbs_datetime_empty() -> None:
    assert _parse_pbs_datetime("") is None


# ── Slurm connector ────────────────────────────────────────────


def test_slurm_connector_init() -> None:
    """SlurmConnector initializes with defaults."""
    c = SlurmConnector()
    assert c.sacct_bin == "sacct"
    assert c.lookback_hours == 24


def test_slurm_connector_parse_sacct_output() -> None:
    """Parses sacct parsable2 output correctly."""
    # 17 fields: JobID|JobName|User|Group|Partition|Account|State|Submit|Start|End|Elapsed|Timelimit|NCPUS|NNodes|ReqMem|MaxRSS|ExitCode
    raw = "12345|test_job|alice|users|normal|default|COMPLETED|2024-01-15T10:00:00|2024-01-15T10:01:00|2024-01-15T11:01:00|01:00:00|02:00:00|4|1|4000Mn||0:0\n"
    c = SlurmConnector()
    jobs = c._parse_sacct_output(raw)
    assert len(jobs) == 1
    assert jobs[0].job_id == 12345
    assert jobs[0].job_name == "test_job"
    assert jobs[0].user == "alice"
    assert jobs[0].runtime_actual_sec == 3600
    assert jobs[0].requested_cpus == 4


def test_slurm_connector_skips_steps() -> None:
    """Job steps (e.g. 12345.batch) are skipped."""
    raw = (
        "12345|main|alice|users|normal|default|COMPLETED|2024-01-15T10:00:00|2024-01-15T10:01:00|2024-01-15T11:01:00|01:00:00|02:00:00|4|1|4000Mn||0:0\n"
        "12345.batch|batch|alice|users|normal|default|COMPLETED|2024-01-15T10:00:00|2024-01-15T10:01:00|2024-01-15T11:01:00|01:00:00|02:00:00|4|1|4000Mn||0:0\n"
    )
    c = SlurmConnector()
    jobs = c._parse_sacct_output(raw)
    assert len(jobs) == 1


def test_slurm_connector_jobs_to_dataframe() -> None:
    """Converts SlurmJob list to canonical DataFrame."""
    c = SlurmConnector()
    jobs = [
        SlurmJob(
            job_id=1,
            job_name="test",
            user="alice",
            group="users",
            partition="normal",
            state="COMPLETED",
            submit_ts=1000,
            start_ts=1010,
            end_ts=1110,
            runtime_actual_sec=100,
            runtime_requested_sec=200,
            requested_cpus=4,
            requested_nodes=1,
            exit_code="0:0",
        )
    ]
    df = c.jobs_to_dataframe(jobs)
    assert len(df) == 1
    assert df.iloc[0]["job_id"] == 1
    assert df.iloc[0]["requested_cpus"] == 4


def test_slurm_connector_sync_no_cluster() -> None:
    """Sync on non-Slurm machine returns graceful error."""
    c = SlurmConnector()
    result = c.sync()
    assert result.jobs_ingested == 0
    assert len(result.errors) > 0


# ── PBS connector ───────────────────────────────────────────────


def test_pbs_connector_init() -> None:
    """PBSConnector initializes with defaults."""
    c = PBSConnector()
    assert c.qstat_bin == "qstat"


def test_pbs_connector_parse_qstat_json() -> None:
    """Parses qstat JSON output correctly."""
    raw = '{"Jobs": {"123.server": {"Job_Name": "test", "Job_Owner": "alice@host", "queue": "batch", "job_state": "F", "Resource_List": {"ncpus": "8", "walltime": "01:00:00"}, "resources_used": {"walltime": "00:30:00"}, "ctime": "2024-01-15T10:00:00", "Exit_status": 0}}}'
    c = PBSConnector()
    jobs = c._parse_qstat_json(raw)
    assert len(jobs) == 1
    assert jobs[0].job_id == "123"
    assert jobs[0].user == "alice"
    assert jobs[0].requested_cpus == 8
    assert jobs[0].runtime_requested_sec == 3600
    assert jobs[0].runtime_actual_sec == 1800


def test_pbs_connector_parse_invalid_json() -> None:
    """Invalid JSON returns empty list."""
    c = PBSConnector()
    jobs = c._parse_qstat_json("not json")
    assert jobs == []


def test_pbs_connector_jobs_to_dataframe() -> None:
    """Converts PBSJob list to canonical DataFrame."""
    c = PBSConnector()
    jobs = [
        PBSJob(
            job_id="1",
            job_name="test",
            user="alice",
            group="users",
            queue="batch",
            state="F",
            submit_ts=1000,
            start_ts=1010,
            end_ts=1110,
            runtime_actual_sec=100,
            runtime_requested_sec=200,
            requested_cpus=4,
            requested_nodes=1,
            exit_status=0,
        )
    ]
    df = c.jobs_to_dataframe(jobs)
    assert len(df) == 1
    assert df.iloc[0]["requested_cpus"] == 4


def test_pbs_connector_sync_no_cluster() -> None:
    """Sync on non-PBS machine returns graceful error."""
    c = PBSConnector()
    result = c.sync()
    assert result.jobs_ingested == 0
    assert len(result.errors) > 0
