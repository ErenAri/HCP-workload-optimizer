"""PBS Pro / Torque live connector.

Reads qstat and tracejob data from a PBS/Torque cluster, converts to
canonical format, and runs predictions.

Usage:
    from hpcopt.integrations.pbs_connector import PBSConnector
    connector = PBSConnector(model_dir=Path("outputs/models/latest"))
    result = connector.sync()
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PBSJob:
    """Parsed PBS/Torque job record."""

    job_id: str
    job_name: str
    user: str
    group: str
    queue: str
    state: str
    submit_ts: int
    start_ts: int | None
    end_ts: int | None
    runtime_actual_sec: int
    runtime_requested_sec: int
    requested_cpus: int
    requested_nodes: int
    exit_status: int


@dataclass
class SyncResult:
    """Result of a sync operation."""

    jobs_ingested: int
    jobs_running: int
    recommendations_generated: int
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


def _parse_pbs_walltime(wt: str) -> int:
    """Parse PBS walltime (HH:MM:SS) to seconds."""
    if not wt:
        return 0
    parts = wt.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def _parse_pbs_datetime(dt_str: str) -> int | None:
    """Parse PBS datetime formats to Unix timestamp."""
    if not dt_str:
        return None
    for fmt in ["%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S"]:
        try:
            dt = datetime.strptime(dt_str.strip(), fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


class PBSConnector:
    """Live connector for PBS Pro / Torque cluster integration."""

    def __init__(
        self,
        model_dir: Path | None = None,
        qstat_bin: str = "qstat",
        tracejob_bin: str = "tracejob",
        server_name: str | None = None,
    ):
        self.model_dir = model_dir
        self.qstat_bin = qstat_bin
        self.tracejob_bin = tracejob_bin
        self.server_name = server_name

    def _run_qstat(self) -> str:
        """Run qstat -f to get detailed job info."""
        cmd = [self.qstat_bin, "-f", "-F", "json"]
        if self.server_name:
            cmd.extend(["@" + self.server_name])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError("qstat failed: " + result.stderr)
        return result.stdout

    def _run_qstat_completed(self) -> str:
        """Run qstat for completed jobs (PBS Pro specific)."""
        cmd = [self.qstat_bin, "-x", "-f", "-F", "json"]
        if self.server_name:
            cmd.extend(["@" + self.server_name])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError("qstat -x failed: " + result.stderr)
        return result.stdout

    def _parse_qstat_json(self, raw: str) -> list[PBSJob]:
        """Parse qstat JSON output."""
        jobs: list[PBSJob] = []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse qstat JSON output")
            return jobs

        jobs_data = data.get("Jobs", {})
        for job_id, info in jobs_data.items():
            resources = info.get("Resource_List", {})
            resources_used = info.get("resources_used", {})

            walltime_req = _parse_pbs_walltime(resources.get("walltime", ""))
            walltime_actual = _parse_pbs_walltime(resources_used.get("walltime", ""))

            ncpus_str = str(resources.get("ncpus", "1"))
            ncpus = int(ncpus_str) if ncpus_str.isdigit() else 1

            # Parse select for nodes
            select = resources.get("select", "")
            nodes = 1
            if select:
                match = re.match(r"(\d+)", select)
                if match:
                    nodes = int(match.group(1))

            jobs.append(
                PBSJob(
                    job_id=job_id.split(".")[0],  # strip server suffix
                    job_name=info.get("Job_Name", ""),
                    user=info.get("Job_Owner", "").split("@")[0],
                    group=info.get("group_list", ""),
                    queue=info.get("queue", ""),
                    state=info.get("job_state", ""),
                    submit_ts=_parse_pbs_datetime(info.get("ctime", "")) or 0,
                    start_ts=_parse_pbs_datetime(info.get("stime", "")),
                    end_ts=_parse_pbs_datetime(info.get("mtime", "")),
                    runtime_actual_sec=walltime_actual,
                    runtime_requested_sec=walltime_req,
                    requested_cpus=ncpus,
                    requested_nodes=nodes,
                    exit_status=int(info.get("Exit_status", 0) or 0),
                )
            )

        return jobs

    def jobs_to_dataframe(self, jobs: list[PBSJob]) -> pd.DataFrame:
        """Convert PBSJob list to canonical DataFrame."""
        records = []
        for j in jobs:
            records.append(
                {
                    "job_id": j.job_id,
                    "submit_ts": j.submit_ts,
                    "start_ts": j.start_ts or 0,
                    "end_ts": j.end_ts or 0,
                    "runtime_actual_sec": j.runtime_actual_sec,
                    "runtime_requested_sec": j.runtime_requested_sec,
                    "requested_cpus": j.requested_cpus,
                    "user_id": j.user,
                    "group_id": j.group,
                    "queue_id": j.queue,
                    "partition_id": j.queue,
                    "requested_mem": 0,
                }
            )
        return pd.DataFrame(records)

    def sync(self) -> SyncResult:
        """Full sync: fetch jobs, run predictions, return results."""
        result = SyncResult(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            jobs_ingested=0,
            jobs_running=0,
            recommendations_generated=0,
        )

        try:
            raw = self._run_qstat_completed()
            jobs = self._parse_qstat_json(raw)
            result.jobs_ingested = len(jobs)
            result.jobs_running = sum(1 for j in jobs if j.state in ("R", "E"))

            if self.model_dir and self.model_dir.exists():
                from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor

                predictor = RuntimeQuantilePredictor(self.model_dir)
                df = self.jobs_to_dataframe(jobs)
                for _, row in df.iterrows():
                    try:
                        predictor.predict_one(row.to_dict())
                        result.recommendations_generated += 1
                    except Exception as exc:
                        result.errors.append(str(exc))

        except FileNotFoundError:
            result.errors.append("qstat not found — not on a PBS cluster")
        except Exception as exc:
            result.errors.append(str(exc))

        return result
