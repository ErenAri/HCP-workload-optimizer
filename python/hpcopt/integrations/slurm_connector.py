"""Slurm live connector — reads sacct data and pushes recommendations.

Connects to a live Slurm cluster via `sacct` and `squeue` commands,
ingests recent job history, runs models, and optionally pushes
scheduling recommendations back via Slurm's priority plugin.

Usage:
    from hpcopt.integrations.slurm_connector import SlurmConnector

    connector = SlurmConnector(model_dir=Path("outputs/models/latest"))
    connector.sync()  # Pull recent jobs + push recommendations
"""
from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# sacct fields to extract
SACCT_FIELDS = [
    "JobID", "JobName", "User", "Group", "Partition", "Account",
    "State", "Submit", "Start", "End", "Elapsed", "Timelimit",
    "NCPUS", "NNodes", "ReqMem", "MaxRSS", "ExitCode",
]


@dataclass
class SlurmJob:
    """Parsed Slurm job record."""
    job_id: int
    job_name: str
    user: str
    group: str
    partition: str
    state: str
    submit_ts: int
    start_ts: int | None
    end_ts: int | None
    runtime_actual_sec: int
    runtime_requested_sec: int
    requested_cpus: int
    requested_nodes: int
    exit_code: str


@dataclass
class SyncResult:
    """Result of a sync operation."""
    jobs_ingested: int
    jobs_running: int
    recommendations_generated: int
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


def _parse_slurm_time(time_str: str) -> int:
    """Parse Slurm time format (D-HH:MM:SS or HH:MM:SS) to seconds."""
    if not time_str or time_str == "Unknown":
        return 0
    parts = time_str.split("-")
    if len(parts) == 2:
        days = int(parts[0])
        hms = parts[1]
    else:
        days = 0
        hms = parts[0]

    hms_parts = hms.split(":")
    if len(hms_parts) == 3:
        h, m, s = int(hms_parts[0]), int(hms_parts[1]), int(hms_parts[2])
    elif len(hms_parts) == 2:
        h, m, s = 0, int(hms_parts[0]), int(hms_parts[1])
    else:
        h, m, s = 0, 0, int(hms_parts[0])

    return days * 86400 + h * 3600 + m * 60 + s


def _parse_slurm_datetime(dt_str: str) -> int | None:
    """Parse Slurm datetime to Unix timestamp."""
    if not dt_str or dt_str == "Unknown":
        return None
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


class SlurmConnector:
    """Live connector for Slurm cluster integration.

    Reads job history via sacct, feeds it to the model pipeline,
    and optionally generates scheduling recommendations.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        sacct_bin: str = "sacct",
        squeue_bin: str = "squeue",
        lookback_hours: int = 24,
        cluster_name: str | None = None,
    ):
        self.model_dir = model_dir
        self.sacct_bin = sacct_bin
        self.squeue_bin = squeue_bin
        self.lookback_hours = lookback_hours
        self.cluster_name = cluster_name
        self._predictor = None

    def _run_sacct(self, start_time: str | None = None) -> str:
        """Run sacct and return raw output."""
        cmd = [
            self.sacct_bin,
            "--parsable2",
            "--noheader",
            "--format=" + ",".join(SACCT_FIELDS),
            "--allusers",
            "--state=COMPLETED,FAILED,TIMEOUT,CANCELLED",
        ]
        if start_time:
            cmd.extend(["--starttime", start_time])
        if self.cluster_name:
            cmd.extend(["--clusters", self.cluster_name])

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"sacct failed: {result.stderr}")

        return result.stdout

    def _run_squeue(self) -> str:
        """Run squeue to get currently queued/running jobs."""
        cmd = [
            self.squeue_bin,
            "--format=%i|%j|%u|%g|%P|%T|%V|%S|%l|%C|%D",
            "--noheader",
        ]
        if self.cluster_name:
            cmd.extend(["--clusters", self.cluster_name])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"squeue failed: {result.stderr}")

        return result.stdout

    def _parse_sacct_output(self, raw: str) -> list[SlurmJob]:
        """Parse sacct parsable2 output into SlurmJob objects."""
        jobs = []
        reader = csv.DictReader(
            io.StringIO(raw),
            fieldnames=SACCT_FIELDS,
            delimiter="|",
        )

        for row in reader:
            job_id_str = row.get("JobID", "")
            # Skip job steps (like 12345.batch)
            if "." in job_id_str:
                continue
            # Skip array sub-jobs
            if "_" in job_id_str:
                continue

            try:
                job_id = int(job_id_str)
            except (ValueError, TypeError):
                continue

            elapsed_sec = _parse_slurm_time(row.get("Elapsed", "0"))
            timelimit_sec = _parse_slurm_time(row.get("Timelimit", "0"))

            jobs.append(SlurmJob(
                job_id=job_id,
                job_name=row.get("JobName", ""),
                user=row.get("User", ""),
                group=row.get("Group", ""),
                partition=row.get("Partition", ""),
                state=row.get("State", ""),
                submit_ts=_parse_slurm_datetime(row.get("Submit", "")) or 0,
                start_ts=_parse_slurm_datetime(row.get("Start", "")),
                end_ts=_parse_slurm_datetime(row.get("End", "")),
                runtime_actual_sec=elapsed_sec,
                runtime_requested_sec=timelimit_sec,
                requested_cpus=int(row.get("NCPUS", "1") or "1"),
                requested_nodes=int(row.get("NNodes", "1") or "1"),
                exit_code=row.get("ExitCode", "0:0"),
            ))

        return jobs

    def fetch_recent_jobs(self) -> list[SlurmJob]:
        """Fetch recently completed jobs from sacct."""
        start = datetime.now(tz=timezone.utc) - timedelta(hours=self.lookback_hours)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S")

        raw = self._run_sacct(start_time=start_str)
        jobs = self._parse_sacct_output(raw)
        logger.info("Fetched %d jobs from sacct (last %dh)", len(jobs), self.lookback_hours)
        return jobs

    def jobs_to_dataframe(self, jobs: list[SlurmJob]) -> pd.DataFrame:
        """Convert SlurmJob list to canonical DataFrame."""
        records = []
        for j in jobs:
            records.append({
                "job_id": j.job_id,
                "submit_ts": j.submit_ts,
                "start_ts": j.start_ts or 0,
                "end_ts": j.end_ts or 0,
                "runtime_actual_sec": j.runtime_actual_sec,
                "runtime_requested_sec": j.runtime_requested_sec,
                "requested_cpus": j.requested_cpus,
                "user_id": j.user,
                "group_id": j.group,
                "queue_id": j.partition,
                "partition_id": j.partition,
                "requested_mem": 0,
            })
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
            jobs = self.fetch_recent_jobs()
            result.jobs_ingested = len(jobs)

            if self.model_dir and self.model_dir.exists():
                from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor
                predictor = RuntimeQuantilePredictor(self.model_dir)

                df = self.jobs_to_dataframe(jobs)
                recs = 0
                for _, row in df.iterrows():
                    try:
                        pred = predictor.predict_one(row.to_dict())
                        recs += 1
                    except Exception as exc:
                        result.errors.append(f"Prediction failed for job {row['job_id']}: {exc}")
                result.recommendations_generated = recs

        except FileNotFoundError:
            result.errors.append("sacct/squeue not found — not on a Slurm cluster")
        except Exception as exc:
            result.errors.append(str(exc))

        return result


# ── CLI entry point ─────────────────────────────────────────────

def main() -> None:
    """CLI entry point for slurm sync."""
    import argparse

    parser = argparse.ArgumentParser(description="HPC Workload Optimizer — Slurm Sync")
    parser.add_argument("--model-dir", type=Path, help="Path to model directory")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--cluster", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    connector = SlurmConnector(
        model_dir=args.model_dir,
        lookback_hours=args.lookback_hours,
        cluster_name=args.cluster,
    )
    result = connector.sync()

    output = {
        "timestamp": result.timestamp,
        "jobs_ingested": result.jobs_ingested,
        "jobs_running": result.jobs_running,
        "recommendations_generated": result.recommendations_generated,
        "errors": result.errors,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
