from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hpcopt.simulate.core import build_observed_queue_series
from hpcopt.simulate.metrics import compute_job_metrics
from hpcopt.simulate.objective import compute_objective_contract_metrics
from hpcopt.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class BatsimConfigResult:
    config_path: Path
    run_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class BatsimInvokeResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class BatsimNormalizationResult:
    run_id: str
    policy_id: str
    jobs_csv_path: Path
    jobs_artifact_path: Path
    queue_artifact_path: Path
    sim_report_path: Path
    invariant_report_path: Path
    metrics: dict[str, float]
    objective_metrics: dict[str, float]
    fallback_accounting: dict[str, Any]


_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
SUPPORTED_EDC_MODES = {"library_file", "library_str", "socket_file", "socket_str"}


def windows_path_to_wsl(path: Path | str) -> str:
    raw = str(path)
    if not _WINDOWS_ABS_PATH_RE.match(raw):
        return raw.replace("\\", "/")
    drive = raw[0].lower()
    tail = raw[2:].replace("\\", "/")
    return f"/mnt/{drive}{tail}"


def _coerce_positive_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed) or parsed <= 0:
        return fallback
    return parsed


def _coerce_positive_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if parsed <= 0:
        return fallback
    return parsed


def _runtime_seconds(row: pd.Series) -> float:
    runtime_actual = _coerce_positive_float(row.get("runtime_actual_sec"), fallback=0.0)
    if runtime_actual > 0:
        return runtime_actual
    runtime_requested = _coerce_positive_float(row.get("runtime_requested_sec"), fallback=0.0)
    if runtime_requested > 0:
        return runtime_requested
    return 1.0


def _requested_resources(row: pd.Series, capacity_cpus: int) -> int:
    requested = _coerce_positive_int(row.get("requested_cpus"), fallback=0)
    if requested == 0:
        requested = _coerce_positive_int(row.get("allocated_cpus"), fallback=1)
    return max(1, min(requested, capacity_cpus))


def _infer_capacity(trace_df: pd.DataFrame) -> int:
    if trace_df.empty:
        return 1
    requested_peak = pd.to_numeric(trace_df.get("requested_cpus"), errors="coerce")
    allocated_peak = pd.to_numeric(trace_df.get("allocated_cpus"), errors="coerce")
    peak = pd.concat([requested_peak, allocated_peak], axis=0).max(skipna=True)
    if pd.isna(peak):
        return 64
    return max(1, int(peak))


def _sanitize_profile_id(delay_sec: float) -> str:
    delay_ms = max(1, int(round(delay_sec * 1000.0)))
    return f"delay_ms_{delay_ms}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def generate_batsim_workload_from_trace(
    trace_dataset: Path,
    out_path: Path,
    capacity_cpus: int,
) -> Path:
    trace_df = pd.read_parquet(trace_dataset)
    if trace_df.empty:
        raise ValueError("trace dataset is empty; cannot generate batsim workload")

    ordered = trace_df.sort_values(by=["submit_ts", "job_id"], kind="mergesort").reset_index(drop=True)
    jobs: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}

    for idx, row in ordered.iterrows():
        subtime = _coerce_positive_float(row.get("submit_ts"), fallback=0.0)
        res = _requested_resources(row=row, capacity_cpus=capacity_cpus)
        delay_sec = _runtime_seconds(row=row)
        profile_id = _sanitize_profile_id(delay_sec)
        if profile_id not in profiles:
            profiles[profile_id] = {
                "type": "DelayProfile",
                "delay": delay_sec,
            }

        job_id = row.get("job_id")
        if pd.isna(job_id):
            job_name = str(idx + 1)
        else:
            try:
                job_id_num = float(job_id)
                job_name = str(int(job_id_num)) if job_id_num.is_integer() else str(job_id)
            except (TypeError, ValueError):
                job_name = str(job_id)

        job_payload: dict[str, Any] = {
            "id": job_name,
            "subtime": subtime,
            "res": res,
            "profile": profile_id,
        }
        walltime = _coerce_positive_float(row.get("runtime_requested_sec"), fallback=0.0)
        if walltime > 0:
            job_payload["walltime"] = walltime

        extra_data = {
            "user_id": None if pd.isna(row.get("user_id")) else int(row.get("user_id")),
            "queue_id": None if pd.isna(row.get("queue_id")) else int(row.get("queue_id")),
            "group_id": None if pd.isna(row.get("group_id")) else int(row.get("group_id")),
            "partition_id": None if pd.isna(row.get("partition_id")) else int(row.get("partition_id")),
        }
        job_payload["extra_data"] = json.dumps(extra_data, sort_keys=True)
        jobs.append(job_payload)

    workload_payload = {
        "description": "Auto-generated from canonical parquet trace by hpcopt",
        "source_trace": str(trace_dataset),
        "nb_res": capacity_cpus,
        "jobs": jobs,
        "profiles": profiles,
    }
    write_json(out_path, workload_payload)
    return out_path


def generate_simple_platform_xml(platform_path: Path, capacity_cpus: int) -> Path:
    if capacity_cpus <= 0:
        raise ValueError("capacity_cpus must be positive")

    radical = f"0-{capacity_cpus - 1}"
    content = (
        "<?xml version='1.0'?>\n"
        '<!DOCTYPE platform SYSTEM "http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd">\n'
        '<platform version="4.1">\n'
        '  <zone id="AS0" routing="Full">\n'
        "    <cluster\n"
        '      id="A" prefix="a" suffix="" '
        f'radical="{radical}"\n'
        '      speed="1Gf"\n'
        '      bw="1GBps" lat="5us"\n'
        '      bb_bw="3GBps" bb_lat="3us"\n'
        "    />\n\n"
        "    <cluster\n"
        '      id="M" prefix="m" suffix="" radical="0-0"\n'
        '      speed="1Gf" bw="1GBps" lat="5us"\n'
        '      bb_bw="3GBps" bb_lat="3us"\n'
        "    >\n"
        '      <prop id="role" value="master" />\n'
        "    </cluster>\n\n"
        '    <link id="backbone" bandwidth="5GBps" latency="2us" />\n\n'
        '    <zoneRoute src="A" dst="M" gw_src="aA_router" gw_dst="mM_router">\n'
        '      <link_ctn id="backbone" />\n'
        "    </zoneRoute>\n"
        "  </zone>\n"
        "</platform>\n"
    )
    ensure_dir(platform_path.parent)
    platform_path.write_text(content, encoding="utf-8")
    return platform_path


def _resolve_default_fcfs_library_path(wsl_distro: str) -> str:
    if shutil.which("wsl") is None:
        return str((Path.home() / ".nix-profile" / "lib" / "libfcfs.so").resolve())
    proc = subprocess.run(
        ["wsl", "-d", wsl_distro, "--", "bash", "-lc", 'printf "%s" "$HOME/.nix-profile/lib/libfcfs.so"'],
        capture_output=True,
        text=True,
        check=False,
    )
    candidate = proc.stdout.strip()
    if proc.returncode == 0 and candidate:
        return candidate
    return "$HOME/.nix-profile/lib/libfcfs.so"


def _resolve_workload_path(
    trace_dataset: Path,
    workload_path: Path | None,
    run_artifact_dir: Path,
    run_id: str,
    capacity_cpus: int,
) -> tuple[Path, str]:
    if workload_path is not None:
        return workload_path.resolve(), "provided"

    if trace_dataset.suffix.lower() == ".json":
        return trace_dataset.resolve(), "trace_json"

    generated = run_artifact_dir / f"{run_id}_workload.json"
    generate_batsim_workload_from_trace(trace_dataset=trace_dataset, out_path=generated, capacity_cpus=capacity_cpus)
    return generated.resolve(), "generated_from_trace"


def _build_edc_args(
    edc_mode: str,
    edc_library_path: str | None,
    edc_socket_endpoint: str | None,
    edc_init_file: Path,
    edc_init_inline: str,
) -> list[str]:
    if edc_mode not in SUPPORTED_EDC_MODES:
        raise ValueError(f"unsupported edc_mode '{edc_mode}'")

    if edc_mode in {"library_file", "library_str"} and not edc_library_path:
        raise ValueError("edc_library_path is required for library EDC modes")
    if edc_mode in {"socket_file", "socket_str"} and not edc_socket_endpoint:
        raise ValueError("edc_socket_endpoint is required for socket EDC modes")

    if edc_mode == "library_file":
        return ["--edc-library-file", str(edc_library_path), str(edc_init_file)]
    if edc_mode == "library_str":
        return ["--edc-library-str", str(edc_library_path), edc_init_inline]
    if edc_mode == "socket_file":
        return ["--edc-socket-file", str(edc_socket_endpoint), str(edc_init_file)]
    return ["--edc-socket-str", str(edc_socket_endpoint), edc_init_inline]


def _extract_cli_arg_value(cli_args: list[Any], flag: str) -> str | None:
    for idx, arg in enumerate(cli_args[:-1]):
        if str(arg) == flag:
            return str(cli_args[idx + 1])
    return None


def _wsl_path_to_windows(raw: str) -> str:
    # Convert '/mnt/c/path/to/file' to 'C:\\path\\to\\file' for local file IO on Windows.
    parts = raw.split("/")
    if len(parts) >= 4 and parts[0] == "" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        tail = "\\".join(parts[3:])
        return f"{drive}:\\{tail}"
    return raw


def _resolve_local_path(raw: str) -> Path:
    candidate = os.path.expandvars(os.path.expanduser(raw))
    path = Path(candidate)
    if path.exists():
        return path
    wsl_converted = Path(_wsl_path_to_windows(raw))
    return wsl_converted


def _parse_job_id(raw: Any, fallback: int) -> int:
    text = str(raw)
    if "!" in text:
        text = text.split("!")[-1]
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return fallback
    return parsed


def _to_int_ts(value: Any, fallback: int = 0) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return max(0, int(math.floor(parsed)))


def _load_workload_job_metadata(workload_path: Path | None) -> dict[str, dict[str, Any]]:
    if workload_path is None or not workload_path.exists():
        return {}
    try:
        payload = json.loads(workload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("id"))
        extra_raw = job.get("extra_data")
        extra = _parse_json_object(extra_raw) if isinstance(extra_raw, str) else {}
        metadata[job_id] = extra
    return metadata


def _build_jobs_df_from_batsim_csv(
    jobs_csv_path: Path,
    workload_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    jobs_csv = pd.read_csv(jobs_csv_path)
    required = {
        "job_id",
        "submission_time",
        "starting_time",
        "finish_time",
        "execution_time",
        "requested_number_of_resources",
    }
    missing = required - set(jobs_csv.columns)
    if missing:
        raise ValueError(f"Batsim jobs.csv missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for idx, row in jobs_csv.iterrows():
        fallback_job_id = 1_000_000 + idx
        raw_id = row.get("job_id")
        parsed_id = _parse_job_id(raw_id, fallback=fallback_job_id)
        raw_id_text = str(raw_id)
        stripped_id_text = raw_id_text.split("!")[-1]
        extra = workload_metadata.get(stripped_id_text, {})

        submit_ts = _to_int_ts(row.get("submission_time"), fallback=0)
        start_ts = _to_int_ts(row.get("starting_time"), fallback=submit_ts)
        finish_ts = _to_int_ts(row.get("finish_time"), fallback=start_ts)
        finish_ts = max(finish_ts, start_ts)
        runtime_actual_sec = max(
            finish_ts - start_ts,
            _to_int_ts(row.get("execution_time"), fallback=0),
        )

        requested_cpus = max(1, _coerce_positive_int(row.get("requested_number_of_resources"), fallback=1))
        requested_time = pd.to_numeric(row.get("requested_time"), errors="coerce")
        runtime_requested_sec = (
            float(requested_time) if pd.notna(requested_time) and float(requested_time) > 0 else None
        )

        rows.append(
            {
                "job_id": int(parsed_id),
                "submit_ts": int(submit_ts),
                "start_ts": int(start_ts),
                "end_ts": int(start_ts + runtime_actual_sec),
                "runtime_actual_sec": int(runtime_actual_sec),
                "runtime_requested_sec": runtime_requested_sec,
                "requested_cpus": int(requested_cpus),
                "user_id": extra.get("user_id"),
                "group_id": extra.get("group_id"),
                "queue_id": extra.get("queue_id"),
                "partition_id": extra.get("partition_id"),
            }
        )

    jobs_df = pd.DataFrame(rows)
    if jobs_df.empty:
        raise ValueError("Batsim jobs.csv produced no jobs")
    jobs_df = jobs_df.sort_values(["submit_ts", "job_id"]).reset_index(drop=True)
    return jobs_df


def _invariant_report_from_jobs(
    run_id: str,
    jobs_df: pd.DataFrame,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for row in jobs_df.itertuples(index=False):
        failed: list[str] = []
        if int(row.start_ts) < int(row.submit_ts):
            failed.append("job_start_before_submit")
        if int(row.end_ts) < int(row.start_ts):
            failed.append("job_end_before_start")
        if int(row.requested_cpus) <= 0:
            failed.append("job_nonpositive_cpu")
        if failed:
            violations.append(
                {
                    "job_id": int(row.job_id),
                    "failed_invariants": failed,
                    "severity": "error",
                }
            )

    return {
        "run_id": run_id,
        "strict_mode": False,
        "step_count": int(len(jobs_df)),
        "violations": violations,
        "source": "batsim_external_output",
    }


def normalize_batsim_run_outputs(
    config_path: Path,
    report_out_dir: Path,
    simulation_out_dir: Path,
    starvation_wait_cap_sec: int = 172800,
) -> BatsimNormalizationResult:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", config_path.stem))
    policy_id = str(payload.get("policy", {}).get("policy_id", "BATSIM_EXTERNAL"))
    policy_slug = policy_id.lower()
    capacity_cpus = _coerce_positive_int(payload.get("resources", {}).get("capacity_cpus"), fallback=1)

    cli_args = payload.get("batsim_cli", {}).get("args")
    if not isinstance(cli_args, list):
        raise ValueError("run config missing batsim_cli.args")
    export_prefix = _extract_cli_arg_value(cli_args, "--export")
    if export_prefix is None:
        raise ValueError("run config missing --export in batsim_cli.args")
    jobs_csv_path = _resolve_local_path(f"{export_prefix}jobs.csv")
    if not jobs_csv_path.exists():
        raise FileNotFoundError(f"batsim jobs.csv not found at {jobs_csv_path}")

    workload_path_raw = payload.get("inputs", {}).get("workload_path")
    workload_path = _resolve_local_path(str(workload_path_raw)) if workload_path_raw else None
    workload_metadata = _load_workload_job_metadata(workload_path)

    jobs_df = _build_jobs_df_from_batsim_csv(
        jobs_csv_path=jobs_csv_path,
        workload_metadata=workload_metadata,
    )
    queue_df = build_observed_queue_series(jobs_df)
    metrics = compute_job_metrics(jobs_df, capacity_cpus=capacity_cpus)
    objective_metrics = compute_objective_contract_metrics(
        jobs_df=jobs_df,
        capacity_cpus=capacity_cpus,
        starvation_wait_cap_sec=starvation_wait_cap_sec,
    )

    jobs_artifact_path = ensure_dir(simulation_out_dir) / f"{run_id}_{policy_slug}_jobs.parquet"
    queue_artifact_path = ensure_dir(simulation_out_dir) / f"{run_id}_{policy_slug}_queue.parquet"
    jobs_df.to_parquet(jobs_artifact_path, index=False)
    queue_df.to_parquet(queue_artifact_path, index=False)

    total_jobs = int(len(jobs_df))
    denominator = total_jobs if total_jobs > 0 else 1
    fallback_accounting = {
        "prediction_used_count": 0,
        "requested_fallback_count": total_jobs,
        "actual_fallback_count": 0,
        "prediction_used_rate": 0.0,
        "requested_fallback_rate": float(total_jobs / denominator),
        "actual_fallback_rate": 0.0,
        "total_scheduled_jobs": total_jobs,
        "runtime_guard_k": None,
        "strict_uncertainty_mode": False,
        "source": "batsim_external_edc",
    }

    invariant_report = _invariant_report_from_jobs(run_id=run_id, jobs_df=jobs_df)
    invariant_report_path = ensure_dir(report_out_dir) / f"{run_id}_{policy_slug}_invariants.json"
    write_json(invariant_report_path, invariant_report)

    sim_report_path = ensure_dir(report_out_dir) / f"{run_id}_{policy_slug}_sim_report.json"
    write_json(
        sim_report_path,
        {
            "run_id": run_id,
            "policy_id": policy_id,
            "status": "ok",
            "metrics": metrics,
            "objective_metrics": objective_metrics,
            "fallback_accounting": fallback_accounting,
            "jobs_artifact": str(jobs_artifact_path),
            "queue_artifact": str(queue_artifact_path),
            "invariant_report": str(invariant_report_path),
            "source": {
                "engine": "batsim",
                "jobs_csv": str(jobs_csv_path),
                "workload_path": str(workload_path) if workload_path else None,
            },
        },
    )

    return BatsimNormalizationResult(
        run_id=run_id,
        policy_id=policy_id,
        jobs_csv_path=jobs_csv_path,
        jobs_artifact_path=jobs_artifact_path,
        queue_artifact_path=queue_artifact_path,
        sim_report_path=sim_report_path,
        invariant_report_path=invariant_report_path,
        metrics=metrics,
        objective_metrics=objective_metrics,
        fallback_accounting=fallback_accounting,
    )


def invoke_batsim_run(
    config_path: Path,
    batsim_bin: str = "batsim",
    dry_run: bool = True,
    use_wsl: bool = False,
    wsl_distro: str = "Ubuntu",
    wsl_load_nix_profile: bool = True,
) -> BatsimInvokeResult:
    resolved_config = config_path.resolve()

    try:
        payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BatsimInvokeResult(
            command=[batsim_bin],
            returncode=2,
            stdout="",
            stderr=f"invalid run config: {exc}",
            status="failed",
            reason="invalid_batsim_run_config",
        )

    cli_args = payload.get("batsim_cli", {}).get("args")
    if not isinstance(cli_args, list) or not cli_args:
        return BatsimInvokeResult(
            command=[batsim_bin],
            returncode=2,
            stdout="",
            stderr="invalid run config: missing batsim_cli.args",
            status="failed",
            reason="invalid_batsim_run_config",
        )

    if use_wsl:
        if shutil.which("wsl") is None:
            return BatsimInvokeResult(
                command=["wsl", "-d", wsl_distro, "--", "bash", "-lc", ""],
                returncode=127,
                stdout="",
                stderr="wsl command not found",
                status="failed",
                reason="missing_wsl_runtime",
            )
        wsl_home = None
        if any(isinstance(arg, str) and arg.startswith("$HOME/") for arg in cli_args):
            home_proc = subprocess.run(
                ["wsl", "-d", wsl_distro, "--", "bash", "-lc", 'printf "%s" "$HOME"'],
                capture_output=True,
                text=True,
                check=False,
            )
            if home_proc.returncode == 0:
                wsl_home = home_proc.stdout.strip()

        def _to_wsl_arg(raw: Any) -> str:
            arg = str(raw)
            if arg.startswith("$HOME/") and wsl_home:
                arg = f"{wsl_home}/{arg[len('$HOME/') :]}"
            if arg.startswith("ipc://"):
                endpoint = arg[len("ipc://") :]
                if _WINDOWS_ABS_PATH_RE.match(endpoint):
                    return f"ipc://{windows_path_to_wsl(endpoint)}"
                return arg
            if _WINDOWS_ABS_PATH_RE.match(arg):
                return windows_path_to_wsl(arg)
            return arg.replace("\\", "/")

        batsim_q = shlex.quote(str(batsim_bin))
        args_q = " ".join(shlex.quote(_to_wsl_arg(a)) for a in cli_args)
        prefix = ". ~/.nix-profile/etc/profile.d/nix.sh >/dev/null 2>&1 || true; " if wsl_load_nix_profile else ""
        shell_cmd = f"{prefix}{batsim_q} {args_q}".strip()
        command = ["wsl", "-d", wsl_distro, "--", "bash", "-lc", shell_cmd]
    else:
        command = [batsim_bin, *[os.path.expandvars(os.path.expanduser(str(a))) for a in cli_args]]

    if dry_run:
        return BatsimInvokeResult(
            command=command,
            returncode=0,
            stdout="",
            stderr="",
            status="dry_run",
            reason=None,
        )

    if not use_wsl and shutil.which(batsim_bin) is None:
        return BatsimInvokeResult(
            command=command,
            returncode=127,
            stdout="",
            stderr=f"batsim binary '{batsim_bin}' not found",
            status="failed",
            reason="missing_batsim_binary",
        )

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    status = "ok" if proc.returncode == 0 else "failed"
    return BatsimInvokeResult(
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        status=status,
        reason=None if proc.returncode == 0 else "batsim_exit_nonzero",
    )


def build_batsim_run_config(
    run_id: str,
    trace_dataset: Path,
    policy_id: str,
    out_dir: Path,
    platform_path: Path | None = None,
    workload_path: Path | None = None,
    scheduler_adapter: str = "rust_sim_runner_adapter_contract",
    capacity_cpus: int | None = None,
    edc_mode: str = "library_file",
    edc_library_path: str | None = None,
    edc_socket_endpoint: str | None = None,
    edc_init_json: str = "{}",
    export_prefix: Path | None = None,
    use_wsl_defaults: bool = True,
    wsl_distro: str = "Ubuntu",
    extra: dict[str, Any] | None = None,
) -> BatsimConfigResult:
    ensure_dir(out_dir)
    run_artifact_dir = ensure_dir(out_dir / run_id)
    trace_path = trace_dataset.resolve()

    if capacity_cpus is None:
        if trace_path.suffix.lower() == ".json":
            try:
                trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                trace_payload = {}
            capacity_cpus = _coerce_positive_int(trace_payload.get("nb_res"), fallback=64)
        else:
            trace_df = pd.read_parquet(trace_path)
            capacity_cpus = _infer_capacity(trace_df)
    capacity_cpus = _coerce_positive_int(capacity_cpus, fallback=64)

    resolved_workload, workload_source = _resolve_workload_path(
        trace_dataset=trace_path,
        workload_path=workload_path,
        run_artifact_dir=run_artifact_dir,
        run_id=run_id,
        capacity_cpus=capacity_cpus,
    )

    if platform_path is not None:
        resolved_platform = platform_path.resolve()
        platform_source = "provided"
    else:
        resolved_platform = generate_simple_platform_xml(
            platform_path=run_artifact_dir / f"{run_id}_platform.xml",
            capacity_cpus=capacity_cpus,
        ).resolve()
        platform_source = "generated"

    edc_init_obj = _parse_json_object(edc_init_json)
    edc_init_path = run_artifact_dir / f"{run_id}_edc_init.json"
    write_json(edc_init_path, edc_init_obj)
    edc_init_inline = json.dumps(edc_init_obj, separators=(",", ":"), sort_keys=True)

    resolved_edc_library = edc_library_path
    if edc_mode in {"library_file", "library_str"} and resolved_edc_library is None:
        resolved_edc_library = (
            _resolve_default_fcfs_library_path(wsl_distro=wsl_distro)
            if use_wsl_defaults
            else "$HOME/.nix-profile/lib/libfcfs.so"
        )

    if edc_mode in {"socket_file", "socket_str"} and edc_socket_endpoint is None:
        edc_socket_endpoint = f"ipc://{run_artifact_dir / 'sock'}"

    resolved_export_prefix = export_prefix.resolve() if export_prefix else (run_artifact_dir / "batout_").resolve()
    ensure_dir(resolved_export_prefix.parent)

    batsim_args = [
        "--export",
        str(resolved_export_prefix),
        "--platform",
        str(resolved_platform),
        "--workload",
        str(resolved_workload),
    ]
    batsim_args.extend(
        _build_edc_args(
            edc_mode=edc_mode,
            edc_library_path=resolved_edc_library,
            edc_socket_endpoint=edc_socket_endpoint,
            edc_init_file=edc_init_path.resolve(),
            edc_init_inline=edc_init_inline,
        )
    )

    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "engine": "batsim",
        "inputs": {
            "trace_dataset": str(trace_path),
            "workload_path": str(resolved_workload),
            "workload_source": workload_source,
            "platform_path": str(resolved_platform),
            "platform_source": platform_source,
            "edc_init_file": str(edc_init_path.resolve()),
        },
        "policy": {
            "policy_id": policy_id,
            "scheduler_adapter": scheduler_adapter,
        },
        "scheduler_adapter_contract": {
            "snapshot_schema": "schemas/adapter_snapshot.schema.json",
            "decision_schema": "schemas/adapter_decision.schema.json",
            "decision_module_ownership": "owned_by_hpcopt",
            "event_engine_ownership": "owned_by_batsim",
        },
        "resources": {
            "capacity_cpus": capacity_cpus,
        },
        "edc": {
            "mode": edc_mode,
            "library_path": resolved_edc_library,
            "socket_endpoint": edc_socket_endpoint,
            "init_inline": edc_init_inline,
        },
        "batsim_cli": {
            "args": batsim_args,
        },
        "extra": extra or {},
    }
    config_path = out_dir / f"{run_id}_batsim_run_config.json"
    write_json(config_path, payload)
    return BatsimConfigResult(config_path=config_path, run_id=run_id, payload=payload)
