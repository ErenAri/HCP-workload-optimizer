import json
from pathlib import Path

import pandas as pd
from hpcopt.simulate.batsim import (
    build_batsim_run_config,
    invoke_batsim_run,
    normalize_batsim_run_outputs,
    windows_path_to_wsl,
)
from hpcopt.utils.io import write_json


def _sample_trace_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0.0,
                "runtime_actual_sec": 10.0,
                "runtime_requested_sec": 20.0,
                "requested_cpus": 2,
                "allocated_cpus": 2,
                "user_id": 10,
                "queue_id": 1,
                "group_id": 1,
                "partition_id": 1,
            },
            {
                "job_id": 2,
                "submit_ts": 5.0,
                "runtime_actual_sec": 3.0,
                "runtime_requested_sec": 6.0,
                "requested_cpus": 1,
                "allocated_cpus": 1,
                "user_id": 11,
                "queue_id": 1,
                "group_id": 1,
                "partition_id": 1,
            },
        ]
    )


def test_build_batsim_run_config_generates_inputs(tmp_path: Path) -> None:
    trace = tmp_path / "trace.parquet"
    _sample_trace_df().to_parquet(trace, index=False)

    result = build_batsim_run_config(
        run_id="batsim_case",
        trace_dataset=trace,
        policy_id="FIFO_STRICT",
        out_dir=tmp_path,
        capacity_cpus=4,
        edc_mode="library_file",
        edc_library_path="/home/test/.nix-profile/lib/libfcfs.so",
        use_wsl_defaults=False,
    )

    assert result.config_path.exists()
    assert result.payload["engine"] == "batsim"
    assert result.payload["policy"]["policy_id"] == "FIFO_STRICT"
    assert result.payload["inputs"]["workload_source"] == "generated_from_trace"

    workload_path = Path(result.payload["inputs"]["workload_path"])
    platform_path = Path(result.payload["inputs"]["platform_path"])
    edc_init_file = Path(result.payload["inputs"]["edc_init_file"])
    assert workload_path.exists()
    assert platform_path.exists()
    assert edc_init_file.exists()

    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    assert workload["nb_res"] == 4
    assert len(workload["jobs"]) == 2
    assert workload["profiles"]

    args = result.payload["batsim_cli"]["args"]
    assert "--platform" in args
    assert "--workload" in args
    assert "--edc-library-file" in args


def test_build_batsim_run_config_with_provided_inputs(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text("{}", encoding="utf-8")
    workload = tmp_path / "workload.json"
    write_json(workload, {"jobs": [], "profiles": {}, "nb_res": 8})
    platform = tmp_path / "platform.xml"
    platform.write_text('<platform version="4.1"></platform>\n', encoding="utf-8")

    result = build_batsim_run_config(
        run_id="provided_case",
        trace_dataset=trace,
        policy_id="FIFO_STRICT",
        out_dir=tmp_path,
        workload_path=workload,
        platform_path=platform,
        capacity_cpus=8,
        edc_mode="socket_str",
        edc_socket_endpoint="ipc:///tmp/hpcopt.sock",
        edc_init_json='{"format_json": true}',
        use_wsl_defaults=False,
    )

    assert result.payload["inputs"]["workload_source"] == "provided"
    assert result.payload["inputs"]["platform_source"] == "provided"
    assert "--edc-socket-str" in result.payload["batsim_cli"]["args"]


def test_invoke_batsim_run_dry_mode(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "batsim_cli": {
                "args": [
                    "--export",
                    str(tmp_path / "batout_"),
                    "--platform",
                    str(tmp_path / "platform.xml"),
                    "--workload",
                    str(tmp_path / "workload.json"),
                    "--edc-library-file",
                    "/tmp/libfcfs.so",
                    str(tmp_path / "edc-init.json"),
                ]
            }
        },
    )

    result = invoke_batsim_run(config_path=config, batsim_bin="batsim", dry_run=True)
    assert result.status == "dry_run"
    assert result.returncode == 0
    assert result.command[0] == "batsim"


def test_invoke_batsim_run_dry_mode_wsl(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "batsim_cli": {
                "args": [
                    "--platform",
                    r"C:\tmp\platform.xml",
                    "--workload",
                    r"C:\tmp\workload.json",
                    "--edc-library-file",
                    "/home/test/.nix-profile/lib/libfcfs.so",
                    r"C:\tmp\edc-init.json",
                ]
            }
        },
    )

    monkeypatch.setattr("hpcopt.simulate.batsim.shutil.which", lambda _: "wsl")
    result = invoke_batsim_run(
        config_path=config,
        batsim_bin="batsim",
        dry_run=True,
        use_wsl=True,
        wsl_distro="Ubuntu",
    )
    assert result.status == "dry_run"
    assert result.returncode == 0
    assert result.command[:4] == ["wsl", "-d", "Ubuntu", "--"]
    assert "/mnt/c/tmp/platform.xml" in result.command[-1]
    assert "--edc-library-file" in result.command[-1]


def test_windows_path_to_wsl_conversion() -> None:
    assert windows_path_to_wsl(r"C:\data\trace.parquet") == "/mnt/c/data/trace.parquet"
    assert windows_path_to_wsl(r"D:\work\run.json") == "/mnt/d/work/run.json"


def test_normalize_batsim_run_outputs(tmp_path: Path) -> None:
    jobs_csv = tmp_path / "batout_jobs.csv"
    pd.DataFrame(
        [
            {
                "job_id": "w0!1",
                "submission_time": 0.0,
                "starting_time": 1.0,
                "finish_time": 6.0,
                "execution_time": 5.0,
                "requested_number_of_resources": 2,
                "requested_time": 10.0,
            },
            {
                "job_id": "w0!2",
                "submission_time": 2.0,
                "starting_time": 6.0,
                "finish_time": 7.0,
                "execution_time": 1.0,
                "requested_number_of_resources": 1,
                "requested_time": -1.0,
            },
        ]
    ).to_csv(jobs_csv, index=False)

    workload = tmp_path / "workload.json"
    write_json(
        workload,
        {
            "nb_res": 4,
            "jobs": [
                {"id": "1", "extra_data": json.dumps({"user_id": 42, "queue_id": 1})},
                {"id": "2", "extra_data": json.dumps({"user_id": 7, "queue_id": 2})},
            ],
            "profiles": {},
        },
    )

    config = tmp_path / "run_config.json"
    write_json(
        config,
        {
            "run_id": "norm_case",
            "policy": {"policy_id": "FIFO_STRICT"},
            "resources": {"capacity_cpus": 4},
            "inputs": {"workload_path": str(workload)},
            "batsim_cli": {
                "args": [
                    "--export",
                    str(tmp_path / "batout_"),
                    "--platform",
                    str(tmp_path / "platform.xml"),
                    "--workload",
                    str(workload),
                    "--edc-library-file",
                    "/tmp/libfcfs.so",
                    str(tmp_path / "edc-init.json"),
                ]
            },
        },
    )

    result = normalize_batsim_run_outputs(
        config_path=config,
        report_out_dir=tmp_path,
        simulation_out_dir=tmp_path,
    )
    assert result.jobs_csv_path == jobs_csv
    assert result.jobs_artifact_path.exists()
    assert result.queue_artifact_path.exists()
    assert result.sim_report_path.exists()
    assert result.invariant_report_path.exists()
    assert result.objective_metrics["job_count"] == 2.0
