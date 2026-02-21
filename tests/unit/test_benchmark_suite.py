from pathlib import Path

import json

import pandas as pd

from hpcopt.artifacts.benchmark import run_benchmark_suite
from hpcopt.ingest.swf import ingest_swf


def test_benchmark_suite_emits_report_and_history(tmp_path: Path) -> None:
    ingest = ingest_swf(
        input_path=Path("tests/fixtures/sample_trace.swf"),
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )
    report_path = tmp_path / "reports" / "bench_run_benchmark_report.json"
    history_path = tmp_path / "reports" / "benchmark_history.jsonl"
    result = run_benchmark_suite(
        trace_dataset=ingest.dataset_path,
        report_path=report_path,
        history_path=history_path,
        raw_trace=Path("tests/fixtures/sample_trace.swf"),
        policy_id="FIFO_STRICT",
        capacity_cpus=4,
        samples=1,
        regression_max_drop=0.10,
        history_window=5,
    )
    assert result.report_path.exists()
    assert result.history_path.exists()
    assert result.status in {"pass", "fail"}
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["parse_benchmark"]["status"] == "ok"
    assert payload["simulation_benchmark"]["status"] == "ok"
    assert payload["pipeline_benchmark"]["status"] == "ok"
    assert "regression_gate" in payload

    # Second run should append to history and still return a structured result.
    second = run_benchmark_suite(
        trace_dataset=ingest.dataset_path,
        report_path=tmp_path / "reports" / "bench_run2_benchmark_report.json",
        history_path=history_path,
        raw_trace=Path("tests/fixtures/sample_trace.swf"),
        policy_id="FIFO_STRICT",
        capacity_cpus=4,
        samples=1,
        regression_max_drop=0.10,
        history_window=5,
    )
    assert second.report_path.exists()
    lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2


def test_benchmark_suite_supports_parse_skip(tmp_path: Path) -> None:
    dataset = tmp_path / "trace.parquet"
    pd.DataFrame(
        [
            {
                "job_id": 1,
                "submit_ts": 0,
                "start_ts": 0,
                "end_ts": 10,
                "runtime_actual_sec": 10,
                "runtime_requested_sec": 20,
                "requested_cpus": 2,
                "requested_mem": None,
                "user_id": 1,
            },
            {
                "job_id": 2,
                "submit_ts": 5,
                "start_ts": 10,
                "end_ts": 30,
                "runtime_actual_sec": 20,
                "runtime_requested_sec": 30,
                "requested_cpus": 2,
                "requested_mem": None,
                "user_id": 2,
            },
            {
                "job_id": 3,
                "submit_ts": 9,
                "start_ts": 11,
                "end_ts": 12,
                "runtime_actual_sec": 1,
                "runtime_requested_sec": 2,
                "requested_cpus": 1,
                "requested_mem": None,
                "user_id": 1,
            },
        ]
    ).to_parquet(dataset, index=False)

    result = run_benchmark_suite(
        trace_dataset=dataset,
        report_path=tmp_path / "reports" / "bench_skip_benchmark_report.json",
        history_path=tmp_path / "reports" / "benchmark_history.jsonl",
        raw_trace=None,
        policy_id="FIFO_STRICT",
        capacity_cpus=4,
        samples=1,
    )
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["parse_benchmark"]["status"] == "skipped"
    assert payload["simulation_benchmark"]["status"] == "ok"
    assert payload["pipeline_benchmark"]["status"] == "ok"

