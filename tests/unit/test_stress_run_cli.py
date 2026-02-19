from pathlib import Path

import json
import yaml
from typer.testing import CliRunner

from hpcopt.cli.main import app
from hpcopt.simulate.stress import generate_stress_scenario


def test_stress_run_cli_emits_report_and_manifest(tmp_path: Path) -> None:
    scenario = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=120,
        seed=9,
        params={"alpha": 1.3},
    )
    policy_path = tmp_path / "policy_fifo.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "FIFO_STRICT",
                "runtime_guard_k": 0.5,
                "strict_uncertainty_mode": False,
                "starvation_wait_cap_sec": 172800,
                "fairness": {
                    "starvation_rate_max": 0.02,
                    "fairness_dev_delta_max": 0.05,
                    "jain_delta_max": 0.03,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    out_dir = tmp_path / "sim_out"
    report_dir = tmp_path / "report_out"
    run_id = "stress_cli_fixture"
    result = runner.invoke(
        app,
        [
            "stress",
            "run",
            "--scenario",
            "heavy_tail",
            "--policy",
            str(policy_path),
            "--model",
            "fixture_model",
            "--dataset",
            str(scenario.dataset_path),
            "--capacity-cpus",
            "64",
            "--out",
            str(out_dir),
            "--report-out",
            str(report_dir),
            "--run-id",
            run_id,
            "--strict-invariants",
        ],
    )
    assert result.exit_code == 0, result.output

    stress_report = report_dir / f"{run_id}_stress_report.json"
    stress_manifest = report_dir / f"{run_id}_stress_manifest.json"
    assert stress_report.exists()
    assert stress_manifest.exists()

    payload = json.loads(stress_report.read_text(encoding="utf-8"))
    assert payload["scenario"] == "heavy_tail"
    assert payload["candidate_policy_id"] == "FIFO_STRICT"
    assert payload["baseline_policy_id"] == "EASY_BACKFILL_BASELINE"
    assert payload["status"] in {"pass", "fail"}
    assert "constraints" in payload
    assert "degrade_signatures" in payload

