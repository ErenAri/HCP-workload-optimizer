from pathlib import Path

import json
import pandas as pd

from hpcopt.simulate.stress import generate_stress_scenario


def test_stress_scenario_generation_heavy_tail(tmp_path: Path) -> None:
    result = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=250,
        seed=7,
        params={"alpha": 1.3},
    )
    assert result.dataset_path.exists()
    assert result.metadata_path.exists()

    df = pd.read_parquet(result.dataset_path)
    assert len(df) == 250
    assert (df["runtime_requested_sec"] >= df["runtime_actual_sec"]).all()

    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["scenario"] == "heavy_tail"
    assert meta["n_jobs"] == 250
