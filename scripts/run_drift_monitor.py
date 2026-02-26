from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from hpcopt.models.drift import compute_drift_report
from hpcopt.models.runtime_quantile import resolve_runtime_model_dir
from hpcopt.utils.io import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scheduled drift monitoring if inputs are configured.")
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    eval_dataset = args.eval_dataset or (
        Path(os.environ["HPCOPT_DRIFT_EVAL_DATASET"]) if "HPCOPT_DRIFT_EVAL_DATASET" in os.environ else None
    )
    model_dir_opt = args.model_dir or (
        Path(os.environ["HPCOPT_DRIFT_MODEL_DIR"]) if "HPCOPT_DRIFT_MODEL_DIR" in os.environ else None
    )

    if eval_dataset is None:
        print("Drift monitor: SKIP (no eval dataset configured)")
        return 0
    if not eval_dataset.exists():
        print(f"Drift monitor: FAIL (eval dataset not found: {eval_dataset})")
        return 1

    # Fast schema sanity check before model ops.
    pd.read_parquet(eval_dataset).head(1)

    resolved_model_dir = resolve_runtime_model_dir(model_dir_opt)
    if resolved_model_dir is None:
        print("Drift monitor: SKIP (no model directory configured)")
        return 0

    ensure_dir(args.out_dir)
    report = compute_drift_report(model_dir=resolved_model_dir, eval_dataset_path=eval_dataset)
    report_path = args.out_dir / f"drift_monitor_{resolved_model_dir.name}.json"
    write_json(report_path, report.to_dict())
    print(f"Drift monitor: PASS ({report_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
