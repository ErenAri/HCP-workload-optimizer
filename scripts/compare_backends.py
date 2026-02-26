"""Run LightGBM GPU training on RTX 2060."""

from __future__ import annotations

import json
import time
from pathlib import Path

DATASET = Path("outputs/quickstart/curated/ctc_sp2_demo_features.parquet")
OUT = Path("outputs/benchmark/models")


def main() -> None:
    from hpcopt.models.runtime_quantile import train_runtime_quantile_models

    if not DATASET.exists():
        print("ERROR: dataset not found: " + str(DATASET))
        return

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LightGBM GPU (RTX 2060)")
    print("=" * 60)

    t0 = time.perf_counter()
    result = train_runtime_quantile_models(
        dataset_path=DATASET,
        out_dir=OUT,
        model_id="ctc_sp2_lightgbm_gpu",
        seed=42,
        backend="lightgbm",
        hyperparams={"device": "gpu", "gpu_use_dp": False},
    )
    elapsed = time.perf_counter() - t0

    with open(result.metrics_path) as f:
        m = json.load(f)

    p50 = m["quantiles"]["p50"]
    coverage = m.get("interval_coverage_p10_p90", 0)

    print("  Time:     " + format(elapsed, ".1f") + "s")
    print("  p50 MAE:  " + format(p50["mae"], ",.0f") + "s")
    print("  p50 Pinball: " + format(p50["pinball_loss"], ",.0f"))
    print("  Coverage: " + format(coverage, ".1%"))

    # Summary
    print()
    print("=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print("  sklearn GBR:     383.4s  (MAE 7,889s, Coverage 78.1%)")
    print("  LightGBM CPU:      7.6s  (MAE 7,854s, Coverage 74.9%)")
    print(
        "  LightGBM GPU:  "
        + format(elapsed, ".1f")
        + "s  (MAE "
        + format(p50["mae"], ",.0f")
        + "s, Coverage "
        + format(coverage, ".1%")
        + ")"
    )
    print()
    print("  sklearn -> LightGBM CPU: " + format(383.4 / 7.6, ".0f") + "x speedup")
    if elapsed > 0:
        print("  sklearn -> LightGBM GPU: " + format(383.4 / elapsed, ".0f") + "x speedup")


if __name__ == "__main__":
    main()
