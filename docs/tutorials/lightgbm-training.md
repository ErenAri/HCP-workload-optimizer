# LightGBM Training

LightGBM provides **50× faster training** compared to sklearn's GradientBoostingRegressor, with identical or better accuracy.

## Auto-Detection

If LightGBM is installed, it's automatically selected as the default backend:

```bash
pip install lightgbm>=4.3.0
# Or with the project extra:
pip install hpc-workload-optimizer[lightgbm]
```

## Training

```python
from pathlib import Path
from hpcopt.models.runtime_quantile import train_runtime_quantile_models

result = train_runtime_quantile_models(
    dataset_path=Path("outputs/curated/ctc_sp2_features.parquet"),
    out_dir=Path("outputs/models"),
    model_id="ctc_sp2_lgbm",
    backend="lightgbm",  # or "sklearn" for the original
)
```

## GPU Acceleration

If you have an NVIDIA GPU (tested on RTX 2060):

```python
result = train_runtime_quantile_models(
    dataset_path=Path("outputs/curated/ctc_sp2_features.parquet"),
    out_dir=Path("outputs/models"),
    model_id="ctc_sp2_gpu",
    backend="lightgbm",
    hyperparams={"device": "gpu"},
)
```

!!! tip "When to use GPU"
    GPU acceleration adds kernel launch overhead. It's faster than CPU for datasets with **500K+ rows**.
    For smaller datasets (under 100K rows), CPU LightGBM is typically faster.

## Performance Comparison

| Backend | Time | p50 MAE | Speedup |
|---|---|---|---|
| sklearn GBR (CPU) | 383.4s | 7,889s | baseline |
| **LightGBM (CPU)** | **7.6s** | 7,854s | **50×** |
| LightGBM (GPU/RTX 2060) | 15.5s | 7,854s | 25× |

## Ensemble Prediction

Combine multiple backends for best accuracy:

```python
from hpcopt.models.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor.from_model_dirs([
    Path("outputs/models/ctc_sp2_sklearn"),
    Path("outputs/models/ctc_sp2_lgbm"),
])

prediction = ensemble.predict_one({"requested_cpus": 4, "user_id": "alice", ...})
# {'p10': 120.5, 'p50': 340.2, 'p90': 890.1}
```

The ensemble auto-weights models by inverse pinball loss — better models get higher weight.
