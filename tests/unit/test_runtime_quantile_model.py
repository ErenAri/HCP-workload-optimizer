from pathlib import Path

from hpcopt.models.runtime_quantile import RuntimeQuantilePredictor, train_runtime_quantile_models
from hpcopt.simulate.stress import generate_stress_scenario


def test_runtime_quantile_training_and_inference(tmp_path: Path) -> None:
    stress = generate_stress_scenario(
        scenario="heavy_tail",
        out_dir=tmp_path,
        n_jobs=160,
        seed=17,
        params={"alpha": 1.3},
    )
    train = train_runtime_quantile_models(
        dataset_path=stress.dataset_path,
        out_dir=tmp_path / "models",
        model_id="runtime_test_model",
        seed=7,
    )

    assert (train.model_dir / "p10.joblib").exists()
    assert (train.model_dir / "p50.joblib").exists()
    assert (train.model_dir / "p90.joblib").exists()
    assert train.metrics_path.exists()
    assert train.metadata_path.exists()

    predictor = RuntimeQuantilePredictor(train.model_dir)
    pred = predictor.predict_one(
        {
            "requested_cpus": 8,
            "runtime_requested_sec": 1800,
            "requested_mem": None,
            "queue_id": 1,
            "partition_id": 1,
            "user_id": 2,
            "group_id": 1,
        }
    )
    assert pred["p10"] <= pred["p50"] <= pred["p90"]
    assert pred["p10"] > 0
