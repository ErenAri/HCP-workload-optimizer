from __future__ import annotations

import json
from pathlib import Path

from hpcopt.models.runtime_quantile import FEATURE_COLUMNS
from hpcopt.simulate.core_helpers import build_prediction_features

_CONTRACT_PATH = Path("tests/fixtures/runtime_feature_contract.json")


def _load_contract() -> dict[str, object]:
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def test_runtime_feature_contract_matches_model_columns() -> None:
    contract = _load_contract()
    required = contract["required_by_predictor"]
    assert isinstance(required, list)
    assert required == FEATURE_COLUMNS


def test_simulation_feature_payload_satisfies_contract_defaults() -> None:
    contract = _load_contract()
    sample_job = contract["sample_job"]
    required = set(contract["required_by_predictor"])
    derived = set(contract["derived_by_predictor"])
    expected_provided = set(contract["provided_by_simulation"])
    defaults = contract["default_expectations"]

    assert isinstance(sample_job, dict)
    payload = build_prediction_features(sample_job)
    provided = set(payload.keys())

    assert required <= (provided | derived)
    assert expected_provided <= provided
    assert payload["user_overrequest_mean_lookback"] == defaults["user_overrequest_mean_lookback"]
    assert payload["queue_congestion_at_submit_jobs"] == defaults["queue_congestion_at_submit_jobs"]
    assert payload["user_runtime_median_lookback"] == sample_job["runtime_requested_sec"]
