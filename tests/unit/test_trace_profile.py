from pathlib import Path

import json

from hpcopt.ingest.swf import ingest_swf
from hpcopt.profile.trace_profile import build_trace_profile


def test_trace_profile_emits_expected_sections(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/sample_trace.swf")
    ingest_result = ingest_swf(
        input_path=fixture,
        out_dir=tmp_path / "curated",
        dataset_id="sample_trace",
        report_dir=tmp_path / "reports",
    )

    profile_result = build_trace_profile(
        dataset_path=ingest_result.dataset_path,
        report_dir=tmp_path / "reports",
        dataset_id="sample_trace",
    )
    assert profile_result.row_count == 3
    assert profile_result.profile_path.exists()

    payload = json.loads(profile_result.profile_path.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "sample_trace"
    assert payload["row_count"] == 3
    assert payload["overrequest_distribution"]["sample_size"] == 2
    assert payload["congestion_regime"]["event_count"] >= 2
