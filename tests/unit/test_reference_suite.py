from pathlib import Path

import pytest
import yaml
from hpcopt.data.reference_suite import (
    assert_reference_by_filename_and_hash,
    lock_reference_suite_hashes,
    match_trace_to_reference,
)


def test_reference_suite_lock_updates_hashes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trace_path = raw_dir / "sample.swf"
    trace_path.write_text("1 0 0 10 1 0 -1 1 10 -1 1 1 1 0 1 1 -1 -1\n", encoding="utf-8")

    cfg = {
        "suite_id": "suite_test",
        "traces": [
            {
                "id": "sample_trace",
                "filename": "sample.swf",
                "source": "local",
                "sha256": None,
            }
        ],
    }
    cfg_path = tmp_path / "reference_suite.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    report = lock_reference_suite_hashes(
        config_path=cfg_path,
        raw_dir=raw_dir,
        out_report_path=tmp_path / "lock_report.json",
        strict_missing=True,
    )
    assert report["updated"] is True
    assert report["missing_files"] == []
    updated = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert updated["traces"][0]["sha256"] is not None


def test_reference_suite_hash_enforcement(tmp_path: Path) -> None:
    cfg = {
        "suite_id": "suite_test",
        "traces": [
            {
                "id": "sample_trace",
                "filename": "sample.swf",
                "source": "local",
                "sha256": "abc",
            }
        ],
    }
    cfg_path = tmp_path / "reference_suite.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError):
        assert_reference_by_filename_and_hash(
            filename="sample.swf",
            sha256_observed="def",
            config_path=cfg_path,
        )

    match = match_trace_to_reference(
        trace_path=tmp_path / "other.swf",
        config_path=cfg_path,
    )
    assert match is None
