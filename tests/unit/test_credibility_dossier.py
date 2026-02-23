"""Tests for the credibility dossier assembly module."""
from __future__ import annotations

import json
from pathlib import Path

from hpcopt.artifacts.credibility_dossier import (
    DossierResult,
    _safe_load_json,
    _collect_trace_reports,
    _build_fidelity_summary,
    _build_recommendation_summary,
    _build_fallback_summary,
    _build_no_improvement_summary,
    assemble_credibility_dossier,
)


def test_safe_load_json_valid(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"key": "value"}), encoding="utf-8")
    result = _safe_load_json(p)
    assert result == {"key": "value"}


def test_safe_load_json_invalid(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert _safe_load_json(p) is None


def test_safe_load_json_missing(tmp_path: Path) -> None:
    assert _safe_load_json(tmp_path / "missing.json") is None


def test_safe_load_json_non_dict(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _safe_load_json(p) is None


def test_collect_trace_reports_missing_dir(tmp_path: Path) -> None:
    result = _collect_trace_reports(tmp_path / "nonexistent")
    assert result["status"] == "missing"


def test_collect_trace_reports_with_files(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "fidelity_report.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    (reports / "sim_report_fifo.json").write_text(
        json.dumps({"policy_id": "FIFO"}), encoding="utf-8"
    )
    result = _collect_trace_reports(tmp_path)
    assert "fidelity" in result
    assert "simulations" in result
    assert result["simulations"]["FIFO"]["policy_id"] == "FIFO"


def test_build_fidelity_summary() -> None:
    per_trace = {
        "trace_a": {"fidelity": {"status": "pass"}},
        "trace_b": {"fidelity": {"status": "fail"}},
    }
    summary = _build_fidelity_summary(per_trace)
    assert summary["all_pass"] is False
    assert len(summary["traces"]) == 2


def test_build_fidelity_summary_all_pass() -> None:
    per_trace = {
        "a": {"fidelity": {"status": "pass"}},
        "b": {"fidelity": {"status": "pass"}},
    }
    assert _build_fidelity_summary(per_trace)["all_pass"] is True


def test_build_recommendation_summary() -> None:
    per_trace = {
        "a": {"recommendation": {"status": "accepted"}},
        "b": {"recommendation": {"status": "blocked", "failure_modes": [{"policy_id": "ML"}]}},
    }
    summary = _build_recommendation_summary(per_trace)
    assert summary["accepted_count"] == 1
    assert summary["blocked_count"] == 1
    assert len(summary["failure_modes"]) == 1


def test_build_fallback_summary() -> None:
    per_trace = {
        "a": {
            "simulations": {
                "ML_BACKFILL_P50": {"fallback_accounting": {"prediction_used_rate": 0.8}}
            }
        },
    }
    summary = _build_fallback_summary(per_trace)
    assert len(summary["per_trace"]) == 1
    assert summary["per_trace"][0]["prediction_used_rate"] == 0.8


def test_build_no_improvement_summary() -> None:
    per_trace = {
        "a": {"recommendation": {"no_improvement_narrative": {"summary": "no gain"}}},
        "b": {"recommendation": {}},
    }
    narratives = _build_no_improvement_summary(per_trace)
    assert len(narratives) == 1
    assert narratives[0]["summary"] == "no gain"


def test_assemble_credibility_dossier_empty(tmp_path: Path) -> None:
    result = assemble_credibility_dossier(
        input_dir=tmp_path / "traces",
        output_path=tmp_path / "output",
    )
    assert isinstance(result, DossierResult)
    assert result.json_path.exists()
    assert result.md_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["traces_collected"] == []


def test_assemble_credibility_dossier_with_traces(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    trace_a = traces_dir / "trace_a" / "reports"
    trace_a.mkdir(parents=True)
    (trace_a / "fidelity_report.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    (trace_a / "recommendation_report.json").write_text(
        json.dumps({"status": "accepted"}), encoding="utf-8"
    )

    result = assemble_credibility_dossier(
        input_dir=traces_dir,
        output_path=tmp_path / "output",
    )
    assert result.json_path.exists()
    payload = result.payload
    assert "trace_a" in payload["traces_collected"]
    assert payload["fidelity_summary"]["all_pass"] is True
    assert payload["recommendation_summary"]["accepted_count"] == 1
    # Markdown should mention trace_a
    md = result.md_path.read_text(encoding="utf-8")
    assert "trace_a" in md
