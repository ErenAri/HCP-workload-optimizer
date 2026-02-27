from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _load_script_module(name: str, relative_path: str):
    script_path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_coverage_thresholds_parser_and_aggregation() -> None:
    mod = _load_script_module("check_coverage_thresholds", "scripts/check_coverage_thresholds.py")

    package, threshold = mod._parse_package_threshold("simulate=86.0")
    assert package == "simulate"
    assert threshold == 86.0

    root = ET.fromstring(
        """
<coverage>
  <packages>
    <package name="hpcopt">
      <classes>
        <class filename="python/hpcopt/api/x.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
        <class filename="python/hpcopt/simulate/y.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""
    )

    global_stat = mod._collect_global_coverage(root)
    assert global_stat.hit_lines == 3
    assert global_stat.total_lines == 4
    assert round(global_stat.percent, 2) == 75.0

    api_stat = mod._collect_package_coverage(root, "api")
    assert api_stat.hit_lines == 1
    assert api_stat.total_lines == 2
    assert round(api_stat.percent, 2) == 50.0


def test_docs_consistency_extractors() -> None:
    mod = _load_script_module("check_docs_consistency", "scripts/check_docs_consistency.py")

    core_text = 'SUPPORTED_POLICIES = {"FIFO_STRICT", "EASY_BACKFILL_BASELINE", "ML_BACKFILL_P10"}'
    runtime_text = """
BACKENDS = ["sklearn", "lightgbm"]
FEATURE_COLUMNS = [
    "requested_cpus",
    "user_overrequest_mean_lookback",
    "user_runtime_median_lookback",
    "queue_congestion_at_submit_jobs",
]
"""

    policies = mod._extract_supported_policies(core_text)
    backends = mod._extract_backends(runtime_text)
    features = mod._extract_feature_columns(runtime_text)

    assert set(policies) == {"FIFO_STRICT", "EASY_BACKFILL_BASELINE", "ML_BACKFILL_P10"}
    assert set(backends) == {"sklearn", "lightgbm"}
    assert "user_overrequest_mean_lookback" in features
    assert "queue_congestion_at_submit_jobs" in features
