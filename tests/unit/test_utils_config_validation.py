"""Tests for config validation utility."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


def test_validate_config_missing_file(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import validate_config
    result = validate_config(tmp_path / "missing.yaml", "test")
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_validate_config_invalid_yaml(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import validate_config

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":::invalid: yaml: [", encoding="utf-8")

    result = validate_config(bad_yaml, "nonexistent_schema")
    assert result["valid"] is False


def test_validate_config_schema_not_found(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import validate_config

    config_path = tmp_path / "ok.yaml"
    config_path.write_text(yaml.dump({"key": "value"}), encoding="utf-8")

    result = validate_config(config_path, "totally_nonexistent_schema_xyz")
    assert result["valid"] is False
    assert any("not found" in e.lower() for e in result["errors"])


def test_validate_config_valid_with_schema(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import validate_config
    import hpcopt.utils.config_validation as mod

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "test.schema.json").write_text(json.dumps(schema), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"name": "hello"}), encoding="utf-8")

    with patch.object(mod, "_SCHEMAS_DIR", schema_dir):
        result = validate_config(config_path, "test")
    assert result["valid"] is True


def test_validate_config_invalid_against_schema(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import validate_config
    import hpcopt.utils.config_validation as mod

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "test.schema.json").write_text(json.dumps(schema), encoding="utf-8")

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.dump({"wrong": 123}), encoding="utf-8")

    with patch.object(mod, "_SCHEMAS_DIR", schema_dir):
        result = validate_config(config_path, "test")
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_load_yaml_non_dict(tmp_path: Path) -> None:
    from hpcopt.utils.config_validation import _load_yaml

    p = tmp_path / "list.yaml"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        _load_yaml(p)
