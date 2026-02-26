"""Validate that all JSON schemas are well-formed and locked down."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
SCHEMA_FILES = sorted(SCHEMAS_DIR.glob("*.schema.json"))


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_json_with_meta(schema_path: Path) -> None:
    raw = schema_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "$schema" in data, f"{schema_path.name} missing $schema key"
    assert "title" in data or "$id" in data, f"{schema_path.name} missing title or $id"


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_root_not_permissive(schema_path: Path) -> None:
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    root_additional = data.get("additionalProperties")
    assert root_additional is not True, f"{schema_path.name} has additionalProperties: true at root — should be false"
