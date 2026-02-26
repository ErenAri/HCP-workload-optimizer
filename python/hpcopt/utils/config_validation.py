"""Configuration validation utilities.

Loads a YAML config file and validates it against a JSON Schema from the
``schemas/`` directory.  Falls back gracefully when ``jsonschema`` is not
installed (it lives in the ``production`` extras group).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve the project-level schemas/ directory.  Works both from an editable
# install (repo root) and from a built wheel (schemas shipped as data).
_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"


def _locate_schema(schema_name: str) -> Path:
    """Return the absolute path to a schema JSON file.

    Parameters
    ----------
    schema_name:
        Bare name **without** the ``.schema.json`` suffix, e.g.
        ``"fidelity_gate_config"``.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist on disk.
    """
    candidate = _SCHEMAS_DIR / f"{schema_name}.schema.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Schema file not found: {candidate}  (looked in {_SCHEMAS_DIR})")
    return candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level of {path}, got {type(data).__name__}")
    return data


def validate_config(path: Path, schema_name: str) -> dict[str, Any]:
    """Validate a YAML configuration file against a JSON Schema.

    Parameters
    ----------
    path:
        Path to the YAML config file to validate.
    schema_name:
        Bare schema name (without ``.schema.json`` suffix) that must exist
        under the project ``schemas/`` directory.  Examples:
        ``"fidelity_gate_config"``, ``"policy_config"``,
        ``"reference_suite_config"``.

    Returns
    -------
    dict
        ``{"valid": True,  "errors": []}`` on success, or
        ``{"valid": False, "errors": [<str>, ...]}`` on failure.
    """
    errors: list[str] = []

    # ---- Load the config ------------------------------------------------
    try:
        config_data = _load_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {"valid": False, "errors": [f"Failed to load config: {exc}"]}

    # ---- Load the schema ------------------------------------------------
    try:
        schema_path = _locate_schema(schema_name)
    except FileNotFoundError as exc:
        return {"valid": False, "errors": [str(exc)]}

    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"Failed to load schema: {exc}"]}

    # ---- Validate -------------------------------------------------------
    try:
        import jsonschema  # noqa: F811
    except ImportError:
        logger.warning(
            "jsonschema is not installed; skipping schema validation.  "
            "Install with:  pip install 'hpc-workload-optimizer[production]'"
        )
        return {"valid": True, "errors": []}

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    for error in validator.iter_errors(config_data):
        json_path = " -> ".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"[{json_path}] {error.message}")

    return {"valid": len(errors) == 0, "errors": errors}
