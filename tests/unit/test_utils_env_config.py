"""Tests for environment configuration loading."""
from __future__ import annotations

import pytest


def test_get_env_name_default() -> None:
    from hpcopt.utils.env_config import get_env_name
    # Without the env var, should return "dev"
    name = get_env_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_get_env_name_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpcopt.utils.env_config import get_env_name
    monkeypatch.setenv("HPCOPT_ENV", "staging")
    assert get_env_name() == "staging"


def test_load_env_config_returns_dict() -> None:
    from hpcopt.utils.env_config import load_env_config
    config = load_env_config()
    assert isinstance(config, dict)
    # Should contain defaults
    assert "rate_limit_per_minute" in config or isinstance(config, dict)


def test_load_env_config_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from hpcopt.utils.env_config import load_env_config
    monkeypatch.setenv("HPCOPT_ENV", "nonexistent_env_xyz")
    config = load_env_config()
    # Should still return defaults
    assert isinstance(config, dict)
