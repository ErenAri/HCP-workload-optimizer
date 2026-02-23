"""Tests for file-based API key loading."""
from __future__ import annotations

import pytest
from pathlib import Path

from hpcopt.utils.secrets import invalidate_api_keys_cache, load_api_keys


@pytest.fixture(autouse=True)
def _clear_key_cache() -> None:
    """Ensure each test starts with a fresh API key cache."""
    invalidate_api_keys_cache()


def test_load_from_file_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keys_file = tmp_path / "api_keys.txt"
    keys_file.write_text("key-alpha\nkey-beta\n# comment\n\n", encoding="utf-8")
    monkeypatch.setenv("HPCOPT_API_KEYS_FILE", str(keys_file))
    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    result = load_api_keys()
    assert result == {"key-alpha", "key-beta"}


def test_load_from_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HPCOPT_API_KEYS_FILE", raising=False)
    monkeypatch.setenv("HPCOPT_API_KEYS", "k1, k2 ,k3")
    result = load_api_keys()
    assert result == {"k1", "k2", "k3"}


def test_load_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HPCOPT_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    result = load_api_keys()
    assert result == set()


def test_missing_file_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCOPT_API_KEYS_FILE", str(tmp_path / "nonexistent.txt"))
    monkeypatch.delenv("HPCOPT_API_KEYS", raising=False)
    result = load_api_keys()
    assert result == set()
