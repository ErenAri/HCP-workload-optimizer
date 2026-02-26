"""Tests for the model registry."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from hpcopt.models.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(registry_path=tmp_path / "registry.jsonl")


def test_register_and_get(registry: ModelRegistry, tmp_path: Path) -> None:
    model_dir = tmp_path / "model_v1"
    model_dir.mkdir()
    entry = registry.register("v1", str(model_dir), metadata={"seed": 42})
    assert entry["model_id"] == "v1"
    assert entry["status"] == "registered"

    fetched = registry.get("v1")
    assert fetched["model_id"] == "v1"


def test_register_duplicate_raises(registry: ModelRegistry, tmp_path: Path) -> None:
    model_dir = tmp_path / "model_dup"
    model_dir.mkdir()
    registry.register("dup", str(model_dir))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", str(model_dir))


def test_get_not_found_raises(registry: ModelRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_list_returns_all(registry: ModelRegistry, tmp_path: Path) -> None:
    for i in range(3):
        d = tmp_path / f"model_{i}"
        d.mkdir()
        registry.register(f"m{i}", str(d))
    entries = registry.list()
    assert len(entries) == 3
    assert [e["model_id"] for e in entries] == ["m0", "m1", "m2"]


def test_promote_and_demote(registry: ModelRegistry, tmp_path: Path) -> None:
    d1 = tmp_path / "m1"
    d1.mkdir()
    d2 = tmp_path / "m2"
    d2.mkdir()
    registry.register("m1", str(d1))
    registry.register("m2", str(d2))

    registry.promote("m1")
    assert registry.get("m1")["status"] == "production"
    assert registry.get_production()["model_id"] == "m1"

    # Promoting m2 demotes m1
    registry.promote("m2")
    assert registry.get("m2")["status"] == "production"
    assert registry.get("m1")["status"] == "registered"


def test_archive(registry: ModelRegistry, tmp_path: Path) -> None:
    d = tmp_path / "archive_model"
    d.mkdir()
    registry.register("arch", str(d))
    registry.archive("arch")
    assert registry.get("arch")["status"] == "archived"


def test_promote_archived_raises(registry: ModelRegistry, tmp_path: Path) -> None:
    d = tmp_path / "a"
    d.mkdir()
    registry.register("a", str(d))
    registry.archive("a")
    with pytest.raises(ValueError, match="Cannot promote archived"):
        registry.promote("a")


def test_get_production_none_initially(registry: ModelRegistry) -> None:
    assert registry.get_production() is None


def test_corrupt_jsonl_recovery(tmp_path: Path) -> None:
    """Registry should skip malformed lines and continue loading."""
    reg_path = tmp_path / "reg.jsonl"
    reg_path.write_text(
        '{"model_id":"good","model_dir":"d","status":"registered","registered_at":"2024-01-01T00:00:00"}\n'
        "this is not json\n"
        '{"model_id":"also_good","model_dir":"d2","status":"registered","registered_at":"2024-01-01T00:00:00"}\n',
        encoding="utf-8",
    )
    reg = ModelRegistry(registry_path=reg_path)
    entries = reg.list()
    assert len(entries) == 2
    assert entries[0]["model_id"] == "good"
    assert entries[1]["model_id"] == "also_good"


def test_concurrent_registration(tmp_path: Path) -> None:
    """Multiple threads registering models should not corrupt the registry."""
    reg = ModelRegistry(registry_path=tmp_path / "concurrent.jsonl")
    errors: list[Exception] = []

    def _register(idx: int) -> None:
        try:
            d = tmp_path / f"cm_{idx}"
            d.mkdir(exist_ok=True)
            reg.register(f"cm_{idx}", str(d))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_register, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    entries = reg.list()
    assert len(entries) == 10
