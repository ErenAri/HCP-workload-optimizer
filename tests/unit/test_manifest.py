import json
from pathlib import Path

from hpcopt.artifacts.manifest import build_manifest, write_manifest


def test_manifest_build_and_write(tmp_path: Path) -> None:
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "out.txt"
    input_file.write_text("in", encoding="utf-8")
    output_file.write_text("out", encoding="utf-8")

    payload = build_manifest(
        command="hpcopt test manifest",
        inputs=[input_file],
        outputs=[output_file],
        params={"seed": 42},
        policy_spec_path=None,
        seeds=[42],
    )
    assert payload["command"] == "hpcopt test manifest"
    assert payload["inputs"][0]["sha256"] is not None
    assert payload["outputs"][0]["sha256"] is not None
    assert payload["params"]["seed"] == 42
    assert payload["immutable"] is True
    assert payload["seeds"] == [42]

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, payload)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["command"] == payload["command"]
    assert saved["manifest_hash_sha256"] is not None
