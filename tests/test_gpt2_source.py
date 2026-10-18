from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_journey.gpt2_source import GPT2Source, GPT2SourceError, load_source_manifest


def _payload() -> dict[str, object]:
    return {
        "model_id": "openai-community/gpt2",
        "revision": "6" * 40,
        "config_filename": "config.json",
        "config_sha256": "a" * 64,
        "weights_filename": "pytorch_model.bin",
        "weights_sha256": "b" * 64,
        "weights_bytes": 548_118_077,
    }


def test_loads_an_immutable_public_source_manifest(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(json.dumps(_payload()))

    source = load_source_manifest(path)

    assert source.artifact_url("config.json") == (
        "https://huggingface.co/openai-community/gpt2/resolve/"
        + "6" * 40
        + "/config.json"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "gpt2", "owner/name"),
        ("revision", "main", "40-character"),
        ("config_filename", "../config.json", "basename"),
        ("weights_sha256", "xyz", "SHA-256"),
        ("weights_bytes", 0, "in \\(0"),
        ("weights_bytes", True, "integer"),
    ],
)
def test_rejects_invalid_source_identities(
    field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises((TypeError, GPT2SourceError), match=message):
        GPT2Source(**payload)  # type: ignore[arg-type]


def test_rejects_duplicate_unknown_missing_and_oversized_manifest_data(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"model_id":"a/b","model_id":"c/d"}')
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({**_payload(), "extra": 1}))
    missing = tmp_path / "missing.json"
    values = _payload()
    del values["revision"]
    missing.write_text(json.dumps(values))
    oversized = tmp_path / "oversized.json"
    oversized.write_text("{}" + " " * 20)

    with pytest.raises(GPT2SourceError, match="duplicate"):
        load_source_manifest(duplicate)
    with pytest.raises(GPT2SourceError, match="fields must be exactly"):
        load_source_manifest(unknown)
    with pytest.raises(GPT2SourceError, match="fields must be exactly"):
        load_source_manifest(missing)
    with pytest.raises(GPT2SourceError, match="byte limit"):
        load_source_manifest(oversized, max_bytes=16)


def test_rejects_symlinked_manifests_and_undeclared_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_payload()))
    symlink = tmp_path / "source.json"
    symlink.symlink_to(manifest)

    with pytest.raises(GPT2SourceError, match="non-symlink"):
        load_source_manifest(symlink)
    with pytest.raises(GPT2SourceError, match="not declared"):
        GPT2Source(**_payload()).artifact_url("tokenizer.json")
