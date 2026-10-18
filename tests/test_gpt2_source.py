from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from ai_journey.gpt2_source import (
    GPT2Source,
    GPT2SourceError,
    load_checkpoint_state_dict,
    load_source_manifest,
    verify_source_artifacts,
)


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


def _write_artifacts(tmp_path: Path) -> GPT2Source:
    config = {
        "model_type": "gpt2",
        "vocab_size": 50_257,
        "n_positions": 1_024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "layer_norm_epsilon": 1e-5,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "activation_function": "gelu_new",
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    weights_bytes = b"bounded-fixture-weights"
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "pytorch_model.bin").write_bytes(weights_bytes)
    return GPT2Source(
        model_id="openai-community/gpt2",
        revision="6" * 40,
        config_filename="config.json",
        config_sha256=sha256(config_bytes).hexdigest(),
        weights_filename="pytorch_model.bin",
        weights_sha256=sha256(weights_bytes).hexdigest(),
        weights_bytes=len(weights_bytes),
    )


def test_verifies_artifact_bytes_and_exact_gpt2_configuration(tmp_path: Path) -> None:
    source = _write_artifacts(tmp_path)

    config = verify_source_artifacts(source, tmp_path)

    assert config.vocab_size == 50_257
    assert config.layer_count == 12


def test_rejects_wrong_artifact_size_and_hash(tmp_path: Path) -> None:
    source = _write_artifacts(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"short")

    with pytest.raises(GPT2SourceError, match="size mismatch"):
        verify_source_artifacts(source, tmp_path)

    source = _write_artifacts(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"tampered-fixture-weights")
    source = GPT2Source(
        **{**source.__dict__, "weights_bytes": len(b"tampered-fixture-weights")}
    )
    with pytest.raises(GPT2SourceError, match="SHA-256 mismatch"):
        verify_source_artifacts(source, tmp_path)


def test_rejects_symlinked_artifact_files(tmp_path: Path) -> None:
    source = _write_artifacts(tmp_path)
    original = tmp_path / "actual.bin"
    (tmp_path / "pytorch_model.bin").replace(original)
    (tmp_path / "pytorch_model.bin").symlink_to(original)

    with pytest.raises(GPT2SourceError, match="non-symlink"):
        verify_source_artifacts(source, tmp_path)


def _write_torch_checkpoint(tmp_path: Path, payload: object) -> GPT2Source:
    source = _write_artifacts(tmp_path)
    weights_path = tmp_path / "pytorch_model.bin"
    torch = pytest.importorskip("torch")
    torch.save(payload, weights_path)
    weights_bytes = weights_path.read_bytes()
    return GPT2Source(
        **{
            **source.__dict__,
            "weights_sha256": sha256(weights_bytes).hexdigest(),
            "weights_bytes": len(weights_bytes),
        }
    )


def test_loads_verified_checkpoint_with_the_weights_only_unpickler(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = _write_torch_checkpoint(tmp_path, {"weight": torch.arange(4).float()})

    config, state = load_checkpoint_state_dict(source, tmp_path)

    assert config.layer_count == 12
    torch.testing.assert_close(state["weight"], torch.arange(4).float())
    assert state["weight"].device.type == "cpu"


def test_rejects_non_tensor_checkpoint_values_and_tensor_count_overflow(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    source = _write_torch_checkpoint(tmp_path, {"weight": "not-a-tensor"})
    with pytest.raises(GPT2SourceError, match="string keys to tensors"):
        load_checkpoint_state_dict(source, tmp_path)

    source = _write_torch_checkpoint(
        tmp_path, {"one": torch.ones(1), "two": torch.ones(1)}
    )
    with pytest.raises(GPT2SourceError, match="tensor-count"):
        load_checkpoint_state_dict(source, tmp_path, max_tensors=1)
