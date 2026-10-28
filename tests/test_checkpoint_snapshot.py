"""A journal digest binds the exact bounded archive bytes deserialized."""

from hashlib import sha256
from io import BytesIO

import pytest
import torch

from ai_journey.checkpoint_archive import load_checkpoint_archive


def test_snapshot_is_deserialized_even_if_path_changes_after_digest(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1.0])}, path)
    digest = sha256(path.read_bytes()).hexdigest()
    original = torch.load

    def replace_before_load(stream, **kwargs):
        assert isinstance(stream, BytesIO)
        torch.save({"value": torch.tensor([2.0])}, path)
        return original(stream, **kwargs)

    monkeypatch.setattr(torch, "load", replace_before_load)
    payload = load_checkpoint_archive(path, max_bytes=10000, expected_sha256=digest)
    assert payload["value"].item() == 1
    assert original(path, weights_only=True)["value"].item() == 2


def test_archive_admission_binds_loaded_bytes_without_an_expected_digest(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1.0])}, path)
    assert path.stat().st_size < 10000
    original = torch.load

    def grow_after_admission(stream, **kwargs):
        torch.save({"value": torch.ones(10000)}, path)
        assert path.stat().st_size > 10000
        return original(stream, **kwargs)

    monkeypatch.setattr(torch, "load", grow_after_admission)
    payload = load_checkpoint_archive(path, max_bytes=10000)
    assert torch.equal(payload["value"], torch.tensor([1.0]))
    assert original(path, weights_only=True)["value"].numel() == 10000


@pytest.mark.parametrize("digest", ["0" * 64, True, "A" * 64, "bad"])
def test_bad_snapshot_identity_rejects_before_deserialization(
    tmp_path, monkeypatch, digest
):
    path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.ones(1)}, path)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: pytest.fail("digest admission must precede loading"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint_archive(path, max_bytes=10000, expected_sha256=digest)
