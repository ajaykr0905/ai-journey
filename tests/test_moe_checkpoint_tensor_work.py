"""Archive bytes cannot bound the logical work of aliased/expanded tensors."""

import pytest
import torch

from ai_journey.moe_checkpoint import (
    MAX_CHECKPOINT_HASH_ITEMS,
    MAX_CHECKPOINT_LOGICAL_BYTES,
    _admit_digest_work,
    _digest,
    read_moe_checkpoint,
)
from test_moe_checkpoint import save
from test_moe_training import state


def test_small_archive_cannot_trigger_billion_element_tensor_hashing(
    tmp_path, monkeypatch
):
    model, optimizer = state()
    path = tmp_path / "state.pt"
    save(path, model, optimizer, 0)
    payload = torch.load(path, weights_only=True)
    expanded = torch.ones(1).expand(2**30)
    assert expanded.untyped_storage().nbytes() == 4
    payload["model_state"]["wte.weight"] = expanded
    torch.save(payload, path)
    assert path.stat().st_size < 1024 * 1024
    original = torch.isfinite

    def bounded_finite(tensor, *args, **kwargs):
        if tensor.numel() >= 2**30:
            pytest.fail("logical-work admission must precede elementwise allocation")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "isfinite", bounded_finite)
    with pytest.raises(ValueError, match="bounded weights-only") as error:
        read_moe_checkpoint(path)
    assert isinstance(error.value.__cause__, ValueError)
    assert "logical tensor work exceeds its budget" in str(error.value.__cause__)


def test_tensor_aliases_are_charged_each_time_the_digest_would_materialize_them():
    shared = torch.ones(1).expand(1024 * 1024)
    # The digest must not get a fresh allowance for each individual leaf.
    with pytest.raises(ValueError, match="logical.*budget"):
        _digest([shared] * 129)


def test_deeply_nested_checkpoint_data_fails_a_declared_hash_work_budget():
    nested = 1
    for _ in range(70):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting.*budget"):
        _digest(nested)


def test_small_noncontiguous_tensors_preserve_exact_value_seals():
    value = torch.arange(12, dtype=torch.float64).reshape(3, 4).transpose(0, 1)
    assert not value.is_contiguous()
    assert _digest(value) == _digest(value.contiguous())
    assert _digest([value, value]) == _digest([value.clone(), value.clone()])


def test_logical_work_exact_boundary_is_checked_without_tensor_operations():
    scalar = torch.ones(1)
    count = MAX_CHECKPOINT_LOGICAL_BYTES // scalar.element_size()
    _admit_digest_work(scalar.expand(count))
    with pytest.raises(ValueError, match="logical.*budget"):
        _admit_digest_work(scalar.expand(count + 1))


def test_container_work_budget_cannot_be_reset_by_small_scalar_leaves():
    _admit_digest_work([0] * (MAX_CHECKPOINT_HASH_ITEMS - 1))
    with pytest.raises(ValueError, match="items.*budget"):
        _admit_digest_work([0] * MAX_CHECKPOINT_HASH_ITEMS)
