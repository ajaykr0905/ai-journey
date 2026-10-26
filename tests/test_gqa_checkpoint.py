"""Grouped model tags bind exact restart, provenance and frozen cache behavior."""

from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from ai_journey.ddp_cache import cached_forward
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gqa_checkpoint import (
    load_gqa_checkpoint,
    load_gqa_model_checkpoint,
    save_gqa_checkpoint,
)
from ai_journey.gqa_config import GQAConfig
from ai_journey.gqa_resources import audit_gqa_cache_storage
from ai_journey.gqa_training import build_gqa_optimizer, train_gqa_step
from ai_journey.moe_checkpoint import _digest
from ai_journey.moe_config import MoEConfig
from ai_journey.training_models import build_training_model, training_model_fingerprint
from test_grouped_rotary_attention import config

SOURCE = b"Public grouped-query checkpoint fixture"
CONTRACT = {
    "seed": 53,
    "batch_order": "fixed",
    "balance_weight": 0.01,
    "z_weight": 0.001,
}
INPUTS, TARGETS = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_exact_restart_then_frozen_native_cache_for_every_group_contract(
    tmp_path, sparse, heads
):
    base = (MoEConfig if sparse else DecoderVariantConfig)(
        config(), position_kind="rope"
    )
    grouped = GQAConfig(base, heads)
    model = build_training_model(grouped)
    optimizer = build_gqa_optimizer(model, learning_rate=0.001)
    train_gqa_step(model, optimizer, INPUTS, TARGETS, z_weight=0.001)
    path = tmp_path / "grouped.pt"
    save_gqa_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        training_contract=CONTRACT,
        source_sha256=sha256(SOURCE).hexdigest(),
    )
    full = train_gqa_step(model, optimizer, INPUTS, TARGETS, z_weight=0.001)
    expected, expected_optimizer = (
        training_model_fingerprint(model),
        _digest(optimizer.state_dict()),
    )
    clone = build_training_model(grouped)
    clone_optimizer = build_gqa_optimizer(clone, learning_rate=0.001)
    assert (
        load_gqa_checkpoint(
            path,
            model=clone,
            optimizer=clone_optimizer,
            training_contract=CONTRACT,
            source_sha256=sha256(SOURCE).hexdigest(),
        )
        == 1
    )
    assert (
        train_gqa_step(clone, clone_optimizer, INPUTS, TARGETS, z_weight=0.001) == full
    )
    assert training_model_fingerprint(clone) == expected
    assert _digest(clone_optimizer.state_dict()) == expected_optimizer
    rng = torch.get_rng_state().clone()
    frozen = load_gqa_model_checkpoint(
        path, config=grouped, training_contract=CONTRACT, source=SOURCE
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    assert not any(module.training for module in frozen.modules())
    first, cache = cached_forward(frozen, INPUTS[:, :1])
    rest, complete = cached_forward(frozen, INPUTS[:, 1:], cache)
    torch.testing.assert_close(
        torch.cat((first, rest), 1), frozen(INPUTS), atol=1e-6, rtol=1e-5
    )
    assert (
        audit_gqa_cache_storage(frozen, complete)["logical_bytes"]
        == complete.storage_bytes
    )


@pytest.mark.parametrize("change", ["rotary_base", "kv_heads", "source"])
def test_wrong_declared_contract_is_rejected_without_live_mutation(tmp_path, change):
    base = DecoderVariantConfig(config(), position_kind="rope")
    grouped = GQAConfig(base, 2)
    model = build_training_model(grouped)
    optimizer = build_gqa_optimizer(model, learning_rate=0.001)
    path = tmp_path / "state.pt"
    digest = sha256(SOURCE).hexdigest()
    save_gqa_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=0,
        training_contract=CONTRACT,
        source_sha256=digest,
    )
    changed = (
        GQAConfig(replace(base, rope_base=1000), 2)
        if change == "rotary_base"
        else GQAConfig(base, 1)
        if change == "kv_heads"
        else grouped
    )
    target = build_training_model(changed)
    target_optimizer = build_gqa_optimizer(target, learning_rate=0.001)
    before, rng = training_model_fingerprint(target), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="architecture|source"):
        load_gqa_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            training_contract=CONTRACT,
            source_sha256=sha256(b"other").hexdigest()
            if change == "source"
            else digest,
        )
    assert training_model_fingerprint(target) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_wrong_public_source_prevents_frozen_model_allocation(tmp_path, monkeypatch):
    grouped = GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 1)
    model = build_training_model(grouped)
    path = tmp_path / "state.pt"
    save_gqa_checkpoint(
        path,
        model=model,
        optimizer=build_gqa_optimizer(model, learning_rate=0.001),
        step=0,
        training_contract=CONTRACT,
        source_sha256=sha256(SOURCE).hexdigest(),
    )
    monkeypatch.setattr(
        "ai_journey.moe_inference.build_training_model",
        lambda *a, **k: pytest.fail("must not allocate"),
    )
    with pytest.raises(ValueError, match="source"):
        load_gqa_model_checkpoint(
            path,
            config=grouped,
            training_contract=CONTRACT,
            source=b"wrong public source",
        )


def test_existing_hugging_face_loader_does_not_claim_gqa_weight_compatibility():
    from ai_journey.gpt2_interop import load_hugging_face_weights

    model = build_training_model(
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 2)
    )
    with pytest.raises(TypeError, match="model"):
        load_hugging_face_weights(model, {})
