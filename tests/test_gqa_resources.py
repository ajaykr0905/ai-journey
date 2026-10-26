"""Logical inventories must match real retained CPU cache storage and work."""

import pytest
import torch

from ai_journey.ddp_cache import cached_forward
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gqa_config import GQAConfig
from ai_journey.gqa_resources import (
    audit_gqa_cache_storage,
    estimate_gqa_execution,
    gqa_parameter_inventory,
)
from ai_journey.moe_config import MoEConfig
from ai_journey.resource_accounting import DecoderSpec, kv_cache_memory
from ai_journey.training_models import build_training_model
from test_grouped_rotary_attention import config


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_inventory_matches_actual_stored_parameters(sparse, kind, heads):
    base = (MoEConfig if sparse else DecoderVariantConfig)(
        config(), position_kind="rope", norm_kind="rmsnorm", mlp_kind=kind
    )
    grouped = GQAConfig(base, heads)
    model = build_training_model(grouped)
    assert sum(gqa_parameter_inventory(grouped).values()) == model.parameter_count


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_returned_native_cache_owns_exact_compact_backing_storage(length, heads):
    grouped = GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), heads)
    model = build_training_model(grouped).eval()
    _, cache = cached_forward(model, torch.ones(1, length, dtype=torch.long))
    measured = audit_gqa_cache_storage(model, cache)
    plan = estimate_gqa_execution(grouped, batch_size=1, sequence_length=length)
    assert (
        measured["logical_bytes"]
        == measured["retained_storage_bytes"]
        == plan["kv_cache_bytes"]
    )
    analytical = DecoderSpec(256, 1, 8, 4, heads, 32)
    assert (
        kv_cache_memory(
            analytical, batch_size=1, sequence_length=length, dtype="float32"
        )["total"]
        == measured["retained_storage_bytes"]
    )


def test_reducing_kv_heads_changes_cache_storage_but_not_query_score_work():
    base = DecoderVariantConfig(config(), position_kind="rope")
    full = estimate_gqa_execution(
        GQAConfig(base, 4), batch_size=2, sequence_length=6, query_tokens=1
    )
    grouped = estimate_gqa_execution(
        GQAConfig(base, 1), batch_size=2, sequence_length=6, query_tokens=1
    )
    assert full["kv_cache_bytes"] == 4 * grouped["kv_cache_bytes"]
    assert full["score_elements"] == grouped["score_elements"]


@pytest.mark.parametrize(
    "constant,match",
    [("MAX_GQA_CACHE_BYTES", "KV-cache"), ("MAX_GQA_SCORE_ELEMENTS", "score-work")],
)
@pytest.mark.parametrize("cached", [False, True])
def test_work_and_cache_admission_precede_embedding_allocation(
    monkeypatch, constant, match, cached
):
    model = build_training_model(
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 1)
    ).eval()
    monkeypatch.setattr("ai_journey.gqa_resources." + constant, 1)
    monkeypatch.setattr(model.wte, "forward", lambda _: pytest.fail("must not embed"))
    with pytest.raises(ValueError, match=match):
        if cached:
            model.forward_with_cache(torch.tensor([[1, 2]]))
        else:
            model(torch.tensor([[1, 2]]))


def test_huge_zero_stride_prefix_rejects_before_finite_scan_or_materialization(
    monkeypatch,
):
    from ai_journey.grouped_rotary_attention import GroupedRotarySelfAttention
    from ai_journey.gpt2_config import GPT2Config

    attention = GroupedRotarySelfAttention(
        GPT2Config(256, 1_000_000_001, 8, 4, 1), kv_heads=1
    ).eval()
    prefix = torch.ones(1, 1, 1, 2).expand(1, 1, 1_000_000_000, 2)
    tokens = torch.ones(1, 1, 8)
    monkeypatch.setattr(
        torch, "isfinite", lambda _: pytest.fail("must reject before finite scan")
    )
    with pytest.raises(ValueError, match="budget"):
        attention.forward_with_cache(tokens, past_key=prefix, past_value=prefix)
