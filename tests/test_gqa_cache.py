"""Model-bound compact caches work through the actual shared serving primitive."""

from dataclasses import replace

import pytest
import torch

from ai_journey.ddp_cache import KVCache, cached_forward
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gqa_config import GQAConfig
from ai_journey.moe_config import MoEConfig
from ai_journey.training_models import build_training_model
from test_grouped_rotary_attention import config


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_shared_cache_full_incremental_parity_and_native_storage(sparse, heads):
    kind = MoEConfig if sparse else DecoderVariantConfig
    model = (
        build_training_model(GQAConfig(kind(config(), position_kind="rope"), heads))
        .eval()
        .double()
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    full = model(tokens)
    first, cache = cached_forward(model, tokens[:, :2])
    frozen = cache.clone()
    middle, next_cache = cached_forward(model, tokens[:, 2:5], cache)
    final, complete = cached_forward(model, tokens[:, 5:], next_cache)
    torch.testing.assert_close(
        torch.cat((first, middle, final), 1), full, atol=1e-12, rtol=1e-12
    )
    assert cache.keys[0].shape == (1, heads, 2, 2)
    assert complete.storage_bytes == 2 * len(model.h) * 1 * heads * 6 * 2 * 8
    assert all(
        torch.equal(left, right)
        for left, right in zip(cache.keys + cache.values, frozen.keys + frozen.values)
    )


def test_cache_identity_binds_kv_grouping_and_rotary_contract():
    base = DecoderVariantConfig(config(), position_kind="rope")
    model = build_training_model(GQAConfig(base, 2)).eval()
    _, cache = cached_forward(model, torch.tensor([[1, 2]]))
    other = build_training_model(GQAConfig(base, 1)).eval()
    with pytest.raises(ValueError, match="belong"):
        cached_forward(other, torch.tensor([[3]]), cache)
    altered = build_training_model(GQAConfig(replace(base, rope_base=1000), 2)).eval()
    with pytest.raises(ValueError, match="belong"):
        cached_forward(altered, torch.tensor([[3]]), cache)


def test_forged_query_head_cache_is_not_accepted_as_compact_kv_cache():
    model = build_training_model(
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 1)
    ).eval()
    _, cache = cached_forward(model, torch.tensor([[1, 2]]))
    wrong = KVCache(
        cache.model_sha256,
        tuple(tensor.repeat_interleave(4, 1) for tensor in cache.keys),
        tuple(tensor.repeat_interleave(4, 1) for tensor in cache.values),
    )
    with pytest.raises(ValueError, match="belong"):
        cached_forward(model, torch.tensor([[3]]), wrong)


def test_cached_decoder_rejects_mixed_training_modes_and_context_overrun():
    model = build_training_model(
        GQAConfig(DecoderVariantConfig(config(), position_kind="rope"), 2)
    ).eval()
    model.h[0].ln_1.train()
    with pytest.raises(ValueError, match="evaluation"):
        cached_forward(model, torch.tensor([[1]]))
    model.eval()
    _, cache = cached_forward(model, torch.ones(1, 8, dtype=torch.long))
    with pytest.raises(ValueError, match="capacity"):
        cached_forward(model, torch.tensor([[1]]), cache)
