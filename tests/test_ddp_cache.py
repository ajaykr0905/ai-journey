import pytest
import torch
from ai_journey.ddp_cache import KVCache
from ai_journey.ddp_cache import cached_attention
from ai_journey.ddp_cache import cached_forward
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_model import GPT2SelfAttention
from ai_journey.gpt2_config import GPT2Config


def test_cache_accounts_storage_and_clones_without_aliases():
    tensor = torch.ones(1, 2, 3, 4, dtype=torch.float64)
    cache = KVCache("a" * 64, (tensor,), (tensor.clone(),))
    assert cache.length == 3
    assert cache.storage_bytes == 384
    cloned = cache.clone()
    cloned.keys[0].zero_()
    assert cache.keys[0].sum() == 24


def test_cache_rejects_shape_mismatch_and_nonfinite_storage():
    with pytest.raises(ValueError):
        KVCache("a" * 64, (torch.ones(1, 2, 3, 4),), (torch.ones(1, 2, 2, 4),))
    with pytest.raises(ValueError):
        KVCache(
            "a" * 64,
            (torch.full((1, 2, 3, 4), float("nan")),),
            (torch.ones(1, 2, 3, 4),),
        )


def test_cached_attention_matches_full_prefix_for_multi_token_decode():
    torch.manual_seed(48)
    attention = GPT2SelfAttention(GPT2Config(16, 8, 8, 2, 1)).double().eval()
    inputs = torch.randn(2, 6, 8, dtype=torch.float64)
    full = attention(inputs)
    first, state = cached_attention(attention, inputs[:, :3])
    rest, state = cached_attention(attention, inputs[:, 3:], state)
    torch.testing.assert_close(
        torch.cat((first, rest), 1), full, atol=1e-12, rtol=1e-12
    )
    assert state[0].shape == (2, 2, 6, 4)


def test_complete_cached_decoder_matches_all_logits_and_rejects_stale_model():
    torch.manual_seed(48)
    model = GPT2LanguageModel(GPT2Config(16, 8, 8, 2, 2)).double().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    first, cache = cached_forward(model, ids[:, :2])
    rest, cache = cached_forward(model, ids[:, 2:], cache)
    torch.testing.assert_close(
        torch.cat((first, rest), 1), model(ids), atol=1e-12, rtol=1e-12
    )
    assert cache.length == 5
    with torch.no_grad():
        model.wte.weight.add_(0.1)
    with pytest.raises(ValueError, match="does not belong"):
        cached_forward(model, torch.tensor([[6]]), cache)
