"""Cache reuse binds full decoder architecture without changing weight hashes."""

from dataclasses import replace

import pytest
import torch

from ai_journey.ddp_cache import KVCache, cached_forward
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.gpt2_config import GPT2Config
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.gpt2_training import gpt2_model_fingerprint
from ai_journey.gqa_config import GQAConfig
from ai_journey.prefix_cache import PrefixKVStore, prefix_identity
from ai_journey.training_models import build_training_model


def models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        config = GPT2Config(16, 8, 8, 2, 2)
        original = GPT2LanguageModel(config).double().eval()
        changed = (
            GPT2LanguageModel(replace(config, layer_norm_epsilon=0.1)).double().eval()
        )
        changed.load_state_dict(original.state_dict())
    return original, changed


def test_identical_weights_do_not_admit_another_layer_norm_architecture():
    original, changed = models()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    assert gpt2_model_fingerprint(original) == gpt2_model_fingerprint(changed)
    _, cache = cached_forward(original, tokens[:, :2])

    with pytest.raises(ValueError, match="architecture"):
        cached_forward(changed, tokens[:, 2:], cache)


@pytest.mark.parametrize("kind", ["baseline", "rotary", "grouped"])
def test_prefix_pool_copies_retain_valid_decoder_reuse(kind):
    config = GPT2Config(16, 8, 8, 2, 2)
    if kind != "baseline":
        config = DecoderVariantConfig(config, position_kind="rope")
    if kind == "grouped":
        config = GQAConfig(config, 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        model = build_training_model(config).double().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    logits, cache = cached_forward(model, tokens[:, :2])
    identity = prefix_identity(model)
    store = PrefixKVStore(max_bytes=16384)
    assert store.publish(identity, (1, 2), cache, logits[:, -1, :])

    with store.acquire(identity, (1, 2, 3, 4, 5)) as match:
        actual, _ = cached_forward(model, tokens[:, 2:], match.cache)
        torch.testing.assert_close(
            actual, model(tokens)[:, 2:, :], atol=1e-12, rtol=1e-12
        )


def test_prefix_publication_cannot_relabel_another_architectures_cache():
    original, changed = models()
    tokens = (1, 2)
    logits, cache = cached_forward(original, torch.tensor([tokens]))
    store = PrefixKVStore(max_bytes=16384)

    with pytest.raises(ValueError, match="model contract"):
        store.publish(prefix_identity(changed), tokens, cache, logits[:, -1, :])
    assert store.snapshot()["entries"] == 0


def test_cloned_generated_cache_keeps_valid_same_architecture_replay():
    original, _ = models()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    _, cache = cached_forward(original, tokens[:, :2])
    copied = cache.clone()

    actual, _ = cached_forward(original, tokens[:, 2:], copied)
    torch.testing.assert_close(
        actual, original(tokens)[:, 2:, :], atol=1e-12, rtol=1e-12
    )
    assert copied.model_sha256 == gpt2_model_fingerprint(original)
    assert copied.architecture_sha256 == prefix_identity(original).architecture_sha256


def test_three_argument_cache_remains_constructible_but_cannot_infer_architecture():
    original, _ = models()
    tokens = torch.tensor([[1, 2]])
    logits, bound = cached_forward(original, tokens)
    legacy = KVCache(bound.model_sha256, bound.keys, bound.values)
    assert legacy.storage_bytes == bound.storage_bytes

    with pytest.raises(ValueError, match="architecture"):
        cached_forward(original, torch.tensor([[3]]), legacy.clone())
    store = PrefixKVStore(max_bytes=16384)
    with pytest.raises(ValueError, match="model contract"):
        store.publish(prefix_identity(original), (1, 2), legacy, logits[:, -1, :])
    assert store.snapshot()["entries"] == 0


@pytest.mark.parametrize("identity", [True, "A" * 64, "short"])
def test_cache_rejects_malformed_explicit_architecture_identity(identity):
    original, _ = models()
    _, bound = cached_forward(original, torch.tensor([[1, 2]]))

    with pytest.raises(ValueError, match="architecture"):
        KVCache(bound.model_sha256, bound.keys, bound.values, identity)
