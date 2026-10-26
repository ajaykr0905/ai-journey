from dataclasses import replace
import itertools

import pytest
import torch

from ai_journey.ddp_cache import cached_forward
from ai_journey.ddp_generation import SamplingConfig, generate_tokens
from ai_journey.ddp_service import GenerationRequest, GenerationService
from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_model import DecoderVariantModel
from ai_journey.gpt2_config import GPT2Config


@pytest.fixture(params=list(itertools.product([False, True], repeat=3)))
def model(request):
    rms, swiglu, rope = request.param
    config = DecoderVariantConfig(
        GPT2Config(8, 4, 8, 2, 2),
        norm_kind="rmsnorm" if rms else "layernorm",
        mlp_kind="swiglu" if swiglu else "gelu",
        position_kind="rope" if rope else "learned",
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(52)
        return DecoderVariantModel(config).double().eval()


def test_every_modern_architecture_matches_full_logits_across_cache_chunks(model):
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(tokens)
        prefix, cache = cached_forward(model, tokens[:, :2])
        suffix, updated = cached_forward(model, tokens[:, 2:], cache)
    torch.testing.assert_close(
        torch.cat((prefix, suffix), dim=1), expected, atol=1e-10, rtol=1e-10
    )
    assert cache.length == 2
    assert updated.length == 4
    assert updated.model_sha256 == cache.model_sha256


def test_modern_cache_is_bound_to_architecture_even_when_shapes_match(model):
    tokens = torch.tensor([[1, 2]])
    _, cache = cached_forward(model, tokens)
    other = (
        DecoderVariantModel(replace(model.config, rope_base=model.config.rope_base + 1))
        .double()
        .eval()
    )
    other.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="belong"):
        cached_forward(other, tokens[:, :1], cache)


def test_modern_cached_generation_matches_uncached_through_context_rollover(model):
    config = SamplingConfig(max_new_tokens=7, seed=52, temperature=0)
    expected = generate_tokens(model, (1, 2), config)
    assert generate_tokens(model, (1, 2), config, use_cache=True) == expected


def test_immutable_service_accepts_modern_decoder_snapshot(model):
    service = GenerationService(model)
    request = GenerationRequest(
        "modern", (1, 2), SamplingConfig(max_new_tokens=5, temperature=0)
    )
    expected = generate_tokens(model, request.prompt, request.sampling)
    actual = service.generate(request)
    assert actual["tokens"] == expected["tokens"]
    assert actual["model_sha256"] == service.model_sha256
    with torch.no_grad():
        model.wte.weight.zero_()
    assert service.generate(request) == actual
