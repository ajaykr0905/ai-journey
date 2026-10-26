from __future__ import annotations

import itertools

import pytest
import torch

from ai_journey.decoder_variant_config import DecoderVariantConfig
from ai_journey.decoder_variant_model import DecoderVariantModel
from ai_journey.gpt2_config import GPT2Config, GPT2ConfigError
from ai_journey.gpt2_model import GPT2LanguageModel
from ai_journey.modern_architecture import RMSNorm
from ai_journey.rotary_attention import RotarySelfAttention
from ai_journey.swiglu import SwiGLU


def _base() -> GPT2Config:
    return GPT2Config(
        vocab_size=23,
        context_length=6,
        embedding_dim=8,
        head_count=2,
        layer_count=2,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def test_default_variant_loads_baseline_weights_and_matches_logits_exactly() -> None:
    torch.manual_seed(52)
    baseline = GPT2LanguageModel(_base()).eval()
    variant = DecoderVariantModel(DecoderVariantConfig(_base())).eval()
    assert variant.state_dict().keys() == baseline.state_dict().keys()
    variant.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    torch.testing.assert_close(variant(tokens), baseline(tokens), rtol=0, atol=0)
    assert variant.parameter_count == baseline.parameter_count
    assert variant.lm_head.weight is variant.wte.weight


@pytest.mark.parametrize(
    ("norm_kind", "mlp_kind", "position_kind"),
    list(
        itertools.product(
            ("layernorm", "rmsnorm"),
            ("gelu", "swiglu"),
            ("learned", "rope"),
        )
    ),
)
def test_all_component_combinations_have_finite_logits_and_gradients(
    norm_kind: str, mlp_kind: str, position_kind: str
) -> None:
    torch.manual_seed(52)
    config = DecoderVariantConfig(
        _base(),
        norm_kind=norm_kind,
        mlp_kind=mlp_kind,
        position_kind=position_kind,
    )
    model = DecoderVariantModel(config)
    tokens = torch.tensor([[1, 2, 3, 4]])

    logits = model(tokens)
    logits.square().mean().backward()

    assert logits.shape == (1, 4, 23)
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert isinstance(model.h[0].ln_1, RMSNorm) == (norm_kind == "rmsnorm")
    assert isinstance(model.h[0].attn, RotarySelfAttention) == (position_kind == "rope")
    if position_kind == "rope":
        assert model.wpe is None
    else:
        assert model.wpe is not None
    if mlp_kind == "swiglu":
        assert isinstance(model.h[0].mlp[0], SwiGLU)


@pytest.mark.parametrize(
    ("tokens", "error", "message"),
    [
        (torch.tensor([1, 2]), TypeError, "rank-two"),
        (torch.tensor([[1.0]]), TypeError, "torch.long"),
        (torch.empty((0, 1), dtype=torch.long), GPT2ConfigError, "non-empty"),
        (torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), GPT2ConfigError, "context_length"),
        (torch.tensor([[-1]]), GPT2ConfigError, "out-of-range"),
        (torch.tensor([[23]]), GPT2ConfigError, "out-of-range"),
    ],
)
def test_variant_rejects_invalid_token_batches(
    tokens: torch.Tensor, error: type[Exception], message: str
) -> None:
    model = DecoderVariantModel(DecoderVariantConfig(_base()))
    with pytest.raises(error, match=message):
        model(tokens)


@pytest.mark.parametrize(
    ("norm_kind", "mlp_kind"),
    list(itertools.product(("layernorm", "rmsnorm"), ("gelu", "swiglu"))),
)
def test_rope_variant_cached_chunks_match_full_logits_across_blocks(
    norm_kind: str, mlp_kind: str
) -> None:
    torch.manual_seed(52)
    model = (
        DecoderVariantModel(
            DecoderVariantConfig(
                _base(), norm_kind=norm_kind, mlp_kind=mlp_kind, position_kind="rope"
            )
        )
        .double()
        .eval()
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    full_logits = model(tokens)

    first_logits, first_cache = model.forward_with_cache(tokens[:, :2])
    snapshot = tuple((key.clone(), value.clone()) for key, value in first_cache)
    middle_logits, middle_cache = model.forward_with_cache(
        tokens[:, 2:4], past_key_values=first_cache
    )
    last_logits, final_cache = model.forward_with_cache(
        tokens[:, 4:], past_key_values=middle_cache
    )
    chunked_logits = torch.cat((first_logits, middle_logits, last_logits), dim=1)

    torch.testing.assert_close(chunked_logits, full_logits, rtol=0, atol=1e-12)
    assert len(final_cache) == _base().layer_count
    for layer_index, ((old_key, old_value), (saved_key, saved_value)) in enumerate(
        zip(first_cache, snapshot)
    ):
        assert torch.equal(old_key, saved_key), layer_index
        assert torch.equal(old_value, saved_value), layer_index
        assert final_cache[layer_index][0].shape == (2, 2, 6, 4)
        assert final_cache[layer_index][1].shape == (2, 2, 6, 4)
        assert torch.equal(final_cache[layer_index][0][:, :, :2], old_key)
        assert torch.equal(final_cache[layer_index][1][:, :, :2], old_value)


def test_rope_variant_prefill_matches_ordinary_forward_exactly() -> None:
    torch.manual_seed(52)
    model = DecoderVariantModel(
        DecoderVariantConfig(_base(), position_kind="rope")
    ).eval()
    tokens = torch.tensor([[1, 2, 3]])

    logits, cache = model.forward_with_cache(tokens)

    assert torch.equal(logits, model(tokens))
    assert len(cache) == 2


def test_learned_position_variant_rejects_cached_forward() -> None:
    model = DecoderVariantModel(DecoderVariantConfig(_base())).eval()

    with pytest.raises(GPT2ConfigError, match="RoPE"):
        model.forward_with_cache(torch.tensor([[1]]))


def test_rope_variant_rejects_cache_that_exceeds_context() -> None:
    model = DecoderVariantModel(
        DecoderVariantConfig(_base(), position_kind="rope")
    ).eval()
    _, cache = model.forward_with_cache(torch.tensor([[1, 2, 3, 4, 5]]))

    with pytest.raises(GPT2ConfigError, match="context_length"):
        model.forward_with_cache(torch.tensor([[6, 7]]), past_key_values=cache)


def test_rope_variant_rejects_malformed_layer_caches() -> None:
    model = DecoderVariantModel(
        DecoderVariantConfig(_base(), position_kind="rope")
    ).eval()
    tokens = torch.tensor([[1]])
    key = torch.zeros(1, 2, 2, 4)
    value = torch.zeros(1, 2, 2, 4)
    valid = ((key, value), (key.clone(), value.clone()))

    invalid = [
        (((key, value),), "one pair per layer"),
        (((key, value), (key,)), "key/value pair"),
        (((key, value), (key, [])), "value must be a torch.Tensor"),
        (((key, value), (torch.zeros(2, 2, 2, 4), value)), "invalid cache shape"),
        (((key, value), (key, torch.zeros(1, 2, 3, 4))), "lengths differ"),
        (
            ((key, value), (torch.zeros(1, 2, 1, 4), value[:, :, :1])),
            "share past length",
        ),
        (((key, value), (key.double(), value)), "dtype and device"),
        (
            ((key, value), (key, torch.empty(1, 2, 2, 4, device="meta"))),
            "dtype and device",
        ),
    ]
    for malformed, message in invalid:
        with pytest.raises((TypeError, GPT2ConfigError), match=message):
            model.forward_with_cache(tokens, past_key_values=malformed)  # type: ignore[arg-type]

    logits, extended = model.forward_with_cache(tokens, past_key_values=valid)
    assert logits.shape == (1, 1, 23)
    assert all(layer[0].shape[2] == 3 for layer in extended)


@pytest.mark.parametrize(
    ("tokens", "error", "message"),
    [
        ([], TypeError, "torch.Tensor"),
        (torch.tensor([1, 2]), TypeError, "rank-two"),
        (torch.tensor([[1.0]]), TypeError, "torch.long"),
        (torch.empty((0, 1), dtype=torch.long), GPT2ConfigError, "non-empty"),
        (torch.tensor([[-1]]), GPT2ConfigError, "out-of-range"),
        (torch.tensor([[23]]), GPT2ConfigError, "out-of-range"),
    ],
)
def test_rope_variant_cached_forward_rejects_invalid_tokens(
    tokens: object, error: type[Exception], message: str
) -> None:
    model = DecoderVariantModel(DecoderVariantConfig(_base(), position_kind="rope"))
    with pytest.raises(error, match=message):
        model.forward_with_cache(tokens)  # type: ignore[arg-type]
