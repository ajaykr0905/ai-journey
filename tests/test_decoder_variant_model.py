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
