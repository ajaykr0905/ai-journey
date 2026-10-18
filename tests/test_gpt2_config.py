from __future__ import annotations

import pytest

from ai_journey.gpt2_config import (
    GPT2_VARIANTS,
    GPT2Config,
    GPT2ConfigError,
    config_from_hugging_face,
    require_variant,
)


def _hf_config(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_type": "gpt2",
        "vocab_size": 50_257,
        "n_positions": 1_024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "layer_norm_epsilon": 1e-5,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "activation_function": "gelu_new",
    }
    payload.update(overrides)
    return payload


def test_extracts_exact_gpt2_contract_from_hugging_face_config() -> None:
    config = config_from_hugging_face(_hf_config())

    assert config == GPT2_VARIANTS["gpt2"]
    assert config.head_dim == 64
    assert len(config.fingerprint()) == 64


def test_architecture_fingerprint_changes_with_model_shape() -> None:
    base = GPT2Config(100, 16, 32, 4, 2)
    wider = GPT2Config(100, 16, 64, 4, 2)

    assert base.fingerprint() != wider.fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vocab_size", 0, "vocab_size must be positive"),
        ("context_length", True, "context_length must be an integer"),
        ("embedding_dim", 30, "embedding_dim must be divisible"),
        ("layer_norm_epsilon", float("nan"), "positive and finite"),
        ("attention_dropout", 1.0, "finite and in"),
        ("activation_function", "gelu", "must be 'gelu_new'"),
    ],
)
def test_rejects_invalid_architecture_controls(
    field: str, value: object, message: str
) -> None:
    values = {
        "vocab_size": 100,
        "context_length": 16,
        "embedding_dim": 32,
        "head_count": 4,
        "layer_count": 2,
    }
    values[field] = value

    with pytest.raises((TypeError, GPT2ConfigError), match=message):
        GPT2Config(**values)  # type: ignore[arg-type]


def test_hugging_face_config_requires_model_type_and_every_architecture_field() -> None:
    wrong_type = _hf_config(model_type="bert")
    missing = _hf_config()
    del missing["n_layer"]

    with pytest.raises(GPT2ConfigError, match="model_type"):
        config_from_hugging_face(wrong_type)
    with pytest.raises(GPT2ConfigError, match="n_layer"):
        config_from_hugging_face(missing)


def test_named_variant_requires_an_exact_contract() -> None:
    require_variant(GPT2_VARIANTS["gpt2"], "gpt2")

    with pytest.raises(GPT2ConfigError, match="unsupported"):
        require_variant(GPT2_VARIANTS["gpt2"], "gpt2-medium")
    with pytest.raises(GPT2ConfigError, match="exactly match"):
        require_variant(GPT2Config(100, 16, 32, 4, 2), "gpt2")
