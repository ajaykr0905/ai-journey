from __future__ import annotations

from ai_journey.gpt2_config import GPT2_VARIANTS, GPT2Config
import pytest
import torch

from ai_journey.gpt2_interop import (
    GPT2InteropError,
    TensorSpec,
    expected_hugging_face_tensors,
    known_hugging_face_buffers,
    validate_tied_embedding_alias,
    validate_hugging_face_state_dict,
)


def _tiny_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=11,
        context_length=4,
        embedding_dim=4,
        head_count=2,
        layer_count=2,
        embedding_dropout=0,
        attention_dropout=0,
        residual_dropout=0,
    )


def test_expected_tensor_plan_covers_every_parameter_and_tied_head_alias() -> None:
    specs = expected_hugging_face_tensors(_tiny_config())

    assert len(specs) == 29
    assert specs["transformer.wte.weight"] == TensorSpec((11, 4))
    assert specs["transformer.wpe.weight"] == TensorSpec((4, 4))
    assert specs["lm_head.weight"] == TensorSpec((11, 4))
    assert specs["transformer.h.1.ln_2.bias"] == TensorSpec((4,))


def test_conv1d_weights_are_the_only_transposed_tensor_classes() -> None:
    specs = expected_hugging_face_tensors(_tiny_config())

    transposed = {key for key, spec in specs.items() if spec.transpose}

    assert transposed == {
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.1.attn.c_proj.weight",
        "transformer.h.1.mlp.c_fc.weight",
        "transformer.h.1.mlp.c_proj.weight",
    }


def test_gpt2_124m_plan_has_the_expected_tensor_count() -> None:
    specs = expected_hugging_face_tensors(GPT2_VARIANTS["gpt2"])

    assert len(specs) == 149
    assert specs["transformer.h.11.attn.c_attn.weight"].shape == (768, 2_304)
    assert specs["transformer.h.11.mlp.c_proj.weight"].shape == (3_072, 768)


def _valid_state_dict() -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros(spec.shape, dtype=torch.float32)
        for key, spec in expected_hugging_face_tensors(_tiny_config()).items()
    }


def test_validates_complete_finite_hugging_face_state() -> None:
    state = _valid_state_dict()
    state["transformer.h.0.attn.bias"] = torch.ones(1, 1, 4, 4)

    ignored = validate_hugging_face_state_dict(state, _tiny_config())

    assert ignored == ("transformer.h.0.attn.bias",)
    assert len(known_hugging_face_buffers(_tiny_config())) == 4


def test_rejects_missing_and_unexpected_state_dict_keys() -> None:
    missing = _valid_state_dict()
    del missing["transformer.ln_f.bias"]
    unexpected = _valid_state_dict()
    unexpected["transformer.secret.weight"] = torch.zeros(1)

    with pytest.raises(GPT2InteropError, match="missing.*ln_f.bias"):
        validate_hugging_face_state_dict(missing, _tiny_config())
    with pytest.raises(GPT2InteropError, match="unexpected.*secret"):
        validate_hugging_face_state_dict(unexpected, _tiny_config())


def test_rejects_wrong_shape_dtype_and_nonfinite_tensors() -> None:
    wrong_shape = _valid_state_dict()
    wrong_shape["transformer.wpe.weight"] = torch.zeros(3, 4)
    integer = _valid_state_dict()
    integer["transformer.ln_f.weight"] = torch.ones(4, dtype=torch.int64)
    nonfinite = _valid_state_dict()
    nonfinite["transformer.ln_f.bias"][2] = float("nan")

    with pytest.raises(GPT2InteropError, match="shape"):
        validate_hugging_face_state_dict(wrong_shape, _tiny_config())
    with pytest.raises(GPT2InteropError, match="floating-point"):
        validate_hugging_face_state_dict(integer, _tiny_config())
    with pytest.raises(GPT2InteropError, match="finite"):
        validate_hugging_face_state_dict(nonfinite, _tiny_config())


def test_requires_the_language_head_to_match_the_token_embedding_exactly() -> None:
    state = _valid_state_dict()
    state["lm_head.weight"][0, 0] = 1

    with pytest.raises(GPT2InteropError, match="exactly equal"):
        validate_hugging_face_state_dict(state, _tiny_config())


def test_tied_embedding_check_rejects_missing_alias_inputs() -> None:
    with pytest.raises(GPT2InteropError, match="require wte and lm_head"):
        validate_tied_embedding_alias({})
