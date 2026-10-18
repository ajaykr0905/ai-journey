from __future__ import annotations

from ai_journey.gpt2_config import GPT2_VARIANTS, GPT2Config
from ai_journey.gpt2_interop import TensorSpec, expected_hugging_face_tensors


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
