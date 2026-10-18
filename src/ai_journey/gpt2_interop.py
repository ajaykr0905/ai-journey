"""Strict Hugging Face state-dict interoperability for the GPT-2 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from .gpt2_config import GPT2Config


@dataclass(frozen=True)
class TensorSpec:
    """Expected Hugging Face tensor shape and required conversion."""

    shape: tuple[int, ...]
    transpose: bool = False


def expected_hugging_face_tensors(config: GPT2Config) -> dict[str, TensorSpec]:
    """Return the complete expected GPT-2 Hugging Face tensor contract."""

    if not isinstance(config, GPT2Config):
        raise TypeError("config must be GPT2Config")
    width = config.embedding_dim
    specs = {
        "transformer.wte.weight": TensorSpec((config.vocab_size, width)),
        "transformer.wpe.weight": TensorSpec((config.context_length, width)),
    }
    for index in range(config.layer_count):
        prefix = f"transformer.h.{index}"
        specs.update(
            {
                f"{prefix}.ln_1.weight": TensorSpec((width,)),
                f"{prefix}.ln_1.bias": TensorSpec((width,)),
                f"{prefix}.attn.c_attn.weight": TensorSpec(
                    (width, 3 * width), transpose=True
                ),
                f"{prefix}.attn.c_attn.bias": TensorSpec((3 * width,)),
                f"{prefix}.attn.c_proj.weight": TensorSpec(
                    (width, width), transpose=True
                ),
                f"{prefix}.attn.c_proj.bias": TensorSpec((width,)),
                f"{prefix}.ln_2.weight": TensorSpec((width,)),
                f"{prefix}.ln_2.bias": TensorSpec((width,)),
                f"{prefix}.mlp.c_fc.weight": TensorSpec(
                    (width, 4 * width), transpose=True
                ),
                f"{prefix}.mlp.c_fc.bias": TensorSpec((4 * width,)),
                f"{prefix}.mlp.c_proj.weight": TensorSpec(
                    (4 * width, width), transpose=True
                ),
                f"{prefix}.mlp.c_proj.bias": TensorSpec((width,)),
            }
        )
    specs.update(
        {
            "transformer.ln_f.weight": TensorSpec((width,)),
            "transformer.ln_f.bias": TensorSpec((width,)),
            "lm_head.weight": TensorSpec((config.vocab_size, width)),
        }
    )
    return specs
