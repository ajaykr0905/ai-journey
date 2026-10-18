"""Strict Hugging Face state-dict interoperability for the GPT-2 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import Tensor

from .gpt2_config import GPT2Config


class GPT2InteropError(ValueError):
    """Raised when external GPT-2 state violates the interop contract."""


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


def known_hugging_face_buffers(config: GPT2Config) -> frozenset[str]:
    """Return legacy causal-mask buffers that carry no learned state."""

    if not isinstance(config, GPT2Config):
        raise TypeError("config must be GPT2Config")
    return frozenset(
        key
        for index in range(config.layer_count)
        for key in (
            f"transformer.h.{index}.attn.bias",
            f"transformer.h.{index}.attn.masked_bias",
        )
    )


def validate_hugging_face_state_dict(
    state_dict: Mapping[str, Tensor], config: GPT2Config
) -> tuple[str, ...]:
    """Validate the complete external key, shape, dtype, and value contract."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    specs = expected_hugging_face_tensors(config)
    keys = set(state_dict)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("state_dict keys must be strings")
    missing = sorted(set(specs) - keys)
    if missing:
        raise GPT2InteropError("missing Hugging Face tensors: " + ", ".join(missing))
    ignored = sorted(keys & known_hugging_face_buffers(config))
    unexpected = sorted(keys - set(specs) - set(ignored))
    if unexpected:
        raise GPT2InteropError(
            "unexpected Hugging Face tensors: " + ", ".join(unexpected)
        )
    for key, spec in specs.items():
        tensor = state_dict[key]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if tuple(tensor.shape) != spec.shape:
            raise GPT2InteropError(
                f"{key} has shape {tuple(tensor.shape)}, expected {spec.shape}"
            )
        if not tensor.is_floating_point():
            raise GPT2InteropError(f"{key} must use a floating-point dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise GPT2InteropError(f"{key} must contain only finite values")
    validate_tied_embedding_alias(state_dict)
    return tuple(ignored)


def validate_tied_embedding_alias(state_dict: Mapping[str, Tensor]) -> None:
    """Require the exported language-model head to equal the token embedding."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    try:
        token_embedding = state_dict["transformer.wte.weight"]
        language_head = state_dict["lm_head.weight"]
    except KeyError as exc:
        raise GPT2InteropError(
            "tied embeddings require wte and lm_head tensors"
        ) from exc
    if not isinstance(token_embedding, Tensor) or not isinstance(language_head, Tensor):
        raise TypeError("tied embedding values must be torch.Tensor objects")
    if not torch.equal(token_embedding, language_head):
        raise GPT2InteropError(
            "lm_head.weight must exactly equal transformer.wte.weight"
        )


def convert_hugging_face_state_dict(
    state_dict: Mapping[str, Tensor], config: GPT2Config
) -> dict[str, Tensor]:
    """Convert validated Hugging Face names and Conv1D layouts to PyTorch."""

    validate_hugging_face_state_dict(state_dict, config)
    specs = expected_hugging_face_tensors(config)
    converted: dict[str, Tensor] = {}
    for external_key, spec in specs.items():
        if external_key.startswith("transformer."):
            internal_key = external_key.removeprefix("transformer.")
        else:
            internal_key = external_key
        value = state_dict[external_key].detach()
        converted[internal_key] = (
            value.transpose(0, 1).contiguous().clone()
            if spec.transpose
            else value.contiguous().clone()
        )
    return converted
