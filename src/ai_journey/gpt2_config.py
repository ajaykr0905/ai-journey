"""Validated architecture contracts for GPT-2 interoperability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


class GPT2ConfigError(ValueError):
    """Raised when a GPT-2 architecture contract is invalid."""


@dataclass(frozen=True)
class GPT2Config:
    """The architecture fields required to reproduce a GPT-2 decoder."""

    vocab_size: int
    context_length: int
    embedding_dim: int
    head_count: int
    layer_count: int
    layer_norm_epsilon: float = 1e-5
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    activation_function: str = "gelu_new"

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "embedding_dim",
            "head_count",
            "layer_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise GPT2ConfigError(f"{name} must be positive")
        if self.embedding_dim % self.head_count:
            raise GPT2ConfigError("embedding_dim must be divisible by head_count")
        if (
            isinstance(self.layer_norm_epsilon, bool)
            or not isinstance(self.layer_norm_epsilon, (int, float))
            or not math.isfinite(self.layer_norm_epsilon)
            or self.layer_norm_epsilon <= 0
        ):
            raise GPT2ConfigError("layer_norm_epsilon must be positive and finite")
        for name in (
            "embedding_dropout",
            "attention_dropout",
            "residual_dropout",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise GPT2ConfigError(f"{name} must be finite and in [0, 1)")
        if self.activation_function != "gelu_new":
            raise GPT2ConfigError("activation_function must be 'gelu_new'")

    @property
    def head_dim(self) -> int:
        """Return the width of one attention head."""

        return self.embedding_dim // self.head_count

    def fingerprint(self) -> str:
        """Return a canonical architecture fingerprint."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


GPT2_VARIANTS: Mapping[str, GPT2Config] = MappingProxyType(
    {
        "gpt2": GPT2Config(
            vocab_size=50_257,
            context_length=1_024,
            embedding_dim=768,
            head_count=12,
            layer_count=12,
        )
    }
)


_HF_FIELDS = {
    "vocab_size": "vocab_size",
    "n_positions": "context_length",
    "n_embd": "embedding_dim",
    "n_head": "head_count",
    "n_layer": "layer_count",
    "layer_norm_epsilon": "layer_norm_epsilon",
    "embd_pdrop": "embedding_dropout",
    "attn_pdrop": "attention_dropout",
    "resid_pdrop": "residual_dropout",
    "activation_function": "activation_function",
}


def config_from_hugging_face(payload: Mapping[str, Any]) -> GPT2Config:
    """Extract the architecture fields from a Hugging Face GPT-2 config."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    model_type = payload.get("model_type")
    if model_type != "gpt2":
        raise GPT2ConfigError("model_type must be 'gpt2'")
    missing = sorted(field for field in _HF_FIELDS if field not in payload)
    if missing:
        raise GPT2ConfigError(
            "Hugging Face config is missing required fields: " + ", ".join(missing)
        )
    values = {target: payload[source] for source, target in _HF_FIELDS.items()}
    return GPT2Config(**values)


def require_variant(config: GPT2Config, variant: str) -> None:
    """Require an exact match with a supported named GPT-2 variant."""

    if not isinstance(config, GPT2Config):
        raise TypeError("config must be GPT2Config")
    if not isinstance(variant, str):
        raise TypeError("variant must be a string")
    expected = GPT2_VARIANTS.get(variant)
    if expected is None:
        raise GPT2ConfigError(f"unsupported GPT-2 variant: {variant!r}")
    if config != expected:
        raise GPT2ConfigError(f"configuration does not exactly match {variant!r}")
