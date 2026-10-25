"""Validated opt-in decoder choices beside the exact GPT-2 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Literal

from .gpt2_config import GPT2Config


class DecoderVariantConfigError(ValueError):
    """Raised when an opt-in decoder architecture contract is invalid."""


@dataclass(frozen=True)
class DecoderVariantConfig:
    """Wrap a GPT-2 contract without changing its baseline model or artifacts."""

    gpt2: GPT2Config
    norm_kind: Literal["layernorm", "rmsnorm"] = "layernorm"
    mlp_kind: Literal["gelu", "swiglu"] = "gelu"
    position_kind: Literal["learned", "rope"] = "learned"
    rope_base: float = 10000.0
    swiglu_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gpt2, GPT2Config):
            raise TypeError("gpt2 must be GPT2Config")
        if self.norm_kind not in ("layernorm", "rmsnorm"):
            raise DecoderVariantConfigError("norm_kind must be layernorm or rmsnorm")
        if self.mlp_kind not in ("gelu", "swiglu"):
            raise DecoderVariantConfigError("mlp_kind must be gelu or swiglu")
        if self.position_kind not in ("learned", "rope"):
            raise DecoderVariantConfigError("position_kind must be learned or rope")
        if isinstance(self.rope_base, bool) or not isinstance(
            self.rope_base, (int, float)
        ):
            raise DecoderVariantConfigError(
                "rope_base must be finite and greater than 1"
            )
        try:
            rope_base = float(self.rope_base)
        except OverflowError as exc:
            raise DecoderVariantConfigError(
                "rope_base must be finite and greater than 1"
            ) from exc
        if not math.isfinite(rope_base) or rope_base <= 1:
            raise DecoderVariantConfigError(
                "rope_base must be finite and greater than 1"
            )
        object.__setattr__(self, "rope_base", rope_base)
        if self.swiglu_hidden_dim is not None:
            if (
                self.mlp_kind != "swiglu"
                or isinstance(self.swiglu_hidden_dim, bool)
                or not isinstance(self.swiglu_hidden_dim, int)
                or self.swiglu_hidden_dim <= 0
            ):
                raise DecoderVariantConfigError(
                    "swiglu_hidden_dim must be a positive integer only for swiglu"
                )
        if self.position_kind == "rope" and self.gpt2.head_dim % 2:
            raise DecoderVariantConfigError("rope requires an even head_dim")

    @property
    def resolved_swiglu_hidden_dim(self) -> int | None:
        """Use H≈8C/3 so three SwiGLU matrices match GELU's 8C² budget."""

        if self.mlp_kind != "swiglu":
            return None
        if self.swiglu_hidden_dim is not None:
            return self.swiglu_hidden_dim
        return (8 * self.gpt2.embedding_dim + 1) // 3

    def fingerprint(self) -> str:
        """Hash the canonical, resolved decoder architecture contract."""

        payload = {
            "gpt2": asdict(self.gpt2),
            "norm_kind": self.norm_kind,
            "mlp_kind": self.mlp_kind,
            "position_kind": self.position_kind,
            "rope_base": self.rope_base,
            "swiglu_hidden_dim": self.resolved_swiglu_hidden_dim,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()
