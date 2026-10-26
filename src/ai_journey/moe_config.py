"""Strict, allocation-free contracts for a CPU top-two expert decoder."""

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
from collections.abc import Mapping

from .decoder_variant_config import DecoderVariantConfig
from .decoder_variant_resources import expected_decoder_variant_parameter_count
from .gpt2_config import GPT2Config


@dataclass(frozen=True)
class MoEConfig(DecoderVariantConfig):
    """Replace selected FFNs with experts; evaluation never drops routes.

    Experts use the inherited GELU (hidden=4C, biased) or SwiGLU (three
    bias-free matrices) contract. Training capacity is token-major, first-come
    per expert; retained route weights are not renormalized after dropping.
    """

    expert_count: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    min_capacity: int = 1
    moe_layers: tuple[int, ...] = (0,)

    def __post_init__(self):
        super().__post_init__()
        if type(self.expert_count) is not int or not 2 <= self.expert_count <= 64:
            raise ValueError("expert_count must be an integer in [2, 64]")
        if type(self.top_k) is not int or self.top_k != 2:
            raise ValueError("the reference MoE contract requires top_k=2")
        if (
            type(self.capacity_factor) not in (int, float)
            or not math.isfinite(self.capacity_factor)
            or not 0 < self.capacity_factor <= 16
        ):
            raise ValueError("capacity_factor must be finite and in (0, 16]")
        object.__setattr__(self, "capacity_factor", float(self.capacity_factor))
        if type(self.min_capacity) is not int or not 1 <= self.min_capacity <= 1024:
            raise ValueError("min_capacity must be an integer in [1, 1024]")
        if (
            not isinstance(self.moe_layers, (tuple, list))
            or not self.moe_layers
            or any(type(index) is not int for index in self.moe_layers)
            or tuple(sorted(set(self.moe_layers))) != tuple(self.moe_layers)
            or not 0
            <= self.moe_layers[0]
            <= self.moe_layers[-1]
            < self.gpt2.layer_count
        ):
            raise ValueError("moe_layers must be sorted unique in-range layer indices")
        object.__setattr__(self, "moe_layers", tuple(self.moe_layers))

    def to_mapping(self) -> dict:
        """Return JSON-safe architecture fields, without training objective weights."""
        payload = asdict(self)
        payload["moe_layers"] = list(self.moe_layers)
        return payload

    def fingerprint(self) -> str:
        payload = self.to_mapping()
        payload["swiglu_hidden_dim"] = self.resolved_swiglu_hidden_dim
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(b"moe-decoder-v1\0" + encoded.encode()).hexdigest()


def moe_config_from_mapping(payload: Mapping) -> MoEConfig:
    """Reject unknown/missing fields rather than silently changing an experiment."""
    if not isinstance(payload, Mapping) or set(payload) != {
        f.name for f in fields(MoEConfig)
    }:
        raise ValueError("MoE configuration fields do not match")
    base = payload["gpt2"]
    if not isinstance(base, Mapping) or set(base) != {
        f.name for f in fields(GPT2Config)
    }:
        raise ValueError("MoE GPT-2 configuration fields do not match")
    return MoEConfig(**{**payload, "gpt2": GPT2Config(**base)})


def expert_parameter_count(config: MoEConfig) -> int:
    """Count one expert, including GELU biases but no nonexistent SwiGLU biases."""
    if not isinstance(config, MoEConfig):
        raise TypeError("config must be MoEConfig")
    width = config.gpt2.embedding_dim
    if config.mlp_kind == "swiglu":
        return 3 * width * config.resolved_swiglu_hidden_dim
    return 8 * width**2 + 5 * width


def expected_moe_parameter_count(config: MoEConfig) -> int:
    """Count stored unique parameters, excluding the tied output-head alias."""
    expert = expert_parameter_count(config)
    router = config.gpt2.embedding_dim * config.expert_count
    return expected_decoder_variant_parameter_count(config) + len(config.moe_layers) * (
        (config.expert_count - 1) * expert + router
    )


def expected_moe_active_parameters(config: MoEConfig) -> int:
    """Count a per-token upper bound, not stored weights or optimizer memory."""
    return expected_moe_parameter_count(config) - len(config.moe_layers) * (
        config.expert_count - config.top_k
    ) * expert_parameter_count(config)
