"""Exact contracts for a small replicated float64 CPU/Gloo expert lab."""

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math

from .decoder_variant_optimizer import DecoderOptimizerConfig
from .moe_config import MoEConfig, moe_config_from_mapping


@dataclass(frozen=True)
class MoEDDPConfig:
    model: MoEConfig
    world_size: int = 2
    micro_batch_size: int = 1
    accumulation_steps: int = 2
    sequence_length: int = 4
    optimizer_steps: int = 4
    learning_rate: float = 0.002
    balance_weight: float = 0.01
    z_weight: float = 0.001
    max_gradient_norm: float = 1.0
    seed: int = 53
    optimizer: DecoderOptimizerConfig = field(default_factory=DecoderOptimizerConfig)
    capacity_scope: str = "rank_microbatch"

    def __post_init__(self):
        if (
            type(self.model) is not MoEConfig
            or type(self.optimizer) is not DecoderOptimizerConfig
        ):
            raise ValueError("replicated MoE requires exact model/optimizer contracts")
        for name, limit in (
            ("world_size", 2),
            ("micro_batch_size", 4),
            ("accumulation_steps", 8),
            ("sequence_length", 128),
            ("optimizer_steps", 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"MoE DDP {name} must be an integer in [1, {limit}]")
        if (
            self.sequence_length > self.model.gpt2.context_length
            or self.model.gpt2.context_length > 128
            or self.capacity_scope != "rank_microbatch"
        ):
            raise ValueError(
                "MoE DDP requires fitting sequences and rank/microbatch capacity"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("MoE DDP seed must be a bounded integer")
        for name, minimum, maximum in (
            ("learning_rate", 0.0, 0.1),
            ("balance_weight", -1.0, 1.0),
            ("z_weight", -1.0, 1.0),
            ("max_gradient_norm", 0.0, 1e6),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not minimum < value <= maximum
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"MoE DDP {name} is outside its finite range")
            object.__setattr__(self, name, float(value))
        from .moe_ddp_resources import require_moe_ddp_budget

        require_moe_ddp_budget(self)

    def to_mapping(self):
        return {
            "schema": "ai-journey-moe-ddp-v1",
            **asdict(self),
            "model": self.model.to_mapping(),
        }

    def fingerprint(self):
        return sha256(
            json.dumps(
                self.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()


def moe_ddp_config_from_mapping(value):
    expected = {item.name for item in fields(MoEDDPConfig)} | {"schema"}
    if (
        type(value) is not dict
        or set(value) != expected
        or value["schema"] != "ai-journey-moe-ddp-v1"
    ):
        raise ValueError("MoE DDP configuration fields differ")
    optimizer = value["optimizer"]
    if type(optimizer) is not dict or set(optimizer) != {
        item.name for item in fields(DecoderOptimizerConfig)
    }:
        raise ValueError("MoE DDP optimizer fields differ")
    if any(
        type(part) not in (int, float) or not 0 <= part <= 1
        for part in optimizer.values()
    ):
        raise ValueError("MoE DDP optimizer numerics must be bounded before conversion")
    return MoEDDPConfig(
        **{
            **{key: part for key, part in value.items() if key != "schema"},
            "model": moe_config_from_mapping(value["model"]),
            "optimizer": DecoderOptimizerConfig(**optimizer),
        }
    )
