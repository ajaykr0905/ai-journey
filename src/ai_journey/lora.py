"""Small, explicit low-rank adapters for the bounded CPU decoder lab."""

from dataclasses import dataclass
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAError(ValueError):
    """An adapter, its target, or its saved state violates the contract."""


@dataclass(frozen=True)
class LoRAConfig:
    """Predeclared adapter capacity, scaling, dropout, and local RNG seed."""

    rank: int = 2
    alpha: float = 4.0
    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise LoRAError("rank must be an integer in [1, 256]")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or not 0 < self.alpha <= 1024
        ):
            raise LoRAError("alpha must be finite and in (0, 1024]")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise LoRAError("dropout must be finite and in [0, 1)")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise LoRAError("seed must be an integer in [0, 2**32)")

    @property
    def scaling(self):
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Frozen dense projection plus ``(alpha/rank) * B @ A``.

    B starts at zero, so installing an adapter preserves the base function.
    A uses a local generator: constructing adapters does not consume the
    caller's training RNG. The supplied dense layer is never modified.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if type(base) is not nn.Linear or not isinstance(config, LoRAConfig):
            raise TypeError("expected a plain nn.Linear and LoRAConfig")
        if base.weight.device.type != "cpu" or not base.weight.is_floating_point():
            raise LoRAError("this adapter lab requires floating-point CPU weights")
        if config.rank > min(base.in_features, base.out_features):
            raise LoRAError("adapter rank exceeds the projection dimensions")
        if any(not bool(torch.isfinite(p).all()) for p in base.parameters()):
            raise LoRAError("base projection parameters must be finite")
        self.config = config
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        self.in_features = base.in_features
        self.out_features = base.out_features
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        a = torch.empty(config.rank, base.in_features, dtype=base.weight.dtype)
        a.normal_(mean=0, std=1 / math.sqrt(base.in_features), generator=generator)
        self.lora_a = nn.Parameter(a)
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, config.rank, dtype=base.weight.dtype)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.train(base.training)

    @property
    def weight(self):
        """Expose the base dtype/device contract used by decoder validation."""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def delta_weight(self):
        return (self.lora_b @ self.lora_a) * self.config.scaling

    def forward(self, inputs):
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.config.scaling
