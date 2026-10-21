"""Bounded sampling and inference for checkpoints produced by the CPU DDP lab."""

from dataclasses import dataclass
import math

import torch

from .ddp_contracts import integer


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int = 48

    def __post_init__(self):
        integer(self.max_new_tokens, "max_new_tokens", 1)
        integer(self.seed, "seed")
        if self.max_new_tokens > 1024:
            raise ValueError("max_new_tokens exceeds CPU lab budget")
        if self.top_k is not None:
            integer(self.top_k, "top_k", 1)
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature and top_p are outside supported ranges")


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Keep the minimal nucleus after a deterministic exact top-k cutoff."""
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(dim=1).all()
    ):
        raise ValueError("each logit row requires at least one finite value")
    values = logits.double().clone()
    if config.temperature > 0:
        values /= config.temperature
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ranked = values.gather(1, order)
    if config.top_k is not None:
        ranked[:, min(config.top_k, values.shape[-1]) :] = -torch.inf
    probabilities = ranked.softmax(-1)
    previous_mass = probabilities.cumsum(-1) - probabilities
    ranked[previous_mass >= config.top_p] = -torch.inf
    return torch.full_like(values, -torch.inf).scatter(1, order, ranked)
