"""Explicit AdamW controls and exhaustive tied-parameter-safe decay groups."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class DecoderOptimizerConfig:
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"optimizer {name} must be finite numeric data")
            object.__setattr__(self, name, float(value))
        if not 0 <= self.weight_decay <= 1:
            raise ValueError("optimizer weight_decay must be in [0, 1]")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must be in [0, 1)")
        if not 0 < self.epsilon <= 1:
            raise ValueError("optimizer epsilon must be in (0, 1]")


def decoder_optimizer_groups(model) -> dict:
    """Matrices decay; vectors do not; shared embedding/head weights occur once."""

    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named or len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("optimizer parameters must be nonempty unique tensors")
    return {
        "decay_names": tuple(name for name, parameter in named if parameter.ndim >= 2),
        "no_decay_names": tuple(
            name for name, parameter in named if parameter.ndim < 2
        ),
        "trainable_parameters": sum(parameter.numel() for _, parameter in named),
    }


def build_decoder_optimizer(
    model, *, learning_rate: float, config: DecoderOptimizerConfig
) -> torch.optim.AdamW:
    if not isinstance(config, DecoderOptimizerConfig):
        raise TypeError("optimizer config must be DecoderOptimizerConfig")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 0.1
    ):
        raise ValueError("optimizer learning_rate must be finite and in (0, 0.1]")
    groups = decoder_optimizer_groups(model)
    parameters = dict(model.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [parameters[name] for name in groups["decay_names"]],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameters[name] for name in groups["no_decay_names"]],
                "weight_decay": 0.0,
            },
        ],
        lr=float(learning_rate),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        foreach=False,
        fused=False,
    )
