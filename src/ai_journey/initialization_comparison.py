"""Controlled fixed-normal versus Kaiming initialization experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ai_journey.transformer_lab import (
    DecoderLanguageModel,
    expected_initialization_std,
)


@dataclass(frozen=True)
class ParameterInitializationAudit:
    """Observed and theoretical statistics for one initialized weight matrix."""

    name: str
    module_type: str
    shape: tuple[int, ...]
    fan_in: int | None
    expected_std: float
    observed_mean: float
    observed_std: float
    relative_std_error: float
    bias_is_zero: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_model_initialization(
    model: DecoderLanguageModel,
) -> tuple[ParameterInitializationAudit, ...]:
    """Inspect every explicitly initialized matrix in stable module order."""

    if not isinstance(model, DecoderLanguageModel):
        raise TypeError("model must be a DecoderLanguageModel")
    audits: list[ParameterInitializationAudit] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            weight = module.weight.detach()
            expected = expected_initialization_std(module, model.config)
            observed = float(weight.std(unbiased=False))
            bias_is_zero = None
            fan_in = None
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                bias_is_zero = module.bias is None or bool(
                    torch.count_nonzero(module.bias.detach()) == 0
                )
            audits.append(
                ParameterInitializationAudit(
                    name=f"{name}.weight",
                    module_type=type(module).__name__,
                    shape=tuple(weight.shape),
                    fan_in=fan_in,
                    expected_std=expected,
                    observed_mean=float(weight.mean()),
                    observed_std=observed,
                    relative_std_error=abs(observed - expected) / expected,
                    bias_is_zero=bias_is_zero,
                )
            )
    return tuple(audits)
