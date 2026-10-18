"""Independent GPT-2 decoder implementation for weight interoperability."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def gelu_new(inputs: Tensor) -> Tensor:
    """Apply the tanh GELU approximation used by the published GPT-2 model."""

    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if not inputs.is_floating_point():
        raise TypeError("inputs must have a floating-point dtype")
    return (
        0.5
        * inputs
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * inputs.pow(3.0))
            )
        )
    )
