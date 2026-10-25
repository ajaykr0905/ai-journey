"""Standalone components for experiments beyond the GPT-2-compatible baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Normalize each feature vector by its root mean square, then scale it."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError("width must be a positive integer")
        if width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise ValueError("eps must be a finite positive number")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")
        self.width = width
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        if inputs.ndim < 1 or inputs.shape[-1] != self.width:
            raise ValueError("inputs last dimension must match width")

        # Low-precision squared values can overflow even when the normalized
        # output is representable. Accumulate the RMS in float32 in that case.
        accumulator = (
            inputs.float()
            if inputs.dtype in (torch.float16, torch.bfloat16)
            else inputs
        )
        inverse_rms = torch.rsqrt(
            accumulator.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = (accumulator * inverse_rms).to(dtype=inputs.dtype)
        return normalized * self.weight.to(dtype=inputs.dtype)
