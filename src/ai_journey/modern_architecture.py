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

        # Accumulate low-precision inputs in float32, then scale each vector
        # before squaring: even a finite float32 value can overflow when
        # squared, while the normalized output remains representable.
        accumulator = (
            inputs.float()
            if inputs.dtype in (torch.float16, torch.bfloat16)
            else inputs
        )
        scale = accumulator.abs().amax(dim=-1, keepdim=True).clamp_min(1)
        scaled = accumulator / scale
        inverse_rms = torch.rsqrt(
            scaled.square().mean(dim=-1, keepdim=True) + self.eps / scale / scale
        )
        normalized = (scaled * inverse_rms).to(dtype=inputs.dtype)
        return normalized * self.weight.to(dtype=inputs.dtype)
