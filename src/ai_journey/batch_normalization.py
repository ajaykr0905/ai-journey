"""Scratch BatchNorm primitives for last-dimension transformer activations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BatchNormalizationError(ValueError):
    """Raised when scratch BatchNorm inputs or controls are invalid."""


class ScratchBatchNorm(nn.Module):
    """Normalize the final feature dimension and maintain running statistics.

    Training statistics use the biased variance for normalization, matching
    ``torch.nn.BatchNorm1d``. The running variance receives the unbiased sample
    estimate so evaluation behavior also matches PyTorch.
    """

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")
        if num_features <= 0:
            raise BatchNormalizationError("num_features must be positive")
        for name, value in (("eps", eps), ("momentum", momentum)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value):
                raise BatchNormalizationError(f"{name} must be finite")
        if eps <= 0:
            raise BatchNormalizationError("eps must be positive")
        if not 0 < momentum <= 1:
            raise BatchNormalizationError("momentum must be in (0, 1]")
        if not isinstance(affine, bool):
            raise TypeError("affine must be a boolean")

        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def reset_running_stats(self) -> None:
        """Restore the running statistics to their construction state."""

        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if not inputs.is_floating_point():
            raise TypeError("inputs must have a floating-point dtype")
        if inputs.ndim < 2:
            raise BatchNormalizationError("inputs must have at least two dimensions")
        if inputs.shape[-1] != self.num_features:
            raise BatchNormalizationError(
                "input feature dimension does not match num_features"
            )
        sample_count = inputs.numel() // self.num_features
        if self.training and sample_count <= 1:
            raise BatchNormalizationError(
                "training requires more than one sample per feature"
            )

        reduce_dims = tuple(range(inputs.ndim - 1))
        if self.training:
            mean = inputs.mean(dim=reduce_dims)
            variance = inputs.var(dim=reduce_dims, unbiased=False)
            unbiased_variance = variance * sample_count / (sample_count - 1)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(unbiased_variance.detach(), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean
            variance = self.running_var

        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight + self.bias
        return normalized

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}"
        )
