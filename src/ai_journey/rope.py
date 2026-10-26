"""Rotary position embeddings for batched decoder attention heads."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def apply_rotary_position_embeddings(
    query: Tensor,
    key: Tensor,
    *,
    position_offset: int = 0,
    base: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Rotate adjacent feature pairs at their absolute token positions.

    Both inputs have shape ``(batch, heads, time, head_dim)``. Query and key
    may use different head counts, as in grouped-query attention. For cached
    decoding, pass the number of already-cached tokens as ``position_offset``
    and rotate only the newly projected query and key tensors.
    """

    for name, value in (("query", query), ("key", key)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")
        if value.ndim != 4 or any(size == 0 for size in value.shape):
            raise ValueError(
                f"{name} must have non-empty (batch, heads, time, head_dim) shape"
            )

    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query and key must share batch, time, and head_dim")
    if query.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even")
    if query.dtype != key.dtype or query.device != key.device:
        raise ValueError("query and key must share dtype and device")
    if not isinstance(position_offset, int) or isinstance(position_offset, bool):
        raise ValueError("position_offset must be a non-negative integer")
    if position_offset < 0 or position_offset + query.shape[2] > 2**53:
        raise ValueError("position_offset must be a representable non-negative integer")
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise ValueError("base must be a finite number greater than one")
    try:
        base_value = float(base)
    except OverflowError as exc:
        raise ValueError("base must be a finite number greater than one") from exc
    if not math.isfinite(base_value) or base_value <= 1:
        raise ValueError("base must be a finite number greater than one")

    head_dim = query.shape[-1]
    positions = (
        torch.arange(query.shape[2], dtype=torch.float64, device=query.device)
        + position_offset
    )
    pair_indices = torch.arange(
        0, head_dim, 2, dtype=torch.float64, device=query.device
    )
    inverse_frequencies = torch.pow(base_value, -pair_indices / head_dim)
    angles = torch.outer(positions, inverse_frequencies)
    compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    cosines = angles.cos().to(dtype=compute_dtype)[None, None, :, :]
    sines = angles.sin().to(dtype=compute_dtype)[None, None, :, :]

    def rotate(value: Tensor) -> Tensor:
        working = value.to(dtype=compute_dtype)
        even, odd = working[..., ::2], working[..., 1::2]
        rotated = torch.stack(
            (even * cosines - odd * sines, even * sines + odd * cosines),
            dim=-1,
        ).flatten(-2)
        return rotated.to(dtype=value.dtype)

    return rotate(query), rotate(key)
