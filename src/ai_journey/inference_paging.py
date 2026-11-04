"""Deterministic paged-KV allocation and internal-fragmentation accounting."""

from __future__ import annotations

from dataclasses import dataclass

from .inference_kv import InferenceCapacityError, _positive_integer


@dataclass(frozen=True)
class PagedAllocation:
    requested_tokens: int
    blocks: int
    block_tokens: int
    allocated_tokens: int
    internal_waste_tokens: int
    allocated_bytes: int


def plan_paged_allocation(
    sequence_lengths: tuple[int, ...], *, block_tokens: int, bytes_per_token: int
) -> PagedAllocation:
    """Allocate whole pages independently for each live sequence."""

    if type(sequence_lengths) is not tuple or not sequence_lengths:
        raise InferenceCapacityError("sequence_lengths must be a nonempty tuple")
    _positive_integer(block_tokens, "block_tokens")
    _positive_integer(bytes_per_token, "bytes_per_token")
    for length in sequence_lengths:
        _positive_integer(length, "sequence length")
    blocks = sum(
        (length + block_tokens - 1) // block_tokens for length in sequence_lengths
    )
    requested = sum(sequence_lengths)
    allocated = blocks * block_tokens
    return PagedAllocation(
        requested,
        blocks,
        block_tokens,
        allocated,
        allocated - requested,
        allocated * bytes_per_token,
    )
