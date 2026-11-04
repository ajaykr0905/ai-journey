"""Deterministic paged-KV allocation and internal-fragmentation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .inference_kv import InferenceCapacityError, _positive_integer


@dataclass(frozen=True)
class PagedAllocation:
    requested_tokens: int
    blocks: int
    block_tokens: int
    allocated_tokens: int
    internal_waste_tokens: int
    allocated_bytes: int

    @property
    def internal_waste_fraction(self) -> Fraction:
        return Fraction(self.internal_waste_tokens, self.allocated_tokens)


@dataclass(frozen=True)
class PagedAdmission:
    allocation: PagedAllocation
    capacity_bytes: int
    admitted: bool


@dataclass(frozen=True)
class SharedPrefixAllocation:
    logical_tokens: int
    physical_tokens: int
    saved_tokens: int
    physical_bytes: int


def plan_shared_prefix(
    sequence_lengths: tuple[int, ...],
    *,
    shared_prefix_tokens: int,
    bytes_per_token: int,
) -> SharedPrefixAllocation:
    """Charge one immutable prefix plus each request's private suffix."""

    if type(sequence_lengths) is not tuple or not sequence_lengths:
        raise InferenceCapacityError("sequence_lengths must be a nonempty tuple")
    _positive_integer(shared_prefix_tokens, "shared_prefix_tokens")
    _positive_integer(bytes_per_token, "bytes_per_token")
    for length in sequence_lengths:
        _positive_integer(length, "sequence length")
        if length < shared_prefix_tokens:
            raise InferenceCapacityError(
                "shared prefix cannot exceed any sequence length"
            )
    logical = sum(sequence_lengths)
    physical = shared_prefix_tokens + sum(
        length - shared_prefix_tokens for length in sequence_lengths
    )
    return SharedPrefixAllocation(
        logical, physical, logical - physical, physical * bytes_per_token
    )


def admit_paged_allocation(
    allocation: PagedAllocation, *, capacity_bytes: int
) -> PagedAdmission:
    """Admit an allocation only when whole pages fit the byte budget."""

    if not isinstance(allocation, PagedAllocation):
        raise TypeError("allocation must be PagedAllocation")
    _positive_integer(capacity_bytes, "capacity_bytes")
    return PagedAdmission(
        allocation, capacity_bytes, allocation.allocated_bytes <= capacity_bytes
    )


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
