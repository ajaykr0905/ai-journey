"""Shared-memory bank model for compute capability 5.x and newer.

Sources: CUDA C++ Programming Guide v12.6, section 16.4.3 (compute capability
5.x shared memory; sections 16.5.3 and 16.6.4 say 6.x and 7.x behave the same
way) and section 5.3.2, and the CUDA C++ Best Practices Guide v13.5, section
10.2.3.1:

- there are 32 banks and successive 32-bit words map to successive banks;
- lanes that access any address within the same 32-bit word do not conflict:
  a read is served by broadcast (several broadcasts coalesce into one
  multicast) and for a write one undefined lane's value is stored;
- a request whose lanes need different words of one bank is split into as
  many conflict-free requests as necessary, an n-way bank conflict.

The conflict degree is therefore the maximum, over banks, of the number of
distinct 32-bit words requested from that bank, and it equals the number of
serialized wavefronts. Loads and stores are counted the same way. Addresses are
byte offsets from a shared allocation that starts at bank 0.

8- and 16-byte accesses are rejected: neither guide states how such accesses
are split into bank requests on these devices, so no rule is guessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .memory_access_pattern import (
    BANK_WIDTH_BYTES,
    SHARED_MEMORY_BANKS,
    MemoryAccessError,
    WarpAccess,
)

MODELED_BANK_ACCESS_BYTES = (1, 2, 4)
EXCLUDED_BANK_ACCESS_BYTES = (8, 16)
BANK_EXCLUSION = (
    "8- and 16-byte shared-memory accesses are excluded: the CUDA guides do not "
    "document how they split into bank requests on compute capability 5.x+"
)


@dataclass(frozen=True)
class BankAnalysis:
    """Bank occupancy and serialization for one warp's shared-memory request."""

    access_bytes: int
    active_lanes: int
    bank_words: tuple[tuple[int, ...], ...]
    multicast_words: int

    @property
    def distinct_words(self) -> int:
        return sum(len(words) for words in self.bank_words)

    @property
    def banks_used(self) -> int:
        return sum(1 for words in self.bank_words if words)

    @property
    def conflict_degree(self) -> int:
        return max(len(words) for words in self.bank_words)

    @property
    def wavefronts(self) -> int:
        return self.conflict_degree

    @property
    def conflict_free(self) -> bool:
        return self.conflict_degree == 1

    def to_mapping(self) -> dict[str, object]:
        return {
            "access_bytes": self.access_bytes,
            "active_lanes": self.active_lanes,
            "distinct_words": self.distinct_words,
            "multicast_words": self.multicast_words,
            "banks_used": self.banks_used,
            "conflict_degree": self.conflict_degree,
            "wavefronts": self.wavefronts,
        }


def analyze_banks(access: WarpAccess) -> BankAnalysis:
    """Map one request to banks and count its serialized wavefronts."""

    if not isinstance(access, WarpAccess):
        raise TypeError("access must be a WarpAccess")
    if access.access_bytes in EXCLUDED_BANK_ACCESS_BYTES:
        raise MemoryAccessError(BANK_EXCLUSION)
    lanes_per_word = Counter(
        address // BANK_WIDTH_BYTES for address in access.active_addresses
    )
    banks: list[list[int]] = [[] for _ in range(SHARED_MEMORY_BANKS)]
    for word in sorted(lanes_per_word):
        banks[word % SHARED_MEMORY_BANKS].append(word)
    return BankAnalysis(
        access_bytes=access.access_bytes,
        active_lanes=len(access.active_addresses),
        bank_words=tuple(tuple(words) for words in banks),
        multicast_words=sum(1 for lanes in lanes_per_word.values() if lanes > 1),
    )


@dataclass(frozen=True)
class BankSummary:
    """Wavefront totals over a sequence of shared-memory requests."""

    requests: int
    wavefronts: int
    max_conflict_degree: int

    @property
    def excess_wavefronts(self) -> int:
        return self.wavefronts - self.requests

    @property
    def wavefronts_per_request(self) -> float:
        return self.wavefronts / self.requests

    def to_mapping(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "wavefronts": self.wavefronts,
            "excess_wavefronts": self.excess_wavefronts,
            "max_conflict_degree": self.max_conflict_degree,
            "wavefronts_per_request": self.wavefronts_per_request,
        }


def summarize_banks(analyses: tuple[BankAnalysis, ...]) -> BankSummary:
    """Add wavefronts over requests that execute one after another."""

    if type(analyses) is not tuple or not analyses:
        raise MemoryAccessError("analyses must be a nonempty tuple")
    if any(not isinstance(item, BankAnalysis) for item in analyses):
        raise TypeError("every analysis must be a BankAnalysis")
    return BankSummary(
        requests=len(analyses),
        wavefronts=sum(item.wavefronts for item in analyses),
        max_conflict_degree=max(item.conflict_degree for item in analyses),
    )
