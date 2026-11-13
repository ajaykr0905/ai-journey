"""Global-memory coalescing model for warp requests.

The model follows the CUDA C++ Best Practices Guide v13.5, section 10.2.1
(Coalesced Access to Global Memory): on compute capability 6.0 and newer, the
concurrent accesses of a warp's active lanes coalesce into the number of
32-byte transactions needed to service them. Inactive lanes request nothing,
and a partially used sector is still transferred in full (section 10.2.1.1).

Natural alignment guarantees that every 1- to 16-byte access lies inside one
32-byte sector and one 128-byte line, so a lane touches exactly one of each.
The line count reports how many naturally aligned 128-byte segments a request
spans, the granularity the guide gives for L1-cached accesses on compute
capability 5.2. No L1 or L2 cache, hit, eviction, prefetch, or reuse across
requests is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_pattern import (
    LINE_BYTES,
    SECTOR_BYTES,
    MemoryAccessError,
    WarpAccess,
)


@dataclass(frozen=True)
class CoalescingAnalysis:
    """Sectors, lines, and byte accounting for one warp request.

    ``requested_bytes`` counts every active lane, including lanes that repeat
    an address; ``unique_bytes`` counts each distinct requested byte once.
    Efficiency is the fraction of transferred bytes that some lane uses.
    """

    access_bytes: int
    active_lanes: int
    sectors: tuple[int, ...]
    lines: tuple[int, ...]
    requested_bytes: int
    unique_bytes: int

    @property
    def sector_count(self) -> int:
        return len(self.sectors)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def transferred_bytes(self) -> int:
        return self.sector_count * SECTOR_BYTES

    @property
    def efficiency(self) -> float:
        return self.unique_bytes / self.transferred_bytes

    def to_mapping(self) -> dict[str, object]:
        return {
            "access_bytes": self.access_bytes,
            "active_lanes": self.active_lanes,
            "sector_indices": list(self.sectors),
            "line_indices": list(self.lines),
            "sectors": self.sector_count,
            "lines": self.line_count,
            "requested_bytes": self.requested_bytes,
            "unique_bytes": self.unique_bytes,
            "transferred_bytes": self.transferred_bytes,
            "efficiency": self.efficiency,
        }


def analyze_coalescing(access: WarpAccess) -> CoalescingAnalysis:
    """Count the 32-byte sectors and 128-byte lines one warp request touches."""

    if not isinstance(access, WarpAccess):
        raise TypeError("access must be a WarpAccess")
    addresses = access.active_addresses
    return CoalescingAnalysis(
        access_bytes=access.access_bytes,
        active_lanes=len(addresses),
        sectors=tuple(sorted({address // SECTOR_BYTES for address in addresses})),
        lines=tuple(sorted({address // LINE_BYTES for address in addresses})),
        requested_bytes=len(addresses) * access.access_bytes,
        unique_bytes=len(set(addresses)) * access.access_bytes,
    )


@dataclass(frozen=True)
class CoalescingSummary:
    """Totals over independent warp requests; no reuse between them."""

    requests: int
    sectors: int
    lines: int
    requested_bytes: int
    unique_bytes: int

    @property
    def transferred_bytes(self) -> int:
        return self.sectors * SECTOR_BYTES

    @property
    def sectors_per_request(self) -> float:
        return self.sectors / self.requests

    @property
    def efficiency(self) -> float:
        return self.unique_bytes / self.transferred_bytes

    def scaled(self, factor: int) -> CoalescingSummary:
        """Repeat the same request mix ``factor`` times."""

        if type(factor) is not int:
            raise TypeError("factor must be an integer")
        if factor <= 0:
            raise MemoryAccessError("factor must be positive")
        return CoalescingSummary(
            requests=self.requests * factor,
            sectors=self.sectors * factor,
            lines=self.lines * factor,
            requested_bytes=self.requested_bytes * factor,
            unique_bytes=self.unique_bytes * factor,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "sectors": self.sectors,
            "lines": self.lines,
            "requested_bytes": self.requested_bytes,
            "unique_bytes": self.unique_bytes,
            "transferred_bytes": self.transferred_bytes,
            "sectors_per_request": self.sectors_per_request,
            "efficiency": self.efficiency,
        }


def summarize_coalescing(
    analyses: tuple[CoalescingAnalysis, ...],
) -> CoalescingSummary:
    """Add per-request accounting without assuming any cache reuse."""

    if type(analyses) is not tuple or not analyses:
        raise MemoryAccessError("analyses must be a nonempty tuple")
    if any(not isinstance(item, CoalescingAnalysis) for item in analyses):
        raise TypeError("every analysis must be a CoalescingAnalysis")
    return CoalescingSummary(
        requests=len(analyses),
        sectors=sum(item.sector_count for item in analyses),
        lines=sum(item.line_count for item in analyses),
        requested_bytes=sum(item.requested_bytes for item in analyses),
        unique_bytes=sum(item.unique_bytes for item in analyses),
    )
