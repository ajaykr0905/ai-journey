"""Declared CUDA memory geometry and warp byte-address pattern generators.

Every generator returns the byte offsets that the 32 lanes of one warp-wide
load or store would request inside one declared allocation. The module only
enumerates addresses; it does not execute a GPU kernel, model a cache, or claim
any device timing.

The geometry is a declared contract taken from public NVIDIA documentation:

- CUDA C++ Programming Guide v12.6, section 4.1 (SIMT Architecture): warps of
  32 threads holding consecutive, increasing thread IDs.
- Programming Guide v12.6, section 5.3.2 (Device Memory Accesses): naturally
  aligned 32-, 64- and 128-byte memory transactions, and 1-, 2-, 4-, 8- and
  16-byte words that compile to one instruction only when naturally aligned.
- CUDA C++ Best Practices Guide v13.5, sections 10.2.1 and 10.2.3.1: 32-byte
  transactions on compute capability 6.0 and newer, 128-byte aligned segments
  for L1-cached accesses, and 32 shared-memory banks of 32-bit words on
  compute capability 5.x and newer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256


class MemoryAccessError(ValueError):
    """Raised when a memory-access contract, pattern, or report is invalid."""


WARP_SIZE = 32
SECTOR_BYTES = 32
LINE_BYTES = 128
SHARED_MEMORY_BANKS = 32
BANK_WIDTH_BYTES = 4
ACCESS_SIZES = (1, 2, 4, 8, 16)
FULL_WARP_MASK = (1 << WARP_SIZE) - 1
MAX_ADDRESS_BYTES = 1 << 40
MAX_STRIDE_ELEMENTS = 1 << 24
MAX_MATRIX_EXTENT = 1 << 20
MAX_PERMUTATION_SEED = (1 << 63) - 1

_DOCUMENTED_GEOMETRY = {
    "warp_size": WARP_SIZE,
    "sector_bytes": SECTOR_BYTES,
    "line_bytes": LINE_BYTES,
    "shared_memory_banks": SHARED_MEMORY_BANKS,
    "bank_width_bytes": BANK_WIDTH_BYTES,
}


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _bounded(value: object, name: str, minimum: int, maximum: int) -> int:
    number = _integer(value, name)
    if not minimum <= number <= maximum:
        raise MemoryAccessError(f"{name} must be in [{minimum}, {maximum}]")
    return number


def require_access_bytes(access_bytes: object) -> int:
    """Validate one supported per-lane access width in bytes."""

    width = _integer(access_bytes, "access_bytes")
    if width not in ACCESS_SIZES:
        raise MemoryAccessError("access_bytes must be 1, 2, 4, 8 or 16")
    return width


def require_aligned_address(address: object, access_bytes: object) -> int:
    """Validate a naturally aligned byte address for one access width.

    Natural alignment means the address is a multiple of the access size, the
    condition under which the Programming Guide compiles a global access into
    a single instruction.
    """

    width = require_access_bytes(access_bytes)
    offset = _integer(address, "address")
    if not 0 <= offset <= MAX_ADDRESS_BYTES - width:
        raise MemoryAccessError("address is outside the declared byte range")
    if offset % width:
        raise MemoryAccessError("address must be naturally aligned to access_bytes")
    return offset


def require_lane_mask(mask: object) -> int:
    """Validate a 32-bit active-lane mask that selects at least one lane."""

    return _bounded(mask, "active_mask", 1, FULL_WARP_MASK)


def lane_mask(lanes: Iterable[int]) -> int:
    """Build an active-lane mask in which bit ``i`` selects lane ``i``."""

    if isinstance(lanes, (str, bytes)):
        raise TypeError("lanes must be an iterable of lane indices")
    mask = 0
    for lane in lanes:
        index = _bounded(lane, "lane", 0, WARP_SIZE - 1)
        if mask >> index & 1:
            raise MemoryAccessError("lane indices must be unique")
        mask |= 1 << index
    return require_lane_mask(mask)


@dataclass(frozen=True)
class MemoryGeometry:
    """The declared CUDA geometry shared by every model in this lab."""

    warp_size: int = WARP_SIZE
    sector_bytes: int = SECTOR_BYTES
    line_bytes: int = LINE_BYTES
    shared_memory_banks: int = SHARED_MEMORY_BANKS
    bank_width_bytes: int = BANK_WIDTH_BYTES

    def __post_init__(self) -> None:
        for name in _DOCUMENTED_GEOMETRY:
            _integer(getattr(self, name), name)
        if self.to_mapping() != _DOCUMENTED_GEOMETRY:
            raise MemoryAccessError(
                "geometry must match the documented CUDA contract: 32-lane warps, "
                "32-byte sectors, 128-byte lines, 32 banks of 4 bytes"
            )

    def to_mapping(self) -> dict[str, int]:
        """Return the canonical geometry mapping recorded in reports."""

        return asdict(self)


CUDA_GEOMETRY = MemoryGeometry()


@dataclass(frozen=True)
class WarpAccess:
    """Byte addresses requested by the lanes of one warp-wide instruction.

    ``lane_addresses`` holds one entry per lane. ``None`` marks an inactive
    lane, which issues no request and is excluded from every model.
    """

    access_bytes: int
    lane_addresses: tuple[int | None, ...]

    def __post_init__(self) -> None:
        require_access_bytes(self.access_bytes)
        if (
            type(self.lane_addresses) is not tuple
            or len(self.lane_addresses) != WARP_SIZE
        ):
            raise MemoryAccessError("lane_addresses must hold one entry per lane")
        active = 0
        for address in self.lane_addresses:
            if address is not None:
                require_aligned_address(address, self.access_bytes)
                active += 1
        if not active:
            raise MemoryAccessError("a warp request needs at least one active lane")

    @property
    def active_mask(self) -> int:
        return sum(
            1 << lane
            for lane, address in enumerate(self.lane_addresses)
            if address is not None
        )

    @property
    def active_lanes(self) -> tuple[int, ...]:
        return tuple(
            lane
            for lane, address in enumerate(self.lane_addresses)
            if address is not None
        )

    @property
    def active_addresses(self) -> tuple[int, ...]:
        return tuple(address for address in self.lane_addresses if address is not None)

    def masked(self, mask: int) -> WarpAccess:
        """Return a copy in which every lane outside ``mask`` is inactive."""

        selected = require_lane_mask(mask)
        return WarpAccess(
            self.access_bytes,
            tuple(
                address if selected >> lane & 1 else None
                for lane, address in enumerate(self.lane_addresses)
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "access_bytes": self.access_bytes,
            "active_mask": self.active_mask,
            "lane_addresses": list(self.lane_addresses),
        }


def strided_access(
    *, stride_elements: int, access_bytes: int = 4, base_bytes: int = 0
) -> WarpAccess:
    """Lane ``i`` requests element ``i * stride_elements`` after ``base_bytes``."""

    stride = _bounded(stride_elements, "stride_elements", 1, MAX_STRIDE_ELEMENTS)
    width = require_access_bytes(access_bytes)
    base = require_aligned_address(base_bytes, width)
    return WarpAccess(
        width, tuple(base + lane * stride * width for lane in range(WARP_SIZE))
    )


def contiguous_access(*, access_bytes: int = 4, base_bytes: int = 0) -> WarpAccess:
    """Lane ``i`` requests the ``i``-th element after ``base_bytes``.

    A ``base_bytes`` that is not a multiple of the 32-byte sector produces the
    misaligned sequential pattern of Best Practices Guide section 10.2.1.2.
    """

    return strided_access(
        stride_elements=1, access_bytes=access_bytes, base_bytes=base_bytes
    )


def broadcast_access(*, address_bytes: int = 0, access_bytes: int = 4) -> WarpAccess:
    """Every lane requests the same element."""

    address = require_aligned_address(address_bytes, access_bytes)
    return WarpAccess(access_bytes, (address,) * WARP_SIZE)


def _uniform_below(seed: int, counter: int, bound: int) -> int:
    limit = (1 << 64) - (1 << 64) % bound
    attempt = 0
    while True:
        block = sha256(
            f"ai-journey-memory-access-permutation:{seed}:{counter}:{attempt}".encode()
        ).digest()
        value = int.from_bytes(block[:8], "big")
        if value < limit:
            return value % bound
        attempt += 1


def seeded_permutation(seed: int, count: int = WARP_SIZE) -> tuple[int, ...]:
    """Return a Fisher-Yates permutation driven by SHA-256 counter blocks.

    The derivation is pinned in this module instead of relying on the
    ``random`` module, so one seed yields the same order on every Python
    version and platform.
    """

    key = _bounded(seed, "seed", 0, MAX_PERMUTATION_SEED)
    size = _bounded(count, "count", 1, 1 << 16)
    order = list(range(size))
    for index in range(size - 1, 0, -1):
        other = _uniform_below(key, index, index + 1)
        order[index], order[other] = order[other], order[index]
    return tuple(order)


def permuted_access(
    *, seed: int, access_bytes: int = 4, base_bytes: int = 0
) -> WarpAccess:
    """Lane ``i`` requests element ``permutation[i]`` of one contiguous span."""

    width = require_access_bytes(access_bytes)
    base = require_aligned_address(base_bytes, width)
    return WarpAccess(
        width, tuple(base + element * width for element in seeded_permutation(seed))
    )


@dataclass(frozen=True)
class MatrixLayout:
    """A row-major 2D array whose rows start ``leading_dimension`` elements apart.

    ``leading_dimension - columns`` elements of padding follow every row, as in
    a pitched allocation.
    """

    rows: int
    columns: int
    leading_dimension: int
    element_bytes: int = 4
    base_bytes: int = 0

    def __post_init__(self) -> None:
        _bounded(self.rows, "rows", 1, MAX_MATRIX_EXTENT)
        _bounded(self.columns, "columns", 1, MAX_MATRIX_EXTENT)
        _bounded(self.leading_dimension, "leading_dimension", 1, MAX_MATRIX_EXTENT)
        if self.leading_dimension < self.columns:
            raise MemoryAccessError("leading_dimension must be at least columns")
        require_aligned_address(self.base_bytes, self.element_bytes)
        if self.base_bytes + self.allocation_bytes > MAX_ADDRESS_BYTES:
            raise MemoryAccessError("matrix allocation exceeds the declared byte range")

    @property
    def padding_elements(self) -> int:
        return self.leading_dimension - self.columns

    @property
    def allocation_bytes(self) -> int:
        return self.rows * self.leading_dimension * self.element_bytes

    def address(self, row: int, column: int) -> int:
        """Return the byte address of one in-bounds element."""

        r = _bounded(row, "row", 0, self.rows - 1)
        c = _bounded(column, "column", 0, self.columns - 1)
        return self.base_bytes + (r * self.leading_dimension + c) * self.element_bytes


def row_access(layout: MatrixLayout, *, row: int, start_column: int = 0) -> WarpAccess:
    """Lane ``i`` reads ``[row][start_column + i]``.

    Lanes that would fall past the last column are inactive, as they are in a
    bounds-checked kernel.
    """

    if not isinstance(layout, MatrixLayout):
        raise TypeError("layout must be a MatrixLayout")
    start = _bounded(start_column, "start_column", 0, layout.columns - 1)
    return WarpAccess(
        layout.element_bytes,
        tuple(
            layout.address(row, start + lane) if start + lane < layout.columns else None
            for lane in range(WARP_SIZE)
        ),
    )


def column_access(
    layout: MatrixLayout, *, column: int, start_row: int = 0
) -> WarpAccess:
    """Lane ``i`` reads ``[start_row + i][column]``.

    Consecutive lanes are ``leading_dimension`` elements apart. Lanes that
    would fall past the last row are inactive.
    """

    if not isinstance(layout, MatrixLayout):
        raise TypeError("layout must be a MatrixLayout")
    start = _bounded(start_row, "start_row", 0, layout.rows - 1)
    return WarpAccess(
        layout.element_bytes,
        tuple(
            layout.address(start + lane, column) if start + lane < layout.rows else None
            for lane in range(WARP_SIZE)
        ),
    )
