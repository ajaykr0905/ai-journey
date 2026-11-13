"""Bank and coalescing analysis of a shared-memory tile transpose.

The tiled transpose stages a 32 x 32 tile in shared memory. In stage one each
warp writes one tile row; in stage two each warp reads one tile column, so the
kernel can read input rows and write output rows. Sources: CUDA C++ Best
Practices Guide v13.5, section 10.2.3.3 (a column access to a w x w float
tile puts every lane in the same bank, and ``[TILE_DIM][TILE_DIM + 1]``
padding makes the lane stride w + 1 banks, equivalent to a unit stride) and
CUDA C++ Programming Guide v12.6, section 5.3.2 (two-dimensional arrays).

The global comparison covers an N x N row-major matrix whose size is a
multiple of 128 elements. Every warp request then has the same alignment, so
one representative request per access kind, repeated N * N / 32 times, gives
exact totals: a naive kernel reads input rows and writes output columns,
while the tiled kernel reads and writes rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_banks import BankSummary, analyze_banks, summarize_banks
from .memory_access_coalescing import (
    CoalescingSummary,
    analyze_coalescing,
    summarize_coalescing,
)
from .memory_access_pattern import (
    WARP_SIZE,
    MatrixLayout,
    MemoryAccessError,
    column_access,
    row_access,
)

TRANSPOSE_TILE = WARP_SIZE
TRANSPOSE_MATRIX_MULTIPLE = 128
MAX_TRANSPOSE_MATRIX = 1 << 16
MAX_TILE_LEADING_DIMENSION = 4096


@dataclass(frozen=True)
class TransposeAnalysis:
    """Shared-tile wavefronts per tile and global traffic per matrix."""

    leading_dimension: int
    element_bytes: int
    matrix_size: int
    shared_row_writes: BankSummary
    shared_column_reads: BankSummary
    naive_loads: CoalescingSummary
    naive_stores: CoalescingSummary
    tiled_loads: CoalescingSummary
    tiled_stores: CoalescingSummary

    @property
    def tile(self) -> int:
        return TRANSPOSE_TILE

    @property
    def tile_bytes(self) -> int:
        return TRANSPOSE_TILE * self.leading_dimension * self.element_bytes

    @property
    def padding_bytes(self) -> int:
        return (
            TRANSPOSE_TILE
            * (self.leading_dimension - TRANSPOSE_TILE)
            * self.element_bytes
        )

    @property
    def tiles(self) -> int:
        return (self.matrix_size // TRANSPOSE_TILE) ** 2

    def to_mapping(self) -> dict[str, object]:
        return {
            "tile": self.tile,
            "leading_dimension": self.leading_dimension,
            "element_bytes": self.element_bytes,
            "matrix_size": self.matrix_size,
            "tile_bytes": self.tile_bytes,
            "padding_bytes": self.padding_bytes,
            "tiles": self.tiles,
            "shared_row_writes_per_tile": self.shared_row_writes.to_mapping(),
            "shared_column_reads_per_tile": self.shared_column_reads.to_mapping(),
            "naive_global_loads": self.naive_loads.to_mapping(),
            "naive_global_stores": self.naive_stores.to_mapping(),
            "tiled_global_loads": self.tiled_loads.to_mapping(),
            "tiled_global_stores": self.tiled_stores.to_mapping(),
        }


def analyze_tile_transpose(
    *, leading_dimension: int = TRANSPOSE_TILE, element_bytes: int = 4, matrix_size: int
) -> TransposeAnalysis:
    """Compare an unpadded or padded 32 x 32 shared tile and both global kernels."""

    for name, value in (
        ("leading_dimension", leading_dimension),
        ("element_bytes", element_bytes),
        ("matrix_size", matrix_size),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if not TRANSPOSE_TILE <= leading_dimension <= MAX_TILE_LEADING_DIMENSION:
        raise MemoryAccessError(
            f"leading_dimension must be in [32, {MAX_TILE_LEADING_DIMENSION}]"
        )
    if (
        not 0 < matrix_size <= MAX_TRANSPOSE_MATRIX
        or matrix_size % TRANSPOSE_MATRIX_MULTIPLE
    ):
        raise MemoryAccessError(
            "matrix_size must be a positive multiple of 128 up to 65536"
        )
    tile = MatrixLayout(
        rows=TRANSPOSE_TILE,
        columns=TRANSPOSE_TILE,
        leading_dimension=leading_dimension,
        element_bytes=element_bytes,
    )
    row_writes = summarize_banks(
        tuple(analyze_banks(row_access(tile, row=row)) for row in range(TRANSPOSE_TILE))
    )
    column_reads = summarize_banks(
        tuple(
            analyze_banks(column_access(tile, column=column))
            for column in range(TRANSPOSE_TILE)
        )
    )
    matrix = MatrixLayout(
        rows=matrix_size,
        columns=matrix_size,
        leading_dimension=matrix_size,
        element_bytes=element_bytes,
    )
    requests = matrix_size * matrix_size // WARP_SIZE
    row_request = summarize_coalescing((analyze_coalescing(row_access(matrix, row=0)),))
    column_request = summarize_coalescing(
        (analyze_coalescing(column_access(matrix, column=0)),)
    )
    return TransposeAnalysis(
        leading_dimension=leading_dimension,
        element_bytes=element_bytes,
        matrix_size=matrix_size,
        shared_row_writes=row_writes,
        shared_column_reads=column_reads,
        naive_loads=row_request.scaled(requests),
        naive_stores=column_request.scaled(requests),
        tiled_loads=row_request.scaled(requests),
        tiled_stores=row_request.scaled(requests),
    )
