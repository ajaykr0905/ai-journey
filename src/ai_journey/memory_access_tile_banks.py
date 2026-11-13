"""Shared-memory bank analysis of the tiled matmul inner loop.

Thread ``(tx, ty)`` of a T x T block has thread ID ``tx + ty * T`` and each
warp holds 32 consecutive thread IDs (CUDA C++ Programming Guide v12.6,
sections 2.2 and 4.1). In inner-loop step ``i`` a thread reads
``A_tile[ty][i]`` and ``B_tile[i][tx]`` (CUDA C++ Best Practices Guide v13.5,
section 10.2.3.2):

- A-tile reads: every lane of one tile row reads the same word (broadcast);
- B-tile reads: lanes read consecutive words of one tile row (conflict-free);
- a B tile stored column-major and read as ``B_cm[tx][i]`` strides lanes by
  its leading dimension, which conflicts unless it is padded to T + 1.

Each tile is a separate shared array starting at bank 0. One request is
analyzed per warp per inner-loop step; the stores that fill the tiles, the
synchronization between phases, and occupancy are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory_access_banks import (
    BankAnalysis,
    BankSummary,
    analyze_banks,
    summarize_banks,
)
from .memory_access_pattern import WARP_SIZE, MemoryAccessError, WarpAccess
from .memory_access_tiling import MAX_MATMUL_TILE

INNER_LOOP_LAYOUTS = (
    "a_row_major",
    "b_row_major",
    "b_column_major",
    "b_column_major_padded",
)


def _tile(tile: object) -> int:
    if type(tile) is not int:
        raise TypeError("tile must be an integer")
    if not 1 <= tile <= MAX_MATMUL_TILE:
        raise MemoryAccessError(f"tile must be in [1, {MAX_MATMUL_TILE}]")
    return tile


def tile_warps(tile: int) -> int:
    """Return how many warps one T x T thread block occupies."""

    width = _tile(tile)
    return -(-width * width // WARP_SIZE)


def tile_warp_threads(tile: int, warp: int) -> tuple[tuple[int, int] | None, ...]:
    """Return ``(ty, tx)`` for each lane of one warp; ``None`` past the block."""

    width = _tile(tile)
    if type(warp) is not int:
        raise TypeError("warp must be an integer")
    if not 0 <= warp < tile_warps(width):
        raise MemoryAccessError("warp is outside the thread block")
    lanes = []
    for lane in range(WARP_SIZE):
        thread = warp * WARP_SIZE + lane
        lanes.append(divmod(thread, width) if thread < width * width else None)
    return tuple(lanes)


def _element_offset(layout: str, tile: int, step: int, ty: int, tx: int) -> int:
    if layout == "a_row_major":
        return ty * tile + step
    if layout == "b_row_major":
        return step * tile + tx
    if layout == "b_column_major":
        return tx * tile + step
    return tx * (tile + 1) + step


def inner_loop_access(
    tile: int, *, warp: int, step: int, layout: str, element_bytes: int = 4
) -> WarpAccess:
    """Return one warp's shared-memory read for one inner-loop step."""

    width = _tile(tile)
    if layout not in INNER_LOOP_LAYOUTS:
        raise MemoryAccessError(
            f"layout must be one of {', '.join(INNER_LOOP_LAYOUTS)}"
        )
    if type(step) is not int:
        raise TypeError("step must be an integer")
    if not 0 <= step < width:
        raise MemoryAccessError("step must index the tile's inner dimension")
    if type(element_bytes) is not int:
        raise TypeError("element_bytes must be an integer")
    return WarpAccess(
        element_bytes,
        tuple(
            None
            if thread is None
            else _element_offset(layout, width, step, *thread) * element_bytes
            for thread in tile_warp_threads(width, warp)
        ),
    )


@dataclass(frozen=True)
class InnerLoopBankAnalysis:
    """Wavefronts of every warp and inner-loop step for each tile layout."""

    tile: int
    element_bytes: int
    summaries: tuple[tuple[str, BankSummary], ...]
    examples: tuple[tuple[str, BankAnalysis], ...]

    @property
    def warps(self) -> int:
        return tile_warps(self.tile)

    def summary(self, layout: str) -> BankSummary:
        return dict(self.summaries)[layout]

    def example(self, layout: str) -> BankAnalysis:
        """Return the warp 0, step 0 request for one layout."""

        return dict(self.examples)[layout]

    def to_mapping(self) -> dict[str, object]:
        examples = dict(self.examples)
        return {
            "tile": self.tile,
            "element_bytes": self.element_bytes,
            "warps": self.warps,
            "steps": self.tile,
            "layouts": {
                layout: {
                    "all_steps": summary.to_mapping(),
                    "warp0_step0": examples[layout].to_mapping(),
                }
                for layout, summary in self.summaries
            },
        }


def analyze_inner_loop_banks(
    tile: int, *, element_bytes: int = 4
) -> InnerLoopBankAnalysis:
    """Analyze every warp's reads in every inner-loop step of one tile phase."""

    width = _tile(tile)
    summaries = []
    examples = []
    for layout in INNER_LOOP_LAYOUTS:
        analyses = tuple(
            analyze_banks(
                inner_loop_access(
                    width,
                    warp=warp,
                    step=step,
                    layout=layout,
                    element_bytes=element_bytes,
                )
            )
            for warp in range(tile_warps(width))
            for step in range(width)
        )
        summaries.append((layout, summarize_banks(analyses)))
        examples.append((layout, analyses[0]))
    return InnerLoopBankAnalysis(
        tile=width,
        element_bytes=element_bytes,
        summaries=tuple(summaries),
        examples=tuple(examples),
    )
