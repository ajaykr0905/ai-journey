"""Run the tiled CUDA kernel's block schedule on the CPU in numpy float32.

``run_tiled_blocks`` holds every block of the grid at once, as ``(gy, gx, T,
T)`` arrays, and runs the steps of the ``gpu_tiled_cuda`` text phase by phase
in the order of ``PHASE_STEPS``:

1. ``load``: thread ``(ty, tx)`` of each block writes ``A[row, phase * T +
   tx]`` into ``As[ty][tx]`` and ``B[phase * T + ty, col]`` into
   ``Bs[ty][tx]``, or 0 for an element outside its matrix. Only an element
   inside its matrix counts as a global load.
2. ``barrier``: a new barrier interval starts.
3. ``fma``: for ``i < min(T, k - phase * T)``, each thread rounds ``sum =
   fma(As[ty][i], Bs[i][tx], sum)`` once to float32. A product of two
   float32 values is exact in float64, and ``matmul_metal_reference`` rounds
   the exact sum once.
4. ``barrier``.

After the last phase, each thread inside C stores its sum, and each element
of C must be stored exactly once. The global loads are counted from the
masks that the loads use, so a report can compare them with the analytical
``memory_access_tiling.TiledMatmulTraffic``.

The shared indices do not depend on ``blockIdx``, and a thread writes a zero
like a loaded value, so the accesses of one block stand for every block. The
simulator logs them between barriers. A read of a slot that another thread
wrote in the same interval, or a write of a slot that another thread read in
it, is a hazard: on a GPU its result would depend on the order of the warps.
The simulator runs each step for all threads before the next step, so its
values are those of one legal order. The hazard counts tell whether every
order gives the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .gpu_tiled_cuda import GpuTiledError, require_shape, require_tile
from .kernel_launch_config import LaunchConfig
from .matmul_metal_reference import _operands, _round_fused, matrix_sha256
from .matmul_model_kernels import X_COLUMNS, matmul_launch
from .memory_access_tiling import MatmulShape, TiledMatmulTraffic

PHASE_STEPS = ("load", "barrier", "fma", "barrier")
STEP_NAMES = ("load", "barrier", "fma")
MAX_STEPS = 8
_TILE_ARRAYS = 2


def require_steps(steps: object) -> tuple[str, ...]:
    """Return a phase schedule: one load before one fma, and any barriers.

    A schedule can drop or move a barrier, so the hazard log can show what
    each barrier prevents. The fma must follow the load, as in the text.
    """

    if type(steps) is not tuple or not 2 <= len(steps) <= MAX_STEPS:
        raise GpuTiledError(f"steps must be a tuple of 2 to {MAX_STEPS} names")
    if any(type(step) is not str or step not in STEP_NAMES for step in steps):
        raise GpuTiledError(f"each step must be one of {STEP_NAMES}")
    if steps.count("load") != 1 or steps.count("fma") != 1:
        raise GpuTiledError("a phase holds exactly one load and one fma step")
    if steps.index("load") > steps.index("fma"):
        raise GpuTiledError("the fma step must follow the load step")
    return steps


@dataclass(frozen=True)
class SharedHazards:
    """Shared-memory accesses of one block that no barrier separates.

    ``read_after_write`` counts reads of a slot that another thread wrote in
    the same barrier interval. ``write_after_read`` counts, for each write,
    the other threads that read the slot in that interval.
    """

    barriers: int
    read_after_write: int
    write_after_read: int

    @property
    def race_free(self) -> bool:
        return self.read_after_write == self.write_after_read == 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "barriers": self.barriers,
            "read_after_write": self.read_after_write,
            "write_after_read": self.write_after_read,
            "race_free": self.race_free,
        }


class _SharedLog:
    """The shared-memory accesses of one block, interval by interval.

    Slot ``s`` of a tile is ``tile[s // T][s % T]``, and thread ``t`` is
    ``(ty, tx) = divmod(t, T)``. The load step of thread ``t`` writes slot
    ``t`` of both tiles.
    """

    def __init__(self, tile: int) -> None:
        threads = tile * tile
        self.tile = tile
        self.interval = 0
        self.barriers = 0
        self.read_after_write = 0
        self.write_after_read = 0
        self.thread = numpy.arange(threads)
        self.written = numpy.full((_TILE_ARRAYS, threads), -1)
        self.writer = numpy.full((_TILE_ARRAYS, threads), -1)
        self.readers = numpy.zeros((_TILE_ARRAYS, threads, threads), dtype=bool)
        self._reads: dict[int, numpy.ndarray] = {}

    def barrier(self) -> None:
        self.interval += 1
        self.barriers += 1
        self.readers[:] = False

    def load(self) -> None:
        if self.readers.any():
            others = self.readers & (self.thread[:, None] != self.thread[None, :])
            self.write_after_read += int(numpy.count_nonzero(others))
        self.written[:] = self.interval
        self.writer[:] = self.thread

    def reads(self, count: int) -> numpy.ndarray:
        """Return ``[tile, thread, slot]``: what one fma step of ``count`` reads.

        Thread ``(ty, tx)`` reads ``As[ty][i]`` and ``Bs[i][tx]`` for each
        ``i < count``.
        """

        if count not in self._reads:
            # Threads and slots share one numbering, so (ty, tx) of thread t
            # and (row, col) of slot s come from the same divmod.
            ty, tx = numpy.divmod(self.thread, self.tile)
            row, col = ty, tx
            a_reads = (row[None, :] == ty[:, None]) & (col[None, :] < count)
            b_reads = (col[None, :] == tx[:, None]) & (row[None, :] < count)
            self._reads[count] = numpy.stack((a_reads, b_reads))
        return self._reads[count]

    def fma(self, count: int) -> None:
        reads = self.reads(count)
        fresh = self.written == self.interval
        if fresh.any():
            foreign = self.writer[:, None, :] != self.thread[None, :, None]
            raced = reads & fresh[:, None, :] & foreign
            self.read_after_write += int(numpy.count_nonzero(raced))
        self.readers |= reads

    def hazards(self) -> SharedHazards:
        return SharedHazards(
            barriers=self.barriers,
            read_after_write=self.read_after_write,
            write_after_read=self.write_after_read,
        )


@dataclass(frozen=True)
class TiledBlockRun:
    """One simulated launch of the tiled kernel: counts, hazards and output."""

    tile: int
    shape: MatmulShape
    launch: LaunchConfig
    steps: tuple[str, ...]
    phases: int
    inner_steps: int
    a_loads: int
    b_loads: int
    zero_filled_slots: int
    tile_slot_writes: int
    stores: int
    stored_once: bool
    hazards: SharedHazards
    output: numpy.ndarray = field(repr=False, compare=False)

    @property
    def output_sha256(self) -> str:
        return matrix_sha256(self.output)

    @property
    def traffic(self) -> TiledMatmulTraffic:
        return TiledMatmulTraffic(self.shape, tile=self.tile)

    @property
    def matches_traffic(self) -> bool:
        """Return whether the counted loads are the analytical traffic model's."""

        model = self.traffic
        return (
            self.a_loads == model.useful_a_loads
            and self.b_loads == model.useful_b_loads
            and self.zero_filled_slots == model.padded_slots
            and self.tile_slot_writes == model.tile_slot_loads
            and self.stores == model.store_elements
            and self.phases == model.phases
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "tile": self.tile,
            "shape": self.shape.to_mapping(),
            "launch": self.launch.geometry(),
            "steps": list(self.steps),
            "phases": self.phases,
            "inner_steps": self.inner_steps,
            "global_loads": {
                "a": self.a_loads,
                "b": self.b_loads,
                "zero_filled_slots": self.zero_filled_slots,
                "tile_slot_writes": self.tile_slot_writes,
            },
            "stores": self.stores,
            "stored_once": self.stored_once,
            "matches_traffic": self.matches_traffic,
            "hazards": self.hazards.to_mapping(),
            "output_sha256": self.output_sha256,
        }


def run_tiled_blocks(
    left: numpy.ndarray,
    right: numpy.ndarray,
    *,
    tile: int,
    steps: tuple[str, ...] = PHASE_STEPS,
) -> TiledBlockRun:
    """Run every block of one tiled launch through the phase steps.

    ``left`` is A and ``right`` is B, both row-major float32. The grid is the
    Day 74 launch of a ``(T, T)`` block with x over the columns of C.
    """

    a, b = _operands(left, right)
    width = require_tile(tile)
    schedule = require_steps(steps)
    shape = require_shape(
        MatmulShape(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
    )
    m, n, k = shape.m, shape.n, shape.k
    launch = matmul_launch(shape, (width, width), X_COLUMNS)
    grid_y, grid_x = launch.grid.y, launch.grid.x
    lanes = numpy.arange(width)
    rows = numpy.arange(grid_y)[:, None] * width + lanes[None, :]
    cols = numpy.arange(grid_x)[:, None] * width + lanes[None, :]
    blocks = (grid_y, grid_x, width, width)
    a_tile = numpy.zeros(blocks, dtype=numpy.float32)
    b_tile = numpy.zeros(blocks, dtype=numpy.float32)
    total = numpy.zeros(blocks, dtype=numpy.float32)
    product = numpy.empty(blocks, dtype=numpy.float64)
    log = _SharedLog(width)
    a_loads = b_loads = slot_writes = inner_steps = 0
    phases = -(-k // width)
    for phase in range(phases):
        count = min(width, k - phase * width)
        for step in schedule:
            if step == "barrier":
                log.barrier()
            elif step == "load":
                log.load()
                a_col = phase * width + lanes
                b_row = phase * width + lanes
                a_inside = numpy.broadcast_to(
                    (rows < m)[:, None, :, None] & (a_col < k)[None, None, None, :],
                    blocks,
                )
                b_inside = numpy.broadcast_to(
                    (b_row < k)[None, None, :, None] & (cols < n)[None, :, None, :],
                    blocks,
                )
                a_values = a[
                    numpy.minimum(rows, m - 1)[:, None, :, None],
                    numpy.minimum(a_col, k - 1)[None, None, None, :],
                ]
                b_values = b[
                    numpy.minimum(b_row, k - 1)[None, None, :, None],
                    numpy.minimum(cols, n - 1)[None, :, None, :],
                ]
                a_tile[...] = numpy.where(a_inside, a_values, numpy.float32(0))
                b_tile[...] = numpy.where(b_inside, b_values, numpy.float32(0))
                a_loads += int(numpy.count_nonzero(a_inside))
                b_loads += int(numpy.count_nonzero(b_inside))
                slot_writes += 2 * a_tile.size
            else:
                log.fma(count)
                a_wide = a_tile.astype(numpy.float64)
                b_wide = b_tile.astype(numpy.float64)
                for i in range(count):
                    numpy.multiply(
                        a_wide[:, :, :, i, None], b_wide[:, :, None, i, :], out=product
                    )
                    total = _round_fused(product, total.astype(numpy.float64))
                inner_steps += count
    inside = (rows < m)[:, None, :, None] & (cols < n)[None, :, None, :]
    row_index = numpy.broadcast_to(rows[:, None, :, None], blocks)[inside]
    col_index = numpy.broadcast_to(cols[None, :, None, :], blocks)[inside]
    flat = row_index * n + col_index
    output = numpy.full(m * n, numpy.nan, dtype=numpy.float32)
    output[flat] = total[inside]
    stored = numpy.bincount(flat, minlength=m * n)
    return TiledBlockRun(
        tile=width,
        shape=shape,
        launch=launch,
        steps=schedule,
        phases=phases,
        inner_steps=inner_steps,
        a_loads=a_loads,
        b_loads=b_loads,
        zero_filled_slots=slot_writes - a_loads - b_loads,
        tile_slot_writes=slot_writes,
        stores=int(flat.size),
        stored_once=bool(numpy.all(stored == 1)),
        hazards=log.hazards(),
        output=output.reshape(m, n),
    )
