"""Run the Day 77 coarsened and register-blocked schedules on the CPU.

Each run holds every block of the grid at once in numpy float32 arrays and
follows the steps of the ``gpu_checklist_cuda`` text, phase by phase. A
product of two float32 values is exact in float64, and the rounding helper of
``matmul_metal_reference`` rounds each exact sum once to float32. Each output
thus gets the rounding chain of ``fma_replay`` only if the schedule adds its
terms in increasing k from +0.0, and a test compares the bits.

* Coarsened: a phase opens with the load of its A tile. Then the phase steps
  run once for each of the C tiles of B: load, barrier, fma, barrier. The fma
  steps of one phase use other accumulators and other B tiles, so their
  values do not depend on their order, and the simulator runs them together
  at the end of the phase.
* Blocked: the phase steps run once per phase. In the load step, each thread
  stages its group of 4 floats of A, transposed, and of B. With the flag of
  the text set, an inside group is one float4 load. The simulator computes
  each float4 offset, and it counts a load as unsafe if its offset is not a
  multiple of 4 or a float of it is outside the row.

The loads come from the masks that the text uses, and only an element inside
its matrix counts. ``ChecklistTraffic`` declares the same counts as closed
forms, so a run can show that it matches them. The shared indices do not
depend on ``blockIdx``, so an access log of one block stands for every block.
Like the Day 76 log, it counts a read of a slot that another thread wrote in
the same barrier interval, and a write of a slot that another thread read in
it. The simulator runs each step for all threads before the next step, so its
values are those of one legal order, and the counts tell whether every order
gives the same values.

This module runs on the CPU only. It compiles and runs no CUDA, and its
counts are a model of the text, not measurements of a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .gpu_checklist_cuda import (
    VECTOR_WIDTH,
    BlockedKernel,
    ChecklistKernel,
    CoarsenedKernel,
    GpuChecklistError,
    require_kernel,
    require_shape,
)
from .gpu_tiled_blocksim import PHASE_STEPS, SharedHazards, require_steps
from .kernel_launch_config import LaunchConfig, ceil_div
from .matmul_metal_reference import _operands, _round_fused, matrix_sha256
from .memory_access_tiling import MatmulShape

ALIGNED_BASES = (True, True)


def require_aligned(aligned: object) -> tuple[bool, bool]:
    """Return whether the bases of A and B are 16-byte aligned, as two booleans.

    ``cudaMalloc`` returns addresses aligned to at least 256 bytes (CUDA C++
    Programming Guide, device memory accesses), so the default is aligned. A
    test can clear a flag to reach the scalar loads of an aligned shape.
    """

    if (
        type(aligned) is not tuple
        or len(aligned) != 2
        or any(type(item) is not bool for item in aligned)
    ):
        raise GpuChecklistError("aligned must be a tuple of two booleans")
    return aligned


def vector_flags(
    kernel: ChecklistKernel, shape: MatmulShape, aligned: tuple[bool, bool]
) -> tuple[bool, bool]:
    """Return the float4 flags of the text: row length a multiple of 4 and aligned."""

    checked = require_kernel(kernel)
    case = require_shape(shape)
    bases = require_aligned(aligned)
    if type(checked) is CoarsenedKernel:
        return False, False
    a_vector = case.k % VECTOR_WIDTH == 0 and bases[0]
    b_vector = case.n % VECTOR_WIDTH == 0 and bases[1]
    return a_vector, b_vector


@dataclass(frozen=True)
class ChecklistTraffic:
    """The declared counts of one launch, as closed forms in the shape.

    Each block column reads all of A once, and each block row reads all of
    B once: the coarsened kernel loads one A tile per phase for its C tiles
    of B, and its C tiles cover all columns of its block. Each slot that a
    phase writes and that holds no element of A or B is zero-filled. With a
    float4 flag set, the row length is a multiple of 4, so each group of 4
    is inside its matrix or outside it as a whole, and an inside group is one
    float4 load.
    """

    kernel: ChecklistKernel
    shape: MatmulShape
    a_vector: bool = False
    b_vector: bool = False

    def __post_init__(self) -> None:
        checked = require_kernel(self.kernel)
        require_shape(self.shape)
        if type(self.a_vector) is not bool or type(self.b_vector) is not bool:
            raise TypeError("the float4 flags must be booleans")
        if type(checked) is CoarsenedKernel and (self.a_vector or self.b_vector):
            raise GpuChecklistError("the coarsened kernel has no float4 loads")
        if (self.a_vector and self.shape.k % VECTOR_WIDTH) or (
            self.b_vector and self.shape.n % VECTOR_WIDTH
        ):
            raise GpuChecklistError("a float4 flag needs a row length that 4 divides")

    @property
    def launch(self) -> LaunchConfig:
        return self.kernel.launch(self.shape)

    @property
    def phases(self) -> int:
        step = (
            self.kernel.tile
            if type(self.kernel) is CoarsenedKernel
            else self.kernel.block_k
        )
        return ceil_div(self.shape.k, step)

    @property
    def a_loads(self) -> int:
        return self.launch.grid.x * self.shape.m * self.shape.k

    @property
    def b_loads(self) -> int:
        return self.launch.grid.y * self.shape.k * self.shape.n

    @property
    def a_vector_loads(self) -> int:
        return self.a_loads // VECTOR_WIDTH if self.a_vector else 0

    @property
    def b_vector_loads(self) -> int:
        return self.b_loads // VECTOR_WIDTH if self.b_vector else 0

    @property
    def tile_slot_writes(self) -> int:
        """Return the slots that all blocks write: each tile slot in each phase."""

        kernel = self.kernel
        if type(kernel) is CoarsenedKernel:
            per_phase = (1 + kernel.coarsening) * kernel.tile * kernel.tile
        else:
            per_phase = kernel.block_k * (kernel.block_m + kernel.block_n)
        return self.launch.blocks * self.phases * per_phase

    @property
    def zero_filled_slots(self) -> int:
        return self.tile_slot_writes - self.a_loads - self.b_loads

    @property
    def barriers_per_block(self) -> int:
        kernel = self.kernel
        pairs = kernel.coarsening if type(kernel) is CoarsenedKernel else 1
        return 2 * pairs * self.phases

    @property
    def fma_steps(self) -> int:
        """Return the fma operations of all launched threads, idle outputs too."""

        return (
            self.launch.launched_threads * self.kernel.outputs_per_thread * self.shape.k
        )

    @property
    def shared_reads(self) -> int:
        """Return the shared loads: 2 per fma, or TM + TN per step of a thread."""

        kernel = self.kernel
        if type(kernel) is CoarsenedKernel:
            return 2 * self.fma_steps
        per_step = kernel.thread_m + kernel.thread_n
        return self.launch.launched_threads * per_step * self.shape.k

    @property
    def stores(self) -> int:
        return self.shape.m * self.shape.n

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel.name,
            "shape": self.shape.to_mapping(),
            "vector": {"a": self.a_vector, "b": self.b_vector},
            "phases": self.phases,
            "a_loads": self.a_loads,
            "b_loads": self.b_loads,
            "a_float4_loads": self.a_vector_loads,
            "b_float4_loads": self.b_vector_loads,
            "tile_slot_writes": self.tile_slot_writes,
            "zero_filled_slots": self.zero_filled_slots,
            "barriers_per_block": self.barriers_per_block,
            "fma_steps": self.fma_steps,
            "shared_reads": self.shared_reads,
            "stores": self.stores,
        }


class _AccessLog:
    """The shared memory accesses of one block, interval by interval.

    Each slot has one writer, so a write step is an array that names the
    writer thread of each slot that it writes, or -1. A read step is a
    boolean [thread, slot] matrix. The log compares the steps of an interval
    only when a write meets earlier reads or a read meets earlier writes, so
    a schedule with each barrier in place costs no matrix work.
    """

    def __init__(self, threads: int, slots: int) -> None:
        self.thread = numpy.arange(threads)
        self.writer = numpy.full(slots, -1)
        self.written = False
        self.reads: list[numpy.ndarray] = []
        self.barriers = 0
        self.read_after_write = 0
        self.write_after_read = 0

    def barrier(self) -> None:
        self.barriers += 1
        if self.written:
            self.writer.fill(-1)
            self.written = False
        self.reads.clear()

    def write(self, writers: numpy.ndarray) -> None:
        slots = numpy.flatnonzero(writers >= 0)
        if self.reads:
            readers = numpy.logical_or.reduce(self.reads)
            others = readers[:, slots].sum(axis=0) - readers[writers[slots], slots]
            self.write_after_read += int(others.sum())
        self.writer[slots] = writers[slots]
        self.written = True

    def read(self, reads: numpy.ndarray) -> None:
        if self.written:
            fresh = self.writer >= 0
            foreign = self.writer[None, :] != self.thread[:, None]
            raced = reads & fresh[None, :] & foreign
            self.read_after_write += int(numpy.count_nonzero(raced))
        self.reads.append(reads)

    def hazards(self) -> SharedHazards:
        return SharedHazards(
            barriers=self.barriers,
            read_after_write=self.read_after_write,
            write_after_read=self.write_after_read,
        )


def _one_writer(*writers: numpy.ndarray) -> bool:
    """Return whether the write steps of a phase give each slot one writer."""

    written = numpy.stack(writers) >= 0
    return bool(numpy.all(written.sum(axis=0) == 1))


@dataclass(frozen=True)
class ChecklistBlockRun:
    """One simulated launch: its counts, its hazards and its output."""

    kernel: ChecklistKernel
    shape: MatmulShape
    launch: LaunchConfig
    steps: tuple[str, ...]
    a_vector: bool
    b_vector: bool
    phases: int
    inner_steps: int
    a_loads: int
    b_loads: int
    a_vector_loads: int
    b_vector_loads: int
    unsafe_vector_loads: int
    tile_slot_writes: int
    one_writer_per_slot: bool
    shared_reads: int
    fma_steps: int
    stores: int
    stored_once: bool
    hazards: SharedHazards
    output: numpy.ndarray = field(repr=False, compare=False)

    @property
    def output_sha256(self) -> str:
        return matrix_sha256(self.output)

    @property
    def zero_filled_slots(self) -> int:
        return self.tile_slot_writes - self.a_loads - self.b_loads

    @property
    def traffic(self) -> ChecklistTraffic:
        return ChecklistTraffic(self.kernel, self.shape, self.a_vector, self.b_vector)

    @property
    def matches_traffic(self) -> bool:
        """Return whether every counted value is the declared closed form."""

        model = self.traffic
        return (
            self.phases == model.phases
            and self.a_loads == model.a_loads
            and self.b_loads == model.b_loads
            and self.a_vector_loads == model.a_vector_loads
            and self.b_vector_loads == model.b_vector_loads
            and self.unsafe_vector_loads == 0
            and self.tile_slot_writes == model.tile_slot_writes
            and self.shared_reads == model.shared_reads
            and self.fma_steps == model.fma_steps
            and self.stores == model.stores
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel.name,
            "shape": self.shape.to_mapping(),
            "launch": self.launch.geometry(),
            "steps": list(self.steps),
            "vector": {"a": self.a_vector, "b": self.b_vector},
            "phases": self.phases,
            "inner_steps": self.inner_steps,
            "global_loads": {
                "a": self.a_loads,
                "b": self.b_loads,
                "a_float4": self.a_vector_loads,
                "b_float4": self.b_vector_loads,
                "unsafe_float4": self.unsafe_vector_loads,
                "zero_filled_slots": self.zero_filled_slots,
                "tile_slot_writes": self.tile_slot_writes,
            },
            "one_writer_per_slot": self.one_writer_per_slot,
            "shared_reads": self.shared_reads,
            "fma_steps": self.fma_steps,
            "stores": self.stores,
            "stored_once": self.stored_once,
            "matches_traffic": self.matches_traffic,
            "hazards": self.hazards.to_mapping(),
            "output_sha256": self.output_sha256,
        }


def _store(
    total: numpy.ndarray, rows: numpy.ndarray, cols: numpy.ndarray, m: int, n: int
) -> tuple[numpy.ndarray, int, bool]:
    """Store each output inside C, as the guard does; count the stores."""

    inside = (rows < m) & (cols < n)
    flat = numpy.broadcast_to(rows * n + cols, total.shape)[
        numpy.broadcast_to(inside, total.shape)
    ]
    output = numpy.full(m * n, numpy.nan, dtype=numpy.float32)
    output[flat] = total[numpy.broadcast_to(inside, total.shape)]
    stored = numpy.bincount(flat, minlength=m * n)
    return output.reshape(m, n), int(flat.size), bool(numpy.all(stored == 1))


def _shape_of(a: numpy.ndarray, b: numpy.ndarray) -> MatmulShape:
    return require_shape(MatmulShape(int(a.shape[0]), int(b.shape[1]), int(a.shape[1])))


def run_coarsened_blocks(
    left: numpy.ndarray,
    right: numpy.ndarray,
    *,
    kernel: CoarsenedKernel,
    steps: tuple[str, ...] = PHASE_STEPS,
) -> ChecklistBlockRun:
    """Run every block of one coarsened launch; ``steps`` repeat for each B tile."""

    a, b = _operands(left, right)
    if type(kernel) is not CoarsenedKernel:
        raise TypeError("kernel must be CoarsenedKernel")
    schedule = require_steps(steps)
    shape = _shape_of(a, b)
    m, n, k = shape.m, shape.n, shape.k
    launch = kernel.launch(shape)
    t, factor = kernel.tile, kernel.coarsening
    gy, gx = launch.grid.y, launch.grid.x
    lanes = numpy.arange(t)
    rows = numpy.arange(gy)[:, None] * t + lanes
    cols = (
        numpy.arange(gx)[:, None, None] * kernel.width
        + numpy.arange(factor)[None, :, None] * t
        + lanes
    )
    blocks = (gy, gx, t, t)
    a_tile = numpy.zeros(blocks, dtype=numpy.float32)
    b_tiles = numpy.zeros((gy, gx, factor, t, t), dtype=numpy.float32)
    total = numpy.zeros((gy, gx, factor, t, t), dtype=numpy.float32)
    product = numpy.empty(total.shape, dtype=numpy.float64)
    threads = t * t
    log = _AccessLog(threads, 2 * threads)
    none = numpy.full(threads, -1)
    writes_a = numpy.concatenate((numpy.arange(threads), none))
    writes_b = numpy.concatenate((none, numpy.arange(threads)))
    ty, tx = numpy.divmod(numpy.arange(threads), t)
    reads_of: dict[int, numpy.ndarray] = {}
    a_loads = b_loads = slot_writes = shared_reads = fma_steps = inner_steps = 0
    phases = ceil_div(k, t)
    for phase in range(phases):
        count = min(t, k - phase * t)
        if count not in reads_of:
            a_reads = numpy.zeros((threads, threads), dtype=bool)
            b_reads = numpy.zeros((threads, threads), dtype=bool)
            for i in range(count):
                a_reads[numpy.arange(threads), ty * t + i] = True
                b_reads[numpy.arange(threads), i * t + tx] = True
            reads_of[count] = numpy.concatenate((a_reads, b_reads), axis=1)
        a_col = phase * t + lanes
        b_row = phase * t + lanes
        # The A load opens the phase, before the rounds of B tiles.
        log.write(writes_a)
        inside = numpy.broadcast_to(
            (rows < m)[:, None, :, None] & (a_col < k)[None, None, None, :], blocks
        )
        a_tile[...] = numpy.where(
            inside,
            a[
                numpy.minimum(rows, m - 1)[:, None, :, None],
                numpy.minimum(a_col, k - 1)[None, None, None, :],
            ],
            numpy.float32(0),
        )
        a_loads += int(numpy.count_nonzero(inside))
        slot_writes += a_tile.size
        for j in range(factor):
            for step in schedule:
                if step == "barrier":
                    log.barrier()
                elif step == "load":
                    log.write(writes_b)
                    b_inside = numpy.broadcast_to(
                        (b_row < k)[None, None, :, None]
                        & (cols[:, j] < n)[None, :, None, :],
                        blocks,
                    )
                    b_tiles[:, :, j] = numpy.where(
                        b_inside,
                        b[
                            numpy.minimum(b_row, k - 1)[None, None, :, None],
                            numpy.minimum(cols[:, j], n - 1)[None, :, None, :],
                        ],
                        numpy.float32(0),
                    )
                    b_loads += int(numpy.count_nonzero(b_inside))
                    slot_writes += a_tile.size
                else:
                    log.read(reads_of[count])
                    shared_reads += 2 * count * a_tile.size
        a_wide = a_tile.astype(numpy.float64)
        b_wide = b_tiles.astype(numpy.float64)
        for i in range(count):
            numpy.multiply(
                a_wide[:, :, None, :, i, None], b_wide[:, :, :, i, None, :], out=product
            )
            total = _round_fused(product, total.astype(numpy.float64))
        inner_steps += count
        fma_steps += count * total.size
    output, stores, stored_once = _store(
        total, rows[:, None, None, :, None], cols[None, :, :, None, :], m, n
    )
    return ChecklistBlockRun(
        kernel=kernel,
        shape=shape,
        launch=launch,
        steps=schedule,
        a_vector=False,
        b_vector=False,
        phases=phases,
        inner_steps=inner_steps,
        a_loads=a_loads,
        b_loads=b_loads,
        a_vector_loads=0,
        b_vector_loads=0,
        unsafe_vector_loads=0,
        tile_slot_writes=slot_writes,
        one_writer_per_slot=_one_writer(writes_a, writes_b),
        shared_reads=shared_reads,
        fma_steps=fma_steps,
        stores=stores,
        stored_once=stored_once,
        hazards=log.hazards(),
        output=output,
    )


def _stage(
    source: numpy.ndarray,
    lines: numpy.ndarray,
    starts: numpy.ndarray,
    limits: tuple[int, int],
    vector: bool,
) -> tuple[numpy.ndarray, int, int, int]:
    """Load one group of 4 floats per thread, as the guarded text loads it.

    ``lines`` and ``starts`` give the row and the first column of each group
    in ``source``, which has ``limits`` rows and columns. With ``vector``, an
    inside group is one float4 load, and it is unsafe if its offset is not a
    multiple of 4 or a float of it is outside the row. Without it, each
    inside float is one scalar load. Return the values and the float4,
    scalar and unsafe counts.
    """

    rows, columns = limits
    lanes = starts[..., None] + numpy.arange(VECTOR_WIDTH)
    inside = (lines < rows)[..., None] & (lanes < columns)
    if vector:
        group = (lines < rows) & (starts < columns)
        offsets = lines * columns + starts
        unsafe = group & ((offsets % VECTOR_WIDTH != 0) | ~inside.all(axis=-1))
        loaded = numpy.broadcast_to(group[..., None], inside.shape)
        counts = (int(numpy.count_nonzero(group)), 0, int(numpy.count_nonzero(unsafe)))
    else:
        loaded = inside
        counts = (0, int(numpy.count_nonzero(inside)), 0)
    values = numpy.where(
        loaded,
        source[
            numpy.minimum(lines, rows - 1)[..., None], numpy.minimum(lanes, columns - 1)
        ],
        numpy.float32(0),
    )
    return values, *counts


def run_blocked_blocks(
    left: numpy.ndarray,
    right: numpy.ndarray,
    *,
    kernel: BlockedKernel,
    steps: tuple[str, ...] = PHASE_STEPS,
    aligned: tuple[bool, bool] = ALIGNED_BASES,
) -> ChecklistBlockRun:
    """Run every block of one register-blocked launch through the phase steps.

    ``aligned`` says whether the bases of A and B are 16-byte aligned, which
    the text tests with the row lengths to choose its float4 loads.
    """

    a, b = _operands(left, right)
    if type(kernel) is not BlockedKernel:
        raise TypeError("kernel must be BlockedKernel")
    schedule = require_steps(steps)
    shape = _shape_of(a, b)
    m, n, k = shape.m, shape.n, shape.k
    a_vector, b_vector = vector_flags(kernel, shape, aligned)
    launch = kernel.launch(shape)
    bm, bn, bk = kernel.block_m, kernel.block_n, kernel.block_k
    tm, tn = kernel.thread_m, kernel.thread_n
    x, _ = kernel.block
    threads = kernel.threads_per_block
    gy, gx = launch.grid.y, launch.grid.x
    thread = numpy.arange(threads)
    a_r, a_c = thread // kernel.a_groups_per_row, thread % kernel.a_groups_per_row * 4
    b_r, b_c = thread // kernel.b_groups_per_row, thread % kernel.b_groups_per_row * 4
    quad = numpy.arange(VECTOR_WIDTH)
    a_rows = numpy.broadcast_to(
        (numpy.arange(gy)[:, None] * bm + a_r)[:, None, :], (gy, gx, threads)
    )
    b_cols = numpy.broadcast_to(
        (numpy.arange(gx)[:, None] * bn + b_c)[None, :, :], (gy, gx, threads)
    )
    a_tile = numpy.zeros((gy, gx, bk, bm), dtype=numpy.float32)
    b_tile = numpy.zeros((gy, gx, bk, bn), dtype=numpy.float32)
    total = numpy.zeros((gy, gx, bm, bn), dtype=numpy.float32)
    product = numpy.empty(total.shape, dtype=numpy.float64)
    a_slots = (a_c[:, None] + quad) * bm + a_r[:, None]
    b_slots = bk * bm + b_r[:, None] * bn + b_c[:, None] + quad
    writes_a = numpy.full(bk * (bm + bn), -1)
    writes_b = writes_a.copy()
    writes_a[a_slots] = numpy.broadcast_to(thread[:, None], a_slots.shape)
    writes_b[b_slots] = numpy.broadcast_to(thread[:, None], b_slots.shape)
    writes = numpy.maximum(writes_a, writes_b)
    one_writer = _one_writer(writes_a, writes_b) and (
        numpy.unique(a_slots).size == a_slots.size == bk * bm
        and numpy.unique(b_slots).size == b_slots.size == bk * bn
    )
    ty, tx = numpy.divmod(thread, x)
    log = _AccessLog(threads, bk * (bm + bn))
    reads_of: dict[int, numpy.ndarray] = {}
    counts = dict.fromkeys(("a4", "a1", "b4", "b1", "unsafe"), 0)
    slot_writes = shared_reads = fma_steps = inner_steps = 0
    phases = ceil_div(k, bk)
    for phase in range(phases):
        count = min(bk, k - phase * bk)
        if count not in reads_of:
            reads = numpy.zeros((threads, bk * (bm + bn)), dtype=bool)
            for i in range(count):
                for mi in range(tm):
                    reads[thread, i * bm + ty * tm + mi] = True
                for ni in range(tn):
                    reads[thread, bk * bm + i * bn + tx * tn + ni] = True
            reads_of[count] = reads
        for step in schedule:
            if step == "barrier":
                log.barrier()
            elif step == "load":
                log.write(writes)
                a_starts = numpy.broadcast_to(phase * bk + a_c, a_rows.shape)
                values, vector, scalar, unsafe = _stage(
                    a, a_rows, a_starts, (m, k), a_vector
                )
                a_tile[:, :, a_c[:, None] + quad, a_r[:, None]] = values
                counts["a4"] += vector
                counts["a1"] += scalar
                counts["unsafe"] += unsafe
                b_lines = numpy.broadcast_to(phase * bk + b_r, b_cols.shape)
                values, vector, scalar, unsafe = _stage(
                    b, b_lines, b_cols, (k, n), b_vector
                )
                b_tile[:, :, b_r[:, None], b_c[:, None] + quad] = values
                counts["b4"] += vector
                counts["b1"] += scalar
                counts["unsafe"] += unsafe
                slot_writes += a_tile.size + b_tile.size
            else:
                log.read(reads_of[count])
                a_wide = a_tile.astype(numpy.float64)
                b_wide = b_tile.astype(numpy.float64)
                for i in range(count):
                    # a_reg[mi] and b_reg[ni] copy As[i][ty * TM + mi] and
                    # Bs[i][tx * TN + ni], so the products are these outers.
                    numpy.multiply(
                        a_wide[:, :, i, :, None], b_wide[:, :, i, None, :], out=product
                    )
                    total = _round_fused(product, total.astype(numpy.float64))
                shared_reads += threads * gy * gx * (tm + tn) * count
                fma_steps += total.size * count
                inner_steps += count
    rows = numpy.arange(gy)[:, None] * bm + numpy.arange(bm)
    cols = numpy.arange(gx)[:, None] * bn + numpy.arange(bn)
    output, stores, stored_once = _store(
        total, rows[:, None, :, None], cols[None, :, None, :], m, n
    )
    return ChecklistBlockRun(
        kernel=kernel,
        shape=shape,
        launch=launch,
        steps=schedule,
        a_vector=a_vector,
        b_vector=b_vector,
        phases=phases,
        inner_steps=inner_steps,
        a_loads=VECTOR_WIDTH * counts["a4"] + counts["a1"],
        b_loads=VECTOR_WIDTH * counts["b4"] + counts["b1"],
        a_vector_loads=counts["a4"],
        b_vector_loads=counts["b4"],
        unsafe_vector_loads=counts["unsafe"],
        tile_slot_writes=slot_writes,
        one_writer_per_slot=bool(one_writer),
        shared_reads=shared_reads,
        fma_steps=fma_steps,
        stores=stores,
        stored_once=stored_once,
        hazards=log.hazards(),
        output=output,
    )


def run_blocks(
    left: numpy.ndarray,
    right: numpy.ndarray,
    *,
    kernel: ChecklistKernel,
    steps: tuple[str, ...] = PHASE_STEPS,
    aligned: tuple[bool, bool] = ALIGNED_BASES,
) -> ChecklistBlockRun:
    """Run one launch of a coarsened or a blocked kernel."""

    checked = require_kernel(kernel)
    require_aligned(aligned)
    if type(checked) is CoarsenedKernel:
        return run_coarsened_blocks(left, right, kernel=checked, steps=steps)
    return run_blocked_blocks(left, right, kernel=checked, steps=steps, aligned=aligned)
