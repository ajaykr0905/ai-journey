"""Device and threadgroup traffic of every checklist kernel, in exact integers.

The Day 74 ``TiledMatmulTraffic`` counts the device loads of one T x T tile
with one output per thread. The Day 77 kernels change three things that it
cannot express: BM x BN tiles with K phases of BK, a TM x TN block of
outputs in the registers of each thread, and float4 staging.
``BlockTraffic`` counts them for one kernel and one shape:

- device loads: each group column reads all of A once and each group row
  reads all of B once, ``ceil(N / BN) * M * K + ceil(M / BM) * K * N``
  elements. A tile slot past M, N or K is zero-filled without a load, as in
  Day 74.
- device load operations: one for each element on the scalar path, one for
  each four elements on the float4 path. An operand takes the float4 path
  when K (for A) or N (for B) is a multiple of 4.
- device stores: each output once, ``M * N``.
- threadgroup stores: every tile slot of every phase, zero-filled or not.
- threadgroup reads: each launched thread reads TM values of A and TN values
  of B for each of its K steps, ``(TM + TN) / (TM * TN)`` reads for each fma.
- barriers: two for each phase, around ``BK * TM * TN`` fma steps of a thread.

The Day 74 kernels keep the no-reuse model of ``NaiveMatmulTraffic``:
``2 * M * N * K`` scalar loads and no threadgroup traffic. A kernel's
intensity is its FLOPs over the bytes of its device loads and stores, as an
exact fraction. The model sees no cache, no coalescing and no bank conflict,
and it counts the steps of padding threads that never store, so it is a
model and not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .kernel_metal_geometry import MAX_THREADGROUP_THREADS, ceil_div, require_integer
from .matmul_checklist_source import (
    MAX_BLOCK,
    MAX_BLOCK_K,
    MAX_THREAD_OUTPUTS,
    VECTOR_WIDTH,
    MatmulChecklistError,
)
from .matmul_checklist_variants import VARIANTS
from .matmul_metal_mapping import FLOAT32_BYTES
from .matmul_model_traffic import ratio_mapping
from .memory_access_tiling import MatmulShape, NaiveMatmulTraffic, TiledMatmulTraffic

NO_REUSE = "no_reuse"
BLOCKED = "blocked"
MODELS = (NO_REUSE, BLOCKED)
MODEL_TEXT = (
    "analytical counts with no cache, coalescing or bank model: no_reuse for a "
    "Day 74 kernel, 2 * M * N * K scalar loads and no threadgroup traffic; "
    "blocked for a BM x BN x BK tile with TM x TN outputs for each thread, "
    "ceil(N / BN) * M * K + ceil(M / BM) * K * N loads, one load operation for "
    "each element or for each float4, every tile slot stored in threadgroup "
    "memory and TM + TN threadgroup reads at each of the K steps of each "
    "launched thread; M * N stores and 4 bytes for each element in both models"
)


class MatmulChecklistBoundError(MatmulChecklistError):
    """Raised when bound inputs are unreadable or invalid, or do not pair."""


class MatmulChecklistBoundVerificationError(MatmulChecklistBoundError):
    """Raised when a timing or bound report, or one of its sources, fails checks."""


def _shape(shape: object) -> MatmulShape:
    if type(shape) is not MatmulShape:
        raise TypeError("shape must be a MatmulShape")
    return shape


@dataclass(frozen=True)
class NoReuseTraffic:
    """A Day 74 kernel: each thread loads its own row of A and column of B."""

    shape: MatmulShape

    def __post_init__(self) -> None:
        _shape(self.shape)

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def a_loads(self) -> int:
        return self.shape.m * self.shape.n * self.shape.k

    @property
    def b_loads(self) -> int:
        return self.shape.m * self.shape.n * self.shape.k

    @property
    def load_operations(self) -> int:
        """The Day 74 kernels load one float at a time."""

        return self.a_loads + self.b_loads

    @property
    def store_elements(self) -> int:
        return self.shape.m * self.shape.n

    @property
    def executed_fma(self) -> int:
        """The guard stops every padding thread before its loop."""

        return self.shape.m * self.shape.n * self.shape.k

    def to_mapping(self) -> dict[str, object]:
        """Return the counts with the fields of a blocked kernel; no tiles."""

        return {
            **_parameters(NO_REUSE, None, (1, 1), (False, False)),
            "tiles": None,
            "device": _device(self),
            "threadgroup": None,
            "executed_fma": self.executed_fma,
        }

    def day_74_check(self) -> bool:
        """Compare the counts with the Day 74 ``NaiveMatmulTraffic``."""

        naive = NaiveMatmulTraffic(self.shape, "float32")
        return (
            naive.load_elements == self.a_loads + self.b_loads
            and naive.store_elements == self.store_elements
            and naive.total_bytes == device_bytes(self)
            and naive.flops == self.flops
        )


@dataclass(frozen=True)
class BlockTraffic:
    """One BM x BN x BK block kernel with TM x TN outputs for each thread."""

    shape: MatmulShape
    block_m: int
    block_n: int
    block_k: int
    thread_m: int
    thread_n: int
    vector_loads: bool

    def __post_init__(self) -> None:
        _shape(self.shape)
        require_integer(self.block_m, "block_m", 1, MAX_BLOCK)
        require_integer(self.block_n, "block_n", 1, MAX_BLOCK)
        require_integer(self.block_k, "block_k", 1, MAX_BLOCK_K)
        require_integer(self.thread_m, "thread_m", 1, MAX_THREAD_OUTPUTS)
        require_integer(self.thread_n, "thread_n", 1, MAX_THREAD_OUTPUTS)
        if type(self.vector_loads) is not bool:
            raise TypeError("vector_loads must be a bool")
        if self.block_m % self.thread_m or self.block_n % self.thread_n:
            raise MatmulChecklistBoundError(
                "each tile side must be whole thread blocks"
            )
        if self.threads_per_group > MAX_THREADGROUP_THREADS:
            raise MatmulChecklistBoundError(
                f"a group of {self.threads_per_group} threads exceeds "
                f"{MAX_THREADGROUP_THREADS}"
            )
        # A float4 path stages whole float4 values of each tile row.
        if self.vector_loads and (
            self.block_k % VECTOR_WIDTH or self.block_n % VECTOR_WIDTH
        ):
            raise MatmulChecklistBoundError(
                "float4 staging needs BK and BN to be multiples of 4"
            )

    @property
    def flops(self) -> int:
        return self.shape.flops

    @property
    def block_rows(self) -> int:
        return ceil_div(self.shape.m, self.block_m)

    @property
    def block_columns(self) -> int:
        return ceil_div(self.shape.n, self.block_n)

    @property
    def groups(self) -> int:
        return self.block_rows * self.block_columns

    @property
    def phases(self) -> int:
        return ceil_div(self.shape.k, self.block_k)

    @property
    def threads_per_group(self) -> int:
        return (self.block_m // self.thread_m) * (self.block_n // self.thread_n)

    @property
    def launched_threads(self) -> int:
        return self.groups * self.threads_per_group

    @property
    def vector_a(self) -> bool:
        """A takes the float4 path when K is a multiple of 4."""

        return self.vector_loads and self.shape.k % VECTOR_WIDTH == 0

    @property
    def vector_b(self) -> bool:
        """B takes the float4 path when N is a multiple of 4."""

        return self.vector_loads and self.shape.n % VECTOR_WIDTH == 0

    @property
    def a_loads(self) -> int:
        """Every group column reads all of A once."""

        return self.block_columns * self.shape.m * self.shape.k

    @property
    def b_loads(self) -> int:
        """Every group row reads all of B once."""

        return self.block_rows * self.shape.k * self.shape.n

    @property
    def a_slots(self) -> int:
        return self.block_m * self.block_k * self.phases * self.groups

    @property
    def b_slots(self) -> int:
        return self.block_k * self.block_n * self.phases * self.groups

    @property
    def padded_slots(self) -> int:
        """Tile slots that hold a zero and cost no device load."""

        return self.a_slots + self.b_slots - self.a_loads - self.b_loads

    @property
    def load_operations(self) -> int:
        """On a float4 path, each float4 lies wholly inside or past the matrix.

        K or N is then a multiple of 4, and so is each tile edge, so a float4
        load always moves four useful elements.
        """

        a = self.a_loads // VECTOR_WIDTH if self.vector_a else self.a_loads
        b = self.b_loads // VECTOR_WIDTH if self.vector_b else self.b_loads
        return a + b

    @property
    def store_elements(self) -> int:
        return self.shape.m * self.shape.n

    @property
    def threadgroup_store_operations(self) -> int:
        a = self.a_slots // VECTOR_WIDTH if self.vector_a else self.a_slots
        b = self.b_slots // VECTOR_WIDTH if self.vector_b else self.b_slots
        return a + b

    @property
    def threadgroup_read_elements(self) -> int:
        """Each launched thread reads TM + TN values at each of its K steps."""

        return self.launched_threads * self.shape.k * (self.thread_m + self.thread_n)

    @property
    def executed_fma(self) -> int:
        """Padding threads run the k loop too; only their stores are guarded."""

        return self.groups * self.block_m * self.block_n * self.shape.k

    @property
    def reads_per_fma(self) -> Fraction:
        """Threadgroup reads for each executed fma: ``(TM + TN) / (TM * TN)``."""

        return Fraction(self.threadgroup_read_elements, self.executed_fma)

    def to_mapping(self) -> dict[str, object]:
        """Return every count, ratio and parameter of the model."""

        return {
            **_parameters(
                BLOCKED,
                [self.block_m, self.block_n, self.block_k],
                (self.thread_m, self.thread_n),
                (self.vector_a, self.vector_b),
            ),
            "tiles": {
                "groups": self.groups,
                "phases": self.phases,
                "launched_threads": self.launched_threads,
                "padded_slots": self.padded_slots,
            },
            "device": _device(self),
            "threadgroup": {
                "store_elements": self.a_slots + self.b_slots,
                "store_operations": self.threadgroup_store_operations,
                "read_elements": self.threadgroup_read_elements,
                "reads_per_fma": ratio_mapping(self.reads_per_fma),
                "barriers_per_thread": 2 * self.phases,
                "fma_per_thread_per_phase": (
                    self.block_k * self.thread_m * self.thread_n
                ),
            },
            "executed_fma": self.executed_fma,
        }

    def day_74_check(self) -> bool:
        """Compare a square tile of one output each with ``TiledMatmulTraffic``."""

        tile = self.block_m
        if (self.block_n, self.block_k, self.thread_m, self.thread_n) != (
            tile,
            tile,
            1,
            1,
        ):
            raise MatmulChecklistBoundError(
                "only a T x T x T tile with one output for each thread has a "
                "Day 74 count"
            )
        tiled = TiledMatmulTraffic(self.shape, tile, "float32")
        return (
            tiled.useful_a_loads == self.a_loads
            and tiled.useful_b_loads == self.b_loads
            and tiled.tile_slot_loads == self.a_slots + self.b_slots
            and tiled.padded_slots == self.padded_slots
            and tiled.total_bytes == device_bytes(self)
            and tiled.blocks == self.groups
            and tiled.phases == self.phases
            and tiled.threads_per_block == self.threads_per_group
        )


def device_bytes(model: NoReuseTraffic | BlockTraffic) -> int:
    """Return the bytes of the device loads and stores, 4 for each element."""

    return FLOAT32_BYTES * (model.a_loads + model.b_loads + model.store_elements)


def intensity(model: NoReuseTraffic | BlockTraffic) -> Fraction:
    """Return the FLOPs for each byte of the device loads and stores."""

    return Fraction(model.flops, device_bytes(model))


def _parameters(
    name: str,
    block: list[int] | None,
    threads: tuple[int, int],
    paths: tuple[bool, bool],
) -> dict[str, object]:
    return {
        "model": name,
        "block": block,
        "outputs_per_thread": list(threads),
        "vector_paths": {"a": paths[0], "b": paths[1]},
    }


def _device(model: NoReuseTraffic | BlockTraffic) -> dict[str, object]:
    shape = model.shape
    loads = model.a_loads + model.b_loads
    return {
        "a_loads": model.a_loads,
        "b_loads": model.b_loads,
        "load_operations": model.load_operations,
        "store_elements": model.store_elements,
        "load_bytes": FLOAT32_BYTES * loads,
        "store_bytes": FLOAT32_BYTES * model.store_elements,
        "bytes": device_bytes(model),
        "loads_per_output": ratio_mapping(Fraction(loads, shape.m * shape.n)),
        "flops_per_byte": ratio_mapping(intensity(model)),
    }


def kernel_traffic(name: object, shape: object) -> NoReuseTraffic | BlockTraffic:
    """Return the model of one registered kernel at one shape.

    The Day 74 kernels have no tile. The Day 76 tile is a T x T x T block
    with one output for each thread and scalar staging. A Day 77 kernel
    takes every size from its ``BlockDesign``.
    """

    if name not in VARIANTS:
        raise MatmulChecklistBoundError(f"{name!r} is not a registered kernel")
    checked = _shape(shape)
    variant = VARIANTS[name]
    if variant.group_size is None:
        return NoReuseTraffic(checked)
    design = variant.design
    if design is None:
        tile = variant.group_size[0]
        return BlockTraffic(checked, tile, tile, tile, 1, 1, False)
    return BlockTraffic(
        checked,
        design.block_m,
        design.block_n,
        design.block_k,
        design.thread_m,
        design.thread_n,
        design.vector_loads,
    )
