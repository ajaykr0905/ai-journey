"""Run both Day 78 CUDA reduction schedules on the CPU in numpy float32.

``simulate`` holds every block of pass 1 at once and follows the steps of a
``gpu_reduce_cuda`` text: the guarded loads, one tree level for each barrier
interval and, in the warp-shuffle kernel, the five shuffles of warp 0. Then
it runs the one block of the combine: the grid-stride loop with four
accumulators, the join ``(a0 + a1) + (a2 + a3)`` and the tree of 1024
values. numpy adds two float32 values with one rounding to nearest even,
the rounding of ``__fadd_rn``, and each step adds the pairs of the text in
its order. Thus each result has the bits of the order of the text.

``__shfl_down_sync`` gives lane l the value of lane l + d. The upper d lanes
get their own value back, as the CUDA C++ Programming Guide states, and the
sum of lane 0 uses none of those values.

The same steps also run on symbolic values. ``summation_tree`` returns the
tree of additions of one result: an element index for each loaded value,
``None`` for each ``0.0f`` and a pair for each addition. A test compares
these trees with the declared orders of the Metal twins, not only the bits.

An access log follows the shared memory of one block, from barrier to
barrier. A read of a slot that another thread wrote in the same interval, or
a write of a slot that another thread read in it, is a hazard. The shared
indices do not depend on ``blockIdx``, so the log of one block stands for
every block. A run can drop barriers, and the log then shows what each
barrier prevents. The simulator runs each step for all threads before the
next step, so its values are those of one legal order, and a dropped barrier
changes no value.

``predict`` runs both kernels on the inputs of a Day 78 config and checks
each result against the bound of its order: the depth of the Metal twin,
because the order is the same. This module runs on the CPU only. It
compiles and runs no CUDA, and its counts are a model of the texts, not
measurements of a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import numpy

from .gpu_checklist_blocksim import _AccessLog
from .gpu_reduce_bound import (
    ReductionCheck,
    benchmark_values,
    case_values,
    exact_sums,
    prefix_sums,
    summation_depth,
)
from .gpu_reduce_config import GpuReduceConfig
from .gpu_reduce_cuda import (
    COMBINE_ACCUMULATORS,
    COMBINE_BARRIERS,
    COMBINE_SPAN,
    COMBINE_STRIDES,
    COMBINE_THREADS,
    PINNED_KERNELS,
    SHUFFLE_OFFSETS,
    CudaReduceKernel,
    GpuReduceCudaError,
    require_kernel,
    require_n,
)
from .gpu_tiled_blocksim import SharedHazards
from .kernel_launch_config import WARP_SIZE, LaunchConfig
from .kernel_launch_io import canonical_json_bytes

MAX_TREE_ELEMENTS = 1 << 16
MAX_TREE_PARTIALS = 1 << 15
TIMING_PATTERN = "timing"
_CHUNK_BLOCKS = 4096


class _Term:
    """A symbolic value: the sum of two terms is the pair of their trees."""

    __slots__ = ("tree",)

    def __init__(self, tree: object) -> None:
        self.tree = tree

    def __add__(self, other: _Term) -> _Term:
        return _Term((self.tree, other.tree))


_ZERO = _Term(None)


def _zeros(shape: int | tuple[int, ...], like: numpy.ndarray) -> numpy.ndarray:
    if like.dtype == object:
        return numpy.full(shape, _ZERO, dtype=object)
    return numpy.zeros(shape, dtype=numpy.float32)


def _zero(like: numpy.ndarray) -> object:
    return _ZERO if like.dtype == object else numpy.float32(0)


def _reads(
    threads: int, slots: list[tuple[numpy.ndarray, numpy.ndarray]]
) -> numpy.ndarray:
    """Return the [thread, slot] reads of one step: each (threads, slots) pair."""

    reads = numpy.zeros((threads, threads), dtype=bool)
    for thread, slot in slots:
        reads[thread, slot] = True
    return reads


def _writes(threads: int, writers: numpy.ndarray) -> numpy.ndarray:
    """Return the writer of each slot: thread t writes slot t."""

    slots = numpy.full(threads, -1)
    slots[writers] = writers
    return slots


def _level(log: _AccessLog, threads: int, stride: int) -> None:
    """Log one tree level: each thread t < s reads t and t + s and writes t."""

    active = numpy.arange(stride)
    log.read(_reads(threads, [(active, active), (active, active + stride)]))
    log.write(_writes(threads, active))


def _require_dropped(kernel: CudaReduceKernel, dropped: object) -> tuple[int, ...]:
    if (
        type(dropped) is not tuple
        or any(type(item) is not int for item in dropped)
        or list(dropped) != sorted(set(dropped))
        or any(not 0 <= item < kernel.barriers_per_block for item in dropped)
    ):
        raise GpuReduceCudaError(
            "dropped_barriers must be sorted distinct barrier numbers from 0 to "
            f"{kernel.barriers_per_block - 1}"
        )
    return dropped


def _pass_one_log(kernel: CudaReduceKernel, dropped: tuple[int, ...]) -> _AccessLog:
    """Log the shared accesses of one pass-1 block, step by step."""

    threads = kernel.block_threads
    thread = numpy.arange(threads)
    log = _AccessLog(threads, threads)
    log.write(_writes(threads, thread))
    barriers = iter(range(kernel.barriers_per_block))
    if next(barriers) not in dropped:
        log.barrier()
    for stride in kernel.tree_strides:
        _level(log, threads, stride)
        if next(barriers) not in dropped:
            log.barrier()
    # Lane l of warp 0 reads slot l, or thread 0 reads slot 0 for its store.
    readers = numpy.arange(WARP_SIZE if kernel.warp_shuffle else 1)
    log.read(_reads(threads, [(readers, readers)]))
    return log


def _pass_one(
    kernel: CudaReduceKernel, values: numpy.ndarray
) -> tuple[numpy.ndarray, int]:
    """Run every block of pass 1; return the partials and the loads inside n.

    The blocks run in chunks, so a 2^26-element input needs little memory.
    """

    n = values.size
    blocks, threads = kernel.launch(n).grid.x, kernel.block_threads
    thread, lanes = numpy.arange(threads), numpy.arange(WARP_SIZE)
    partials, loads = [], 0
    for start in range(0, blocks, _CHUNK_BLOCKS):
        chunk = numpy.arange(start, min(start + _CHUNK_BLOCKS, blocks))
        first = chunk[:, None] * kernel.elements_per_block + thread
        loaded = []
        for offset in range(0, kernel.elements_per_block, threads):
            index = first + offset
            inside = index < n
            loads += int(numpy.count_nonzero(inside))
            loaded.append(
                numpy.where(inside, values[numpy.minimum(index, n - 1)], _zero(values))
            )
        # One load, or __fadd_rn of the two loads 256 apart.
        shared = loaded[0] if len(loaded) == 1 else loaded[0] + loaded[1]
        for stride in kernel.tree_strides:
            shared[:, :stride] = shared[:, :stride] + shared[:, stride : 2 * stride]
        if kernel.warp_shuffle:
            warp = shared[:, :WARP_SIZE].copy()
            for offset in SHUFFLE_OFFSETS:
                source = numpy.where(lanes + offset < WARP_SIZE, lanes + offset, lanes)
                warp = warp + warp[:, source]
            shared = warp
        partials.append(shared[:, 0])
    return numpy.concatenate(partials), loads


def _combine(partials: numpy.ndarray) -> tuple[object, int, _AccessLog]:
    """Run the one block of the combine; return out[0], the reads and the log."""

    count, threads = partials.size, COMBINE_THREADS
    thread = numpy.arange(threads)
    accumulators = [_zeros(threads, partials) for _ in range(COMBINE_ACCUMULATORS)]
    index, reads = thread.copy(), 0
    while (active := index + (COMBINE_ACCUMULATORS - 1) * threads < count).any():
        for slot, accumulator in enumerate(accumulators):
            source = index[active] + slot * threads
            accumulator[active] = accumulator[active] + partials[source]
            reads += source.size
        index[active] += COMBINE_SPAN
    while (active := index < count).any():
        first = accumulators[0]
        first[active] = first[active] + partials[index[active]]
        reads += int(numpy.count_nonzero(active))
        index[active] += threads
    a0, a1, a2, a3 = accumulators
    shared = (a0 + a1) + (a2 + a3)
    log = _AccessLog(threads, threads)
    log.write(_writes(threads, thread))
    log.barrier()
    for stride in COMBINE_STRIDES:
        _level(log, threads, stride)
        shared[:stride] = shared[:stride] + shared[stride : 2 * stride]
        log.barrier()
    log.read(_reads(threads, [(thread[:1], thread[:1])]))
    return shared[0], reads, log


@dataclass(frozen=True)
class CudaReduceRun:
    """One simulated call: both launches, their counts, hazards and result."""

    kernel: CudaReduceKernel
    n: int
    launch: LaunchConfig
    dropped_barriers: tuple[int, ...]
    loads: int
    zero_fills: int
    partial_writes: int
    combine_reads: int
    hazards: SharedHazards
    combine_hazards: SharedHazards
    output: float
    partials: numpy.ndarray = field(repr=False, compare=False)

    @property
    def output_bits(self) -> str:
        return f"0x{int(numpy.float32(self.output).view(numpy.uint32)):08x}"

    @property
    def matches_model(self) -> bool:
        """Return whether every count is the closed form of the text.

        The loads read each element once, the zero fills cover the rest of the
        grid, each block writes one partial, the combine reads each partial
        once, and each thread reaches the barriers that the run kept.
        """

        blocks = self.launch.grid.x
        kept = self.kernel.barriers_per_block - len(self.dropped_barriers)
        return (
            self.loads == self.n
            and self.zero_fills == blocks * self.kernel.elements_per_block - self.n
            and self.partial_writes == blocks == self.combine_reads
            and self.hazards.barriers == kept
            and self.combine_hazards.barriers == COMBINE_BARRIERS
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel.name,
            "n": self.n,
            "launch": self.launch.to_mapping(),
            "dropped_barriers": list(self.dropped_barriers),
            "loads": self.loads,
            "zero_fills": self.zero_fills,
            "partial_writes": self.partial_writes,
            "combine_reads": self.combine_reads,
            "hazards": self.hazards.to_mapping(),
            "combine_hazards": self.combine_hazards.to_mapping(),
            "matches_model": self.matches_model,
            "output": self.output,
            "output_bits": self.output_bits,
        }


def _require_values(values: object) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
    ):
        raise GpuReduceCudaError("values must be a 1D float32 array")
    require_n(values.size)
    return values


def simulate(
    values: numpy.ndarray,
    *,
    kernel: CudaReduceKernel,
    dropped_barriers: tuple[int, ...] = (),
) -> CudaReduceRun:
    """Run one call of a kernel text on the CPU: pass 1, then the combine.

    ``dropped_barriers`` numbers the barriers of pass 1 in program order: 0
    after the load, then one for each tree level.
    """

    checked = require_kernel(kernel)
    array = _require_values(values)
    dropped = _require_dropped(checked, dropped_barriers)
    partials, loads = _pass_one(checked, array)
    output, reads, combine_log = _combine(partials)
    launch = checked.launch(array.size)
    return CudaReduceRun(
        kernel=checked,
        n=array.size,
        launch=launch,
        dropped_barriers=dropped,
        loads=loads,
        zero_fills=launch.grid.x * checked.elements_per_block - loads,
        partial_writes=partials.size,
        combine_reads=reads,
        hazards=_pass_one_log(checked, dropped).hazards(),
        combine_hazards=combine_log.hazards(),
        output=float(output),
        partials=partials,
    )


def _leaves(count: int) -> numpy.ndarray:
    leaves = numpy.empty(count, dtype=object)
    leaves[:] = [_Term(index) for index in range(count)]
    return leaves


def summation_tree(kernel: CudaReduceKernel, n: int) -> object:
    """Return the tree of additions of one call for n elements.

    A leaf is an element index, or ``None`` for a ``0.0f``. Each addition is
    the pair (left, right) of the two values that it adds.
    """

    checked = require_kernel(kernel)
    count = require_n(n)
    if count > MAX_TREE_ELEMENTS:
        raise GpuReduceCudaError(f"a tree holds at most {MAX_TREE_ELEMENTS} elements")
    partials, _ = _pass_one(checked, _leaves(count))
    output, _, _ = _combine(partials)
    return output.tree


def combine_tree(count: int) -> object:
    """Return the tree of additions of the combine for ``count`` partials."""

    if type(count) is not int:
        raise TypeError("count must be an integer")
    if not 1 <= count <= MAX_TREE_PARTIALS:
        raise GpuReduceCudaError(f"count must be between 1 and {MAX_TREE_PARTIALS}")
    output, _, _ = _combine(_leaves(count))
    return output.tree


@dataclass(frozen=True)
class Prediction:
    """The simulated result of one kernel on one input, against its bound."""

    kernel: str
    input: str
    pattern: str
    n: int
    run: CudaReduceRun = field(repr=False)
    check: ReductionCheck = field(repr=False)

    @property
    def passed(self) -> bool:
        return (
            self.check.within
            and self.run.matches_model
            and self.run.hazards.race_free
            and self.run.combine_hazards.race_free
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "input": self.input,
            "pattern": self.pattern,
            "n": self.n,
            "output": self.run.output,
            "output_bits": self.run.output_bits,
            "depth": self.check.depth,
            "bound_ratio": self.check.bound_ratio,
            "within_bound": self.check.within,
            "exact_result": self.check.error == 0,
            "race_free": self.run.hazards.race_free
            and self.run.combine_hazards.race_free,
            "matches_model": self.run.matches_model,
        }


def _predictions(label: str, pattern: str, values: numpy.ndarray, sums) -> list:
    found = []
    for kernel in PINNED_KERNELS:
        run = simulate(values, kernel=kernel)
        depth = summation_depth(kernel.metal_twin, values.size)
        check = ReductionCheck(run.output, depth, sums)
        found.append(Prediction(kernel.name, label, pattern, values.size, run, check))
    return found


def predict(config: GpuReduceConfig) -> tuple[Prediction, ...]:
    """Simulate both kernels on every input of a Day 78 config, in config order.

    The inputs are the correctness cases and then the prefixes of the one
    timing input, drawn from the seed as the Metal run draws them.
    """

    if not isinstance(config, GpuReduceConfig):
        raise TypeError("config must be GpuReduceConfig")
    found: list[Prediction] = []
    for index, case in enumerate(config.correctness.cases):
        values = case_values(config.seed, index, case.pattern, case.n)
        found += _predictions(f"case-{index}", case.pattern, values, exact_sums(values))
    sizes = config.benchmark.sizes
    timing = benchmark_values(config.seed, max(sizes))
    sums = prefix_sums(timing, sizes)
    for index, n in enumerate(sizes):
        found += _predictions(f"timing-{index}", TIMING_PATTERN, timing[:n], sums[n])
    return tuple(found)


def predictions_sha256(predictions: tuple[Prediction, ...]) -> str:
    """Bind every predicted result to one digest of its canonical JSON."""

    payload = [prediction.to_mapping() for prediction in predictions]
    return sha256(canonical_json_bytes(payload)).hexdigest()
