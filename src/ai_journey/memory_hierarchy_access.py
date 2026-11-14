"""Access-pattern contrast: one source array read into one output in four orders.

``copy`` streams the source with ``numpy.copyto``. ``sequential``,
``strided`` and ``random`` gather it with ``ndarray.take`` through an int64
index array that visits every source element exactly once: in order, in
stride-``S`` passes (elements ``j, j + S, j + 2S, ...`` for ``j = 0 .. S -
1``), or in a seeded permutation. The three gathers therefore move the same
accounted bytes (one index, one source element and one output element per
element) and differ only in the order of their source reads. A slower
gather pays for its order alone: the stride gives up spatial locality, and
the permutation also gives up predictability, so its misses overlap only as
far as the core keeps independent loads in flight.

``take`` runs with ``mode="clip"`` because with the default ``"raise"`` and
an ``out`` argument NumPy buffers the whole output in a temporary copy.
Every index is in range, so clipping never changes a value. Like the STREAM
kernels, the accounting counts operand bytes, not cache lines or hardware
traffic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy

from .memory_hierarchy_stream import (
    CHECK_CHUNK_ELEMENTS,
    MAX_SEED,
    MemoryHierarchyError,
    element_bytes,
    require_integer,
)
from .memory_hierarchy_timing import (
    BatchTiming,
    RunBudget,
    calls_per_sample,
    per_nanosecond,
    time_batches,
)

INDEX_BYTES = 8
MIN_ACCESS_ELEMENTS = 1 << 10
MAX_ACCESS_ELEMENTS = 1 << 24
MAX_STRIDE_ELEMENTS = 1 << 12


@dataclass(frozen=True)
class AccessPattern:
    """One way of reading the source: its call, visiting order and index use."""

    name: str
    call: str
    order: str
    indexed: bool

    def bytes_per_call(self, elements: int, dtype: str) -> int:
        """Return the operand bytes one call reads plus writes.

        Each element costs one source read and one output write, plus one
        8-byte index read for a gather; equal totals across the gathers are
        what isolate the effect of order.
        """

        count = require_integer(elements, "elements", 1, MAX_ACCESS_ELEMENTS)
        index = INDEX_BYTES if self.indexed else 0
        return count * (index + 2 * element_bytes(dtype))

    def to_mapping(self) -> dict[str, object]:
        """Return the pattern's call and order as recorded in a report."""

        return {
            "name": self.name,
            "call": self.call,
            "order": self.order,
            "index_bytes": INDEX_BYTES if self.indexed else 0,
        }


def _gather(order: str) -> str:
    return f"source.take({order}, out=output, mode='clip')"


ACCESS_PATTERNS = {
    pattern.name: pattern
    for pattern in (
        AccessPattern(
            "copy",
            "numpy.copyto(output, source)",
            "contiguous, no index array",
            False,
        ),
        AccessPattern(
            "sequential",
            _gather("sequential"),
            "element i at position i",
            True,
        ),
        AccessPattern(
            "strided",
            _gather("strided"),
            "stride passes: elements j, j + S, j + 2S, ... for j = 0 .. S - 1",
            True,
        ),
        AccessPattern(
            "random",
            _gather("random"),
            "numpy.random.default_rng((seed, 1)).permutation(n)",
            True,
        ),
    )
}
PATTERN_NAMES = tuple(ACCESS_PATTERNS)


def validate_access(elements: object, stride_elements: object) -> tuple[int, int]:
    """Require a source that splits into whole stride passes of two or more reads.

    Whole passes let the strided order visit every element exactly once, so
    all three gathers read identical bytes.
    """

    count = require_integer(
        elements, "access elements", MIN_ACCESS_ELEMENTS, MAX_ACCESS_ELEMENTS
    )
    stride = require_integer(stride_elements, "stride_elements", 2, MAX_STRIDE_ELEMENTS)
    if count % stride or count // stride < 2:
        raise MemoryHierarchyError(
            "access elements must be a multiple of stride_elements with at "
            "least two reads per pass"
        )
    return count, stride


class AccessBuffers:
    """A seeded source, an output of the same size, and the gather orders."""

    def __init__(
        self, elements: int, stride_elements: int, dtype: str, seed: int
    ) -> None:
        if numpy.dtype(numpy.intp).itemsize != INDEX_BYTES:
            raise MemoryHierarchyError("gather accounting assumes 8-byte numpy.intp")
        self.elements, self.stride_elements = validate_access(elements, stride_elements)
        element_bytes(dtype)
        self.seed = require_integer(seed, "seed", 0, MAX_SEED)
        kind = numpy.dtype(dtype)
        generator = numpy.random.default_rng(self.seed)
        self.source = generator.standard_normal(self.elements, dtype=kind)
        self.output = numpy.full(self.elements, numpy.nan, dtype=kind)

    def indices(self, pattern: str) -> numpy.ndarray | None:
        """Build one pattern's int64 visiting order; ``copy`` needs none.

        Orders are built on demand, one at a time, so a run holds at most one
        index array next to the source and output.
        """

        name = access_pattern(pattern).name
        count, stride = self.elements, self.stride_elements
        if name == "copy":
            return None
        if name == "sequential":
            return numpy.arange(count, dtype=numpy.int64)
        if name == "strided":
            # Broadcasting j + S * i allocates only the order itself; a
            # transposed arange would hold two full index arrays at once.
            passes = numpy.arange(stride, dtype=numpy.int64)[:, None]
            offsets = stride * numpy.arange(count // stride, dtype=numpy.int64)
            return (passes + offsets).reshape(-1)
        order = numpy.random.default_rng((self.seed, 1)).permutation(count)
        return order.astype(numpy.int64, copy=False)

    def bind(self, pattern: str, indices: numpy.ndarray | None) -> Callable[[], object]:
        """Return a no-argument callable reading the source in one order."""

        if access_pattern(pattern).indexed:
            self._require_order(indices)
            return partial(self.source.take, indices, None, self.output, "clip")
        if indices is not None:
            raise MemoryHierarchyError("copy reads the source without an index array")
        return partial(numpy.copyto, self.output, self.source)

    def _require_order(self, indices: object) -> None:
        if (
            not isinstance(indices, numpy.ndarray)
            or indices.dtype != numpy.int64
            or indices.shape != (self.elements,)
            or not indices.flags.c_contiguous
        ):
            raise MemoryHierarchyError(
                "a gather needs one contiguous int64 index per source element"
            )

    def reset_output(self) -> None:
        """Fill the output with NaN so a read that skips elements fails its check."""

        self.output[:] = numpy.nan

    def check(self, pattern: str, indices: numpy.ndarray | None) -> None:
        """Require the output to hold the source in the pattern's order.

        Fancy indexing builds the expected values independently of the timed
        call, chunk by chunk to keep temporaries small.
        """

        indexed = access_pattern(pattern).indexed
        if indexed:
            self._require_order(indices)
        for start in range(0, self.elements, CHECK_CHUNK_ELEMENTS):
            stop = min(self.elements, start + CHECK_CHUNK_ELEMENTS)
            want = (
                self.source[indices[start:stop]] if indexed else self.source[start:stop]
            )
            if not numpy.array_equal(self.output[start:stop], want):
                raise MemoryHierarchyError(
                    f"{pattern} output does not hold the source in its order"
                )


def access_pattern(name: object) -> AccessPattern:
    """Look up an access pattern by name, rejecting unknown names."""

    if not isinstance(name, str) or name not in ACCESS_PATTERNS:
        raise MemoryHierarchyError(f"pattern must be one of {', '.join(PATTERN_NAMES)}")
    return ACCESS_PATTERNS[name]


@dataclass(frozen=True)
class AccessMeasurement:
    """One access pattern timed over the whole source, with its accounting."""

    pattern: str
    elements: int
    dtype: str
    timing: BatchTiming

    @property
    def bytes_per_sample(self) -> int:
        """Accounted bytes moved by all calls of one timed sample."""

        per_call = access_pattern(self.pattern).bytes_per_call(
            self.elements, self.dtype
        )
        return per_call * self.timing.calls_per_sample

    @property
    def elements_per_sample(self) -> int:
        """Source elements read by all calls of one timed sample."""

        return self.elements * self.timing.calls_per_sample

    @property
    def median_ns_per_element(self) -> float:
        """Median sample time per source element read, the contrast's unit."""

        return self.timing.summary.median_ns / self.elements_per_sample

    @property
    def median_elements_per_ns(self) -> float:
        """Source elements read per nanosecond at the median sample."""

        return per_nanosecond(self.elements_per_sample, self.timing.summary.median_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples with every rate derived from them."""

        summary = self.timing.summary
        return {
            **access_pattern(self.pattern).to_mapping(),
            "bytes_per_call": access_pattern(self.pattern).bytes_per_call(
                self.elements, self.dtype
            ),
            **self.timing.to_mapping(),
            "bytes_per_sample": self.bytes_per_sample,
            "elements_per_sample": self.elements_per_sample,
            "median_ns_per_element": self.median_ns_per_element,
            "median_elements_per_ns": self.median_elements_per_ns,
            "median_gb_per_second": per_nanosecond(
                self.bytes_per_sample, summary.median_ns
            ),
            "best_gb_per_second": per_nanosecond(self.bytes_per_sample, summary.min_ns),
        }


def measure_access_patterns(
    *,
    elements: int,
    stride_elements: int,
    dtype: str,
    seed: int,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[AccessMeasurement, ...]:
    """Time and check copy, sequential, strided and random reads of one source.

    Each pattern's index array is built before its own measurement and
    released after it, outside every timed region.
    """

    buffers = AccessBuffers(elements, stride_elements, dtype, seed)
    results = []
    for name in PATTERN_NAMES:
        indices = buffers.indices(name)
        calls = calls_per_sample(
            access_pattern(name).bytes_per_call(buffers.elements, dtype),
            target_bytes_per_sample,
        )
        buffers.reset_output()
        timing = time_batches(
            buffers.bind(name, indices),
            calls_per_sample=calls,
            warmup=warmup,
            repeats=repeats,
            clock=clock,
            budget=budget,
        )
        buffers.check(name, indices)
        results.append(AccessMeasurement(name, buffers.elements, dtype, timing))
        del indices
    return tuple(results)


def access_contrast(
    measurements: tuple[AccessMeasurement, ...],
) -> dict[str, float]:
    """Express each gather's median time per element relative to sequential.

    Ratios of the same accounted bytes isolate the cost of order; the copy
    ratio shows how far the sequential gather sits from streaming bandwidth.
    """

    by_name = {item.pattern: item for item in measurements}
    if tuple(by_name) != PATTERN_NAMES or len(measurements) != len(PATTERN_NAMES):
        raise MemoryHierarchyError("the contrast needs every pattern once, in order")
    sequential = by_name["sequential"].median_ns_per_element
    return {
        "sequential_to_copy_time_ratio": sequential
        / by_name["copy"].median_ns_per_element,
        "strided_to_sequential_time_ratio": by_name["strided"].median_ns_per_element
        / sequential,
        "random_to_sequential_time_ratio": by_name["random"].median_ns_per_element
        / sequential,
        "random_to_strided_time_ratio": by_name["random"].median_ns_per_element
        / by_name["strided"].median_ns_per_element,
    }
