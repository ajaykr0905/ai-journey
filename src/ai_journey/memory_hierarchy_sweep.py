"""Working-set sweep: STREAM bandwidth from cache-sized to DRAM-sized arrays.

One ``StreamBuffers`` allocation sized for the largest point serves the whole
sweep; each point binds the kernels to the first ``elements`` entries of each
array. Points run smallest first and, at each point, every kernel's output is
reset to NaN, timed in byte-sized batches after untimed warmups, and checked
against its STREAM definition after the clock stops.

The curve plots each kernel's bandwidth against its working set, the bytes of
the distinct arrays one call touches. It names no cache level: where the
bandwidth falls is read from the data alone. At the smallest points NumPy's
per-call overhead rather than the memory system dominates each call, which
``median_ns_per_call`` exposes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .memory_hierarchy_stream import (
    KERNEL_NAMES,
    MAX_STREAM_ELEMENTS,
    MemoryHierarchyError,
    StreamBuffers,
    StreamKernel,
    require_integer,
    stream_kernel,
)
from .memory_hierarchy_timing import (
    BatchTiming,
    RunBudget,
    calls_per_sample,
    per_nanosecond,
    time_batches,
)

MIN_SWEEP_ELEMENTS = 64
MAX_SWEEP_POINTS = 16


def validate_sweep(
    kernels: object, elements: object
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Require known kernels in STREAM order and strictly increasing points.

    A fixed order makes the measurement list reproducible from the controls,
    and increasing points make a curve's last point its largest working set.
    """

    if type(kernels) is not tuple or not kernels:
        raise MemoryHierarchyError("kernels must be a nonempty tuple")
    for name in kernels:
        stream_kernel(name)
    if list(kernels) != [name for name in KERNEL_NAMES if name in kernels]:
        raise MemoryHierarchyError(
            "kernels must be unique and listed in STREAM order: "
            + ", ".join(KERNEL_NAMES)
        )
    if type(elements) is not tuple or not 1 <= len(elements) <= MAX_SWEEP_POINTS:
        raise MemoryHierarchyError(
            f"sweep elements must be a tuple of 1 to {MAX_SWEEP_POINTS} points"
        )
    for count in elements:
        require_integer(
            count, "sweep elements", MIN_SWEEP_ELEMENTS, MAX_STREAM_ELEMENTS
        )
    if any(later <= earlier for earlier, later in zip(elements, elements[1:])):
        raise MemoryHierarchyError("sweep elements must be strictly increasing")
    return kernels, elements


@dataclass(frozen=True)
class StreamMeasurement:
    """One kernel timed at one sweep point, with its exact byte accounting."""

    kernel: str
    elements: int
    dtype: str
    timing: BatchTiming

    @property
    def spec(self) -> StreamKernel:
        """The kernel definition whose accounting these samples are divided by."""

        return stream_kernel(self.kernel)

    @property
    def working_set_bytes(self) -> int:
        """Bytes of the distinct arrays one call touches: the curve's x-axis."""

        return self.spec.working_set_bytes(self.elements, self.dtype)

    @property
    def bytes_per_sample(self) -> int:
        """Accounted bytes moved by all calls of one timed sample."""

        bytes_per_call = self.spec.bytes_per_call(self.elements, self.dtype)
        return bytes_per_call * self.timing.calls_per_sample

    @property
    def flops_per_sample(self) -> int:
        """FLOPs performed by all calls of one timed sample."""

        return self.spec.flops_per_call(self.elements) * self.timing.calls_per_sample

    @property
    def median_gb_per_second(self) -> float:
        """Accounted bytes per nanosecond at the median sample (decimal GB/s)."""

        return per_nanosecond(self.bytes_per_sample, self.timing.summary.median_ns)

    @property
    def best_gb_per_second(self) -> float:
        """Accounted bytes per nanosecond at the fastest sample (decimal GB/s)."""

        return per_nanosecond(self.bytes_per_sample, self.timing.summary.min_ns)

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples with every rate derived from them."""

        median_ns = self.timing.summary.median_ns
        return {
            "kernel": self.kernel,
            "elements": self.elements,
            "working_set_bytes": self.working_set_bytes,
            "bytes_per_call": self.spec.bytes_per_call(self.elements, self.dtype),
            "flops_per_call": self.spec.flops_per_call(self.elements),
            **self.timing.to_mapping(),
            "bytes_per_sample": self.bytes_per_sample,
            "flops_per_sample": self.flops_per_sample,
            "median_ns_per_call": median_ns / self.timing.calls_per_sample,
            "median_gb_per_second": self.median_gb_per_second,
            "best_gb_per_second": self.best_gb_per_second,
            "median_gflops_per_second": per_nanosecond(
                self.flops_per_sample, median_ns
            ),
        }


def measure_stream_sweep(
    *,
    kernels: tuple[str, ...],
    elements: tuple[int, ...],
    dtype: str,
    seed: int,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[StreamMeasurement, ...]:
    """Time and check every kernel at every sweep point, smallest point first.

    The NaN reset before and the exact check after each measurement mean a
    kernel that skipped work fails the run instead of reporting a bandwidth.
    """

    kernels, elements = validate_sweep(kernels, elements)
    buffers = StreamBuffers(elements[-1], dtype, seed)
    results = []
    for count in elements:
        for name in kernels:
            calls = calls_per_sample(
                stream_kernel(name).bytes_per_call(count, dtype),
                target_bytes_per_sample,
            )
            buffers.reset_output(count)
            timing = time_batches(
                buffers.bind(name, count),
                calls_per_sample=calls,
                warmup=warmup,
                repeats=repeats,
                clock=clock,
                budget=budget,
            )
            buffers.check(name, count)
            results.append(StreamMeasurement(name, count, dtype, timing))
    return tuple(results)


def stream_curves(
    measurements: tuple[StreamMeasurement, ...],
) -> dict[str, list[dict[str, object]]]:
    """Group sweep results into one bandwidth curve per kernel, in sweep order."""

    curves: dict[str, list[dict[str, object]]] = {}
    for item in measurements:
        if not isinstance(item, StreamMeasurement):
            raise TypeError("measurements must be StreamMeasurement values")
        curves.setdefault(item.kernel, []).append(
            {
                "elements": item.elements,
                "working_set_bytes": item.working_set_bytes,
                "median_gb_per_second": item.median_gb_per_second,
                "best_gb_per_second": item.best_gb_per_second,
            }
        )
    return curves


def curve_summary(curve: list[dict[str, object]]) -> dict[str, object]:
    """Locate a curve's peak and its largest point without naming a cache level.

    The peak is the first point with the highest median bandwidth; the ratio
    shows how much of that bandwidth survives at the largest working set.
    """

    if type(curve) is not list or not curve:
        raise MemoryHierarchyError("a curve needs at least one point")
    peak = max(curve, key=lambda point: point["median_gb_per_second"])
    largest = curve[-1]
    return {
        "peak_working_set_bytes": peak["working_set_bytes"],
        "peak_median_gb_per_second": peak["median_gb_per_second"],
        "largest_working_set_bytes": largest["working_set_bytes"],
        "largest_median_gb_per_second": largest["median_gb_per_second"],
        "largest_to_peak_ratio": (
            largest["median_gb_per_second"] / peak["median_gb_per_second"]
        ),
    }
