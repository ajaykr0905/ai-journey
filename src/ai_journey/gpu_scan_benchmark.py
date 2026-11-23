"""Time every scan, ``torch.cumsum`` and the Day 74 copy in one rotated run.

Each point draws one seeded input, places it on the device once and binds
every side to it: the admitted kernels in registry order, then
``torch.cumsum(x, 0, out=y)`` into a preallocated output, then the Day 74
``stream_copy`` of a zero-padded copy of the input. The timing is the Day 74
to 77 method, run by the Day 74 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again. One call of a kernel side queues every
   dispatch of one scan.
3. Repeat r starts with side r mod S and runs each of the S sides once in
   order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``: 8 bytes for each element, one float32 read and
one float32 write, the least that any scan moves. Every side of a point uses
that batch. GB/s divides the bytes of a sample by its median time: 8 N for
each scan side and for ``torch.cumsum``, and the copy's own ``32 ceil(N /
4)`` bytes for the copy.

After the last sample, every side's latest output goes to the host. A kernel
must stay within its bound of the float64 reference, or the run stops, and
the copy must hold its input bit for bit. ``torch.cumsum`` is measured. Each
check also records whether the side holds the bits of S0.

Each point records the modeled device bytes of one scan: every level reads
and writes its elements and writes its block sums, and every add reads and
writes the elements after the first block and reads each earlier block sum
once. The three kernels share these bytes. A model value is not a
measurement. Only ``time.perf_counter_ns`` makes a measurement; any other
clock is named ``injected``. These are host-specific Apple GPU (Metal)
timings; nothing here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from .gpu_scan_config import BYTES_PER_ELEMENT, scan_calls_per_sample
from .gpu_scan_correctness import runnable
from .gpu_scan_plan import COPY, VENDOR
from .gpu_scan_reference import (
    BENCHMARK_STREAM,
    NORMAL,
    error_bound,
    float64_reference,
    max_relative_error,
    scan_inputs,
)
from .gpu_scan_runtime import (
    LevelLaunch,
    copy_float4,
    copy_launch,
    padded_copy_input,
    scan_plan,
)
from .gpu_scan_source import (
    BLOCK,
    FLOAT32_BYTES,
    S0_HILLIS_STEELE,
    GpuScanError,
    block_count,
    dispatches,
    level_lengths,
)
from .kernel_metal_benchmark import MAX_POINTS, DeviceTiming, RunBudget, _rates
from .kernel_metal_geometry import CheckedLaunch
from .memory_access_benchmark import timer_name
from .metal_launch_sweep_benchmark import time_shapes

TRAFFIC_MODEL = (
    "modeled device bytes of one scan with no cache: 4 * (2 L + B) for each "
    "level of L elements in B blocks, and 4 * (2 (L - min(L, 1024)) + B - 1) for "
    "the add below each level but the top; one float32 read and one float32 "
    "write of each element, 8 N bytes, is the compulsory part"
)
CHECK_RULE = (
    "every kernel output within its bound of the float64 reference, or the run "
    "stops; the copy equal to its input bit for bit; torch.cumsum measured; "
    "same_bits_as_s0 compares each output with the S0 output bit for bit"
)


def compulsory_bytes(length: int) -> int:
    """The bytes any scan must move: one float32 read and one write of each element."""

    return BYTES_PER_ELEMENT * length


def copy_bytes(length: int) -> int:
    """The bytes the copy reads and writes: two float4 values for each of its walks."""

    return 2 * 4 * FLOAT32_BYTES * copy_float4(length)


def modeled_bytes(length: int) -> int:
    """The device bytes of one scan in the no-cache model of ``TRAFFIC_MODEL``."""

    lengths = level_lengths(length)
    total = 0
    for level, count in enumerate(lengths):
        blocks = block_count(count)
        total += FLOAT32_BYTES * (2 * count + blocks)
        if level < len(lengths) - 1:
            total += FLOAT32_BYTES * (2 * (count - min(count, BLOCK)) + blocks - 1)
    return total


def _gb_rates(timing: DeviceTiming, bytes_per_call: int) -> dict[str, object]:
    """The Day 73 rates of one side, with the GB/s range from its IQR."""

    entry = _rates(timing, bytes_per_call)
    moved = entry["bytes_per_sample"]
    summary = timing.summary
    return {
        **entry,
        "iqr_gb_per_second": [moved / summary.p75_ns, moved / summary.p25_ns],
    }


@dataclass(frozen=True)
class ScanPoint:
    """One sweep point: every side timed on the same device input."""

    index: int
    length: int
    launches: Mapping[str, tuple[LevelLaunch, ...]]
    copy: CheckedLaunch
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, Mapping[str, object]]

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: the kernels, the vendor, the copy."""

        return tuple(timing.implementation for timing in self.timings)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics, rates, model and checks."""

        bytes_per_call = compulsory_bytes(self.length)
        return {
            "index": self.index,
            "length": self.length,
            "levels": list(level_lengths(self.length)),
            "dispatches": dispatches(self.length),
            "bytes_per_call": bytes_per_call,
            "copy_bytes_per_call": copy_bytes(self.length),
            "modeled_bytes_per_call": modeled_bytes(self.length),
            "side_order": list(self.sides),
            "launches": {
                name: [level.to_mapping() for level in levels]
                for name, levels in self.launches.items()
            },
            "copy_launch": self.copy.to_mapping(),
            "sides": {
                timing.implementation: _gb_rates(
                    timing,
                    copy_bytes(self.length)
                    if timing.implementation == COPY
                    else bytes_per_call,
                )
                for timing in self.timings
            },
            "checks": {name: dict(check) for name, check in self.checks.items()},
        }


def _same_bits(first: numpy.ndarray, second: numpy.ndarray) -> bool:
    return first.shape == second.shape and numpy.array_equal(
        first.view(numpy.uint32), second.view(numpy.uint32)
    )


def _checks(
    outputs: Mapping[str, numpy.ndarray],
    values: numpy.ndarray,
    names: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, object]]:
    """Check every side's latest output on the CPU after the timing."""

    length = values.size
    if set(outputs) != {*names, VENDOR, COPY}:
        raise GpuScanError(f"{label}: the workload must return every side")
    if not _same_bits(outputs[COPY], padded_copy_input(values)):
        raise GpuScanError(f"{label}: the copy reference changed its input")
    reference, scale = float64_reference(values)
    first = outputs.get(S0_HILLIS_STEELE)
    checks = {}
    for name in (*names, VENDOR):
        output = outputs[name]
        if (
            not isinstance(output, numpy.ndarray)
            or output.dtype != numpy.float32
            or output.shape != (length,)
        ):
            raise GpuScanError(f"{label}: {name} returned another output shape")
        error = max_relative_error(output, reference, scale)
        bound = None if name == VENDOR else error_bound(name, length)
        if not error < math.inf:
            raise GpuScanError(f"{label}: the timed output of {name} is not finite")
        if bound is not None and not error <= bound:
            raise GpuScanError(
                f"{label}: the timed output of {name} has relative error {error!r}, "
                f"above its bound {bound!r}"
            )
        checks[name] = {
            "max_relative_error": error,
            "bound": bound,
            "bound_ratio": None if bound is None else error / bound,
            "same_bits_as_s0": None if first is None else _same_bits(output, first),
        }
    checks[COPY] = {"same_bits_as_input": True}
    return checks


def benchmark_point(
    backend: object,
    *,
    seed: int,
    index: int,
    length: int,
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> ScanPoint:
    """Time every side of one point on one device input."""

    names = runnable(backend, names)
    values = scan_inputs(seed, BENCHMARK_STREAM, index, length, NORMAL)
    launches = {name: scan_plan(name, backend.limits, length) for name in names}
    copy = copy_launch(backend.limits, length)
    workload = backend.workload(values, names)
    sides = tuple(workload.sides)
    if sides != (*names, VENDOR, COPY):
        raise GpuScanError(
            "a workload must bind the kernels, then torch.cumsum, then the copy"
        )
    calls = scan_calls_per_sample(length, target_bytes_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    checks = _checks(workload.check(), values, names, f"point {index} ({length})")
    timer = timer_name(clock)
    timings = tuple(
        DeviceTiming(name, calls, warmup, repeats, series, timer)
        for name, series in zip(sides, samples)
    )
    return ScanPoint(index, length, launches, copy, timings, checks)


def benchmark_sweep(
    backend: object,
    *,
    seed: int,
    lengths: tuple[int, ...],
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[ScanPoint, ...]:
    """Time every side at every sweep length; ``names`` are the admitted kernels."""

    if type(lengths) is not tuple or not 1 <= len(lengths) <= MAX_POINTS:
        raise GpuScanError(f"a sweep holds 1 to {MAX_POINTS} lengths")
    return tuple(
        benchmark_point(
            backend,
            seed=seed,
            index=index,
            length=length,
            names=names,
            warmup=warmup,
            repeats=repeats,
            target_bytes_per_sample=target_bytes_per_sample,
            clock=clock,
            budget=budget,
        )
        for index, length in enumerate(lengths)
    )
