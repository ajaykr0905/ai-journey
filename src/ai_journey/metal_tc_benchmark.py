"""Time the seven sides of one size in one rotated run.

Each point draws one pair of seeded square operands and places them on the
device once in each input type that a supported side reads. It binds the
seven sides in plan order:
S1, M1, M2, M3, and ``torch.matmul`` in float32, float16 and bfloat16. The
timing is the Day 74 and Day 76 method, run by the Day 78 sweep timer
``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again.
3. Repeat r starts with side r mod 7 and runs each of the seven sides once
   in order, so the order shifts by one side at every repeat.

An unsupported side keeps its turn as a call that does nothing, and its
samples are dropped, so each supported side still takes each position
equally often. A sample holds the fewest whole calls whose ``2 n**3``
operations reach ``target_flops_per_sample``, the same count for every
side, so a ratio of median times compares equal work. After the last
sample, each kernel's latest output must be within its declared bound of
the float64 reference at the checked rows, and each PyTorch output within
its sanity bound, or the run stops.

Only the host timer ``time.perf_counter_ns`` makes a measurement; any
other clock is named ``injected``. These are host-specific Apple GPU
(Metal) timings; nothing here measures CUDA, NVIDIA hardware or Tensor
Cores.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from .kernel_metal_benchmark import DeviceTiming, RunBudget
from .memory_access_benchmark import timer_name
from .metal_launch_sweep_benchmark import matmul_flops, time_shapes
from .metal_tc_config import TcPoint, calls_per_sample
from .metal_tc_correctness import compare
from .metal_tc_plan import SIDE_PRECISION, SIDES, TORCH_FLOAT32
from .metal_tc_reference import BENCHMARK_STREAM, operands, tc_reference
from .metal_tc_runtime import launch_mapping
from .metal_tc_source import FLOAT32_BYTES, INPUT_BYTES, M1, S1, VARIANTS, MetalTcError

ARITHMETIC = (
    "2 n**3 floating-point operations for each call of size n: one multiply and "
    "one add for each of the n**3 products"
)
COMPULSORY = (
    "(2 e + c) n**2 bytes for each call: read A and B once at e bytes for each "
    "input value and write C once at c bytes, with c = 4 for a kernel and c = e "
    "for a PyTorch call, whose output has the input type"
)


def compulsory_bytes(side: str, size: int) -> int:
    """Bytes that one call of a side must move at least, by ``COMPULSORY``."""

    if side not in SIDES:
        raise MetalTcError(f"{side!r} is not a timed side")
    element = INPUT_BYTES[SIDE_PRECISION[side]]
    output = FLOAT32_BYTES if side in VARIANTS else element
    return (2 * element + output) * size * size


def side_rates(timing: DeviceTiming, flops: int, compulsory: int) -> dict:
    """Return the raw samples with their statistics, TFLOP/s and GB/s."""

    summary = timing.summary
    calls = timing.calls_per_sample
    work, moved = flops * calls, compulsory * calls
    return {
        **timing.to_mapping(),
        "median_ns_per_call": summary.median_ns / calls,
        "iqr_ns_per_call": summary.iqr_ns / calls,
        "flops_per_call": flops,
        "compulsory_bytes_per_call": compulsory,
        "tflops_per_second": work / summary.median_ns / 1000,
        "iqr_tflops_per_second": [
            work / summary.p75_ns / 1000,
            work / summary.p25_ns / 1000,
        ],
        "effective_gb_per_second": moved / summary.median_ns,
        "intensity_flops_per_byte": flops / compulsory,
    }


@dataclass(frozen=True)
class SideCheck:
    """How one side's latest output compares with the float64 reference."""

    max_abs_error: float
    max_bound_ratio: float
    max_relative_error: float

    def to_mapping(self) -> dict[str, object]:
        """Return the check as a report records it."""

        return {
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "max_relative_error": self.max_relative_error,
        }


def check_outputs(
    outputs: Mapping[str, object],
    supported: tuple[str, ...],
    reference,
    size: int,
    label: str,
) -> dict[str, SideCheck]:
    """Compare each supported side's latest output at the checked rows."""

    if tuple(outputs) != supported:
        raise MetalTcError(f"{label}: the outputs must cover the supported sides")
    checks = {}
    for side in supported:
        output = outputs[side]
        if (
            not isinstance(output, numpy.ndarray)
            or output.dtype != numpy.float32
            or output.shape != (size * size,)
        ):
            raise MetalTcError(f"{label}: {side} returned no flat float32 output")
        precision = SIDE_PRECISION[side]
        bound = (
            reference.bound(precision)
            if side in VARIANTS
            else reference.vendor_bound(precision)
        )
        checks[side] = SideCheck(*compare(output, reference, bound, f"{label} {side}"))
    return checks


def _same_bits(outputs: Mapping[str, numpy.ndarray], side: str) -> bool:
    return bool(
        numpy.array_equal(
            outputs[side].view(numpy.uint32), outputs[S1].view(numpy.uint32)
        )
    )


@dataclass(frozen=True)
class TimedPoint:
    """One timing point: the supported sides timed on the same operands."""

    point: TcPoint
    check_rows: int
    supported: tuple[str, ...]
    digests: dict[str, dict[str, str]]
    reference: dict[str, str]
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]
    same_bits: dict[str, bool]

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, rates, checks and bit facts."""

        size = self.point.size
        flops = matmul_flops(size, size, size)
        return {
            **self.point.to_mapping(),
            "launch": launch_mapping(self.point.shape()),
            "inputs": self.digests,
            "check_rows": self.check_rows,
            **self.reference,
            "side_order": list(SIDES),
            "supported": list(self.supported),
            "sides": {
                timing.implementation: side_rates(
                    timing, flops, compulsory_bytes(timing.implementation, size)
                )
                for timing in self.timings
            },
            "checks": {name: check.to_mapping() for name, check in self.checks.items()},
            **self.same_bits,
        }


def benchmark_point(
    backend: object,
    *,
    seed: int,
    point: TcPoint,
    check_rows: int,
    warmup: int,
    repeats: int,
    target_flops_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> TimedPoint:
    """Time the seven sides of one point in rotation, then check their outputs."""

    if not isinstance(point, TcPoint):
        raise TypeError("a timing point must be a TcPoint")
    shape = point.shape()
    values = operands(seed, BENCHMARK_STREAM, point.index, shape)
    if budget is not None:
        budget.check()
    workload = backend.workload(values)
    if tuple(workload.sides) != SIDES:
        raise MetalTcError("a workload must bind the seven sides in plan order")
    supported = tuple(workload.supported)
    if not {S1, M1, TORCH_FLOAT32} <= set(supported):
        raise MetalTcError("S1, M1 and the float32 PyTorch call must run")
    calls = calls_per_sample(point.size, target_flops_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"point {point.index} ({point.size} x {point.size})"
    reference = tc_reference(values, check_rows)
    outputs = workload.outputs()
    checks = check_outputs(outputs, supported, reference, point.size, label)
    timer = timer_name(clock)
    return TimedPoint(
        point=point,
        check_rows=check_rows,
        supported=supported,
        digests=values.digests(),
        reference=reference.digests(),
        timings=tuple(
            DeviceTiming(name, calls, warmup, repeats, series, timer)
            for name, series in zip(SIDES, samples, strict=True)
            if name in supported
        ),
        checks=checks,
        same_bits={
            "m1_same_bits_as_s1": _same_bits(outputs, M1),
            "torch_float32_same_bits_as_s1": _same_bits(outputs, TORCH_FLOAT32),
        },
    )
