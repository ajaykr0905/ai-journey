"""Time N0, F1 and the PyTorch call in one rotated run per point.

Each point draws one batch of seeded sequences. It first measures the
memory of each side alone (``metal_flash_memory``). It then places the batch
on the device once and binds the three sides to it in plan order: N0, F1,
then ``torch.nn.functional.scaled_dot_product_attention``. The timing is the
Day 74 and Day 76 method, run by the Day 78 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again. One call of N0 is its three launches.
3. Repeat r starts with side r mod 3 and runs each of the three sides once
   in order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``. Every side runs the same batch, so a ratio of
median rates is the inverse ratio of median sample times. After the last
sample, the latest output of each kernel must be within its declared bound
of the float64 reference at the checked rows, and the output of PyTorch
within its sanity bound, or the run stops.

Each point also records a traffic model and a count of the arithmetic. A
model value is not a measurement. Only the host timer
``time.perf_counter_ns`` makes a measurement; any other clock is named
``injected``. These are host-specific Apple GPU (Metal) timings; nothing
here measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from .kernel_metal_benchmark import DeviceTiming, RunBudget, calls_per_sample
from .memory_access_benchmark import timer_name
from .metal_flash_config import FlashPoint, compulsory_bytes
from .metal_flash_correctness import compare
from .metal_flash_memory import point_memory
from .metal_flash_plan import SIDES, TORCH_SDPA
from .metal_flash_reference import (
    BENCHMARK_STREAM,
    FlashLayout,
    flash_inputs,
    flash_reference,
    float64_digest,
)
from .metal_flash_runtime import launch_mappings
from .metal_flash_source import (
    BLOCK_KEYS,
    F1,
    FLOAT32_BYTES,
    HEAD_DIM,
    N0,
    MetalFlashError,
)
from .metal_fused_benchmark import side_rates
from .metal_launch_sweep_benchmark import time_shapes

FLOPS_PER_PAIR = 4 * HEAD_DIM
ARITHMETIC = (
    "4 * 64 floating-point operations for each visible (query, key) pair: one "
    "multiply and one add for each of the 64 products of q k and of p v; the "
    "exp, max and sum are not counted"
)
TRAFFIC_MODEL = {
    N0: (
        "16 * 64 bytes for each token (read Q, K and V once, write O once) plus "
        "16 bytes for each visible pair: S written once, read and written by "
        "the softmax, and read once by P V; the model ignores the rereads of K "
        "and V rows, which the cache can serve"
    ),
    F1: (
        "8 * 64 bytes for each token (read Q once, write O once) plus 2 * 32 * "
        "64 * 4 bytes for each K/V tile that a query block loads"
    ),
    TORCH_SDPA: "no model, because PyTorch does not name its kernel",
}


def modeled_bytes(side: str, layout: FlashLayout, mode: str) -> int | None:
    """Return the traffic model's device bytes for one call of one side."""

    if side == N0:
        return compulsory_bytes(layout) + 16 * layout.pairs(mode)
    if side == F1:
        tile = 2 * BLOCK_KEYS * HEAD_DIM * FLOAT32_BYTES
        return 2 * FLOAT32_BYTES * HEAD_DIM * layout.tokens + tile * layout.tile_loads(
            mode
        )
    if side == TORCH_SDPA:
        return None
    raise MetalFlashError(f"{side!r} is not a timed side")


def flash_rates(
    timing: DeviceTiming, compulsory: int, modeled: int | None, flops: int
) -> dict[str, object]:
    """Return the Day 79 rates of one side with its GFLOP/s added."""

    rates = side_rates(timing, compulsory, modeled)
    summary = timing.summary
    work = flops * timing.calls_per_sample
    return {
        **rates,
        "flops_per_call": flops,
        "gflops_per_second": work / summary.median_ns,
        "iqr_gflops_per_second": [work / summary.p75_ns, work / summary.p25_ns],
    }


@dataclass(frozen=True)
class SideCheck:
    """How one side's latest output compares with the float64 reference."""

    max_abs_error: float
    max_bound_ratio: float
    differing_from_rounded_reference: int

    def to_mapping(self) -> dict[str, object]:
        """Return the check as a report records it."""

        return {
            "max_abs_error": self.max_abs_error,
            "max_bound_ratio": self.max_bound_ratio,
            "differing_from_rounded_reference": self.differing_from_rounded_reference,
        }


def check_outputs(
    outputs: Mapping[str, object], reference, size: int, label: str
) -> dict[str, SideCheck]:
    """Compare each side's latest output at the checked rows, or stop the run."""

    checks = {}
    for side in SIDES:
        output = outputs.get(side)
        if (
            not isinstance(output, numpy.ndarray)
            or output.dtype != numpy.float32
            or output.shape != (size,)
        ):
            raise MetalFlashError(f"{label}: {side} returned no flat float32 output")
        bound = reference.vendor_bound if side == TORCH_SDPA else reference.bound
        error, ratio, differing = compare(output, reference, bound, f"{label} {side}")
        checks[side] = SideCheck(float(error.max()), ratio, differing)
    return checks


@dataclass(frozen=True)
class TimedPoint:
    """One timing point: three sides timed on the same device rows."""

    point: FlashPoint
    check_rows: int
    digests: dict[str, str]
    reference_sha256: str
    memory: dict[str, object]
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, rates, memory, models and checks."""

        layout = self.point.shape.layout()
        mode = self.point.mode
        compulsory = compulsory_bytes(layout)
        flops = FLOPS_PER_PAIR * layout.pairs(mode)
        return {
            **self.point.to_mapping(),
            "layout": layout.to_mapping(),
            "launches": {
                variant: launch_mappings(variant, mode, layout) for variant in (N0, F1)
            },
            **self.digests,
            "check_rows": self.check_rows,
            "reference_sha256": self.reference_sha256,
            "pairs": layout.pairs(mode),
            "compulsory_bytes_per_call": compulsory,
            "side_order": [timing.implementation for timing in self.timings],
            "sides": {
                timing.implementation: flash_rates(
                    timing,
                    compulsory,
                    modeled_bytes(timing.implementation, layout, mode),
                    flops,
                )
                for timing in self.timings
            },
            "checks": {name: check.to_mapping() for name, check in self.checks.items()},
            "memory": self.memory,
        }


def benchmark_point(
    backend: object,
    *,
    seed: int,
    point: FlashPoint,
    inputs: str,
    check_rows: int,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> TimedPoint:
    """Measure the memory of each side, then time the three sides of one point."""

    if not isinstance(point, FlashPoint):
        raise TypeError("a timing point must be a FlashPoint")
    layout, mode = point.shape.layout(), point.mode
    values = flash_inputs(seed, BENCHMARK_STREAM, point.index, layout, inputs)
    memory = point_memory(layout, backend.memory(values, mode))
    if budget is not None:
        budget.check()
    workload = backend.workload(values, mode)
    sides = tuple(workload.sides)
    if sides != SIDES:
        raise MetalFlashError("a workload must bind N0, F1 and PyTorch in order")
    calls = calls_per_sample(compulsory_bytes(layout), target_bytes_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"point {point.index} ({layout.sequences} x {layout.longest} {mode})"
    reference = flash_reference(values, mode, check_rows)
    checks = check_outputs(workload.outputs(), reference, layout.buffer_elements, label)
    timer = timer_name(clock)
    return TimedPoint(
        point=point,
        check_rows=check_rows,
        digests=values.digests(),
        reference_sha256=float64_digest(reference.output),
        memory=memory,
        timings=tuple(
            DeviceTiming(name, calls, warmup, repeats, series, timer)
            for name, series in zip(sides, samples)
        ),
        checks=checks,
    )
