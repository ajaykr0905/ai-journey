"""Time every softmax step and ``torch.softmax`` in one rotated run.

Each point draws one batch of seeded rows, places it on the device once and
binds every side to it: the admitted steps in plan order, then
``torch.softmax(x, dim=-1, out=y)``. The timing is the Day 74 and Day 76
method, run by the Day 74 sweep timer ``time_shapes``:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again. One call of U0 is its three launches.
3. Repeat r starts with side r mod S and runs each of the S sides once in
   order, so the order shifts by one side at every repeat.

A sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``, with the Day 73 ``calls_per_sample``. Every
side runs the same batch, so a ratio of median rates is the inverse ratio of
median sample times. After the last sample, the latest output of each step
must be within its bound of the float64 reference, and the output of
``torch.softmax`` within ``VENDOR_BOUND``, or the run stops.

Each point also records the bytes of a model with no cache (``TRAFFIC_MODEL``)
for each step. A model value is not a measurement. Only the host timer
``time.perf_counter_ns`` makes a measurement; any other clock is named
``injected``. These are host-specific Apple GPU (Metal) timings; nothing here
measures CUDA or NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from .kernel_metal_benchmark import DeviceTiming, RunBudget, calls_per_sample
from .kernel_metal_geometry import vector_launch
from .memory_access_benchmark import timer_name
from .metal_fused_config import TimingShape, compulsory_bytes
from .metal_fused_plan import TORCH_SOFTMAX
from .metal_fused_reference import (
    BENCHMARK_STREAM,
    FLOOR,
    RowLayout,
    SoftmaxReference,
    error_bounds,
    float32_sha256,
    float64_sha256,
    softmax_inputs,
    softmax_reference,
)
from .metal_fused_source import (
    FLOAT32_BYTES,
    STEP_VARIANTS,
    U0,
    UINT32_BYTES,
    MetalFusedError,
)
from .metal_launch_sweep_benchmark import time_shapes

TRAFFIC_MODEL = (
    "device bytes of one call, with no cache, for n elements in r rows: U0 reads "
    "the row in each of its three kernels and writes it once, 16 n bytes, and "
    "moves 44 r bytes of row data (8 r of starts and lengths in each kernel, "
    "the max written once and read twice, the sum written once and read once); "
    "a fused step reads the row once and writes it once, 8 n bytes, plus 8 r of "
    "starts and lengths; torch.softmax has no model, because PyTorch does not "
    "name its kernel"
)
COMPULSORY = (
    "8 n bytes for every side: each of the n elements read once and written "
    "once; a ragged point credits torch.softmax with the n elements, not with "
    "the padding it also processes"
)
VENDOR_BOUND = (
    "|y - y_ref| <= 2**-16 * y_ref + 2**-125 for every torch.softmax output: a "
    "sanity bound only, because PyTorch names no kernel and no derived bound "
    "applies"
)
VENDOR_UNIT = 2.0**-16
ROW_BYTES = 2 * UINT32_BYTES


def modeled_bytes(variant: str, layout: RowLayout) -> int:
    """Return the no-cache model's device bytes for one call of one step."""

    if variant not in STEP_VARIANTS:
        raise MetalFusedError(f"{variant!r} is not a registered step")
    elements, rows = layout.elements, layout.rows
    if variant == U0:
        row_data = 3 * ROW_BYTES + 3 * FLOAT32_BYTES + 2 * FLOAT32_BYTES
        return 4 * FLOAT32_BYTES * elements + row_data * rows
    return 2 * FLOAT32_BYTES * elements + ROW_BYTES * rows


def side_rates(
    timing: DeviceTiming, compulsory: int, modeled: int | None
) -> dict[str, object]:
    """Return the raw samples with their statistics and GB/s rates."""

    summary = timing.summary
    calls = timing.calls_per_sample
    moved = compulsory * calls
    return {
        **timing.to_mapping(),
        "median_ns_per_call": summary.median_ns / calls,
        "iqr_ns_per_call": summary.iqr_ns / calls,
        "compulsory_bytes_per_call": compulsory,
        "modeled_bytes_per_call": modeled,
        "effective_gb_per_second": moved / summary.median_ns,
        "iqr_effective_gb_per_second": [
            moved / summary.p75_ns,
            moved / summary.p25_ns,
        ],
        "modeled_gb_per_second": None
        if modeled is None
        else modeled * calls / summary.median_ns,
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
    outputs: Mapping[str, object],
    sides: tuple[str, ...],
    reference: SoftmaxReference,
    label: str,
) -> dict[str, SideCheck]:
    """Compare every side's latest output with the reference, or stop the run."""

    layout = reference.layout
    rounded = reference.output.astype(numpy.float32).view(numpy.uint32)
    checks = {}
    for name in sides:
        output = outputs.get(name)
        if (
            not isinstance(output, numpy.ndarray)
            or output.dtype != numpy.float32
            or output.shape != (layout.elements,)
        ):
            raise MetalFusedError(f"{label}: {name} returned no packed float32 rows")
        if not numpy.all(numpy.isfinite(output)):
            raise MetalFusedError(f"{label}: an output of {name} is not finite")
        error = numpy.abs(output.astype(numpy.float64) - reference.output)
        if name == TORCH_SOFTMAX:
            bound = VENDOR_UNIT * reference.output + FLOOR
        else:
            bound = error_bounds(name, reference).elements
        ratio = float(numpy.max(error / bound))
        if not ratio <= 1.0:
            raise MetalFusedError(f"{label}: {name} exceeds its bound by {ratio!r}")
        checks[name] = SideCheck(
            max_abs_error=float(error.max()),
            max_bound_ratio=ratio,
            differing_from_rounded_reference=int(
                numpy.count_nonzero(output.view(numpy.uint32) != rounded)
            ),
        )
    return checks


@dataclass(frozen=True)
class FusedPoint:
    """One timing point: every side timed on the same device rows."""

    index: int
    shape: TimingShape
    layout: RowLayout
    input_sha256: str
    reference_sha256: str
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: the steps, then ``torch.softmax``."""

        return tuple(timing.implementation for timing in self.timings)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, rates, the model and the checks."""

        compulsory = compulsory_bytes(self.layout)
        size = self.layout.group_size
        return {
            "index": self.index,
            "shape": self.shape.to_mapping(),
            "layout": self.layout.to_mapping(),
            "launch": vector_launch(self.layout.rows * size, size).to_mapping(),
            "torch_input": {
                "rows": self.layout.rows,
                "cols": self.layout.longest,
                "padded": not self.layout.dense,
            },
            "input_sha256": self.input_sha256,
            "reference_sha256": self.reference_sha256,
            "compulsory_bytes_per_call": compulsory,
            "side_order": list(self.sides),
            "sides": {
                timing.implementation: side_rates(
                    timing,
                    compulsory,
                    None
                    if timing.implementation == TORCH_SOFTMAX
                    else modeled_bytes(timing.implementation, self.layout),
                )
                for timing in self.timings
            },
            "checks": {name: check.to_mapping() for name, check in self.checks.items()},
        }


def benchmark_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: TimingShape,
    inputs: str,
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> FusedPoint:
    """Time every side of one point on one batch of device rows."""

    if not isinstance(shape, TimingShape):
        raise TypeError("a timing shape must be a TimingShape")
    layout = shape.layout(seed, index)
    values = softmax_inputs(seed, BENCHMARK_STREAM, index, layout, inputs)
    workload = backend.workload(values, layout, names)
    sides = tuple(workload.sides)
    if sides != (*names, TORCH_SOFTMAX):
        raise MetalFusedError(
            "a workload must bind every admitted step and then torch.softmax"
        )
    calls = calls_per_sample(compulsory_bytes(layout), target_bytes_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"point {index} ({layout.rows} rows, longest {layout.longest})"
    reference = softmax_reference(values, layout)
    checks = check_outputs(workload.outputs(), sides, reference, label)
    timer = timer_name(clock)
    timings = tuple(
        DeviceTiming(name, calls, warmup, repeats, series, timer)
        for name, series in zip(sides, samples)
    )
    return FusedPoint(
        index=index,
        shape=shape,
        layout=layout,
        input_sha256=float32_sha256(values),
        reference_sha256=float64_sha256(reference.output),
        timings=timings,
        checks=checks,
    )
