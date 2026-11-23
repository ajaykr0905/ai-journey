"""Time each fused kernel and its PyTorch sequence in one rotated run per point.

Each point draws one batch of seeded rows, places it on the device once and
binds both sides to it: the fused kernel, then the PyTorch sequence of the
family. The timing is the Day 74 and Day 76 method, run by the Day 78 sweep
timer ``time_shapes``, as Day 79 runs it:

1. Each side runs ``warmup`` untimed samples.
2. Each timed sample calls ``synchronize``, reads ``perf_counter_ns``,
   launches ``calls_per_sample`` back-to-back calls, calls ``synchronize``
   again and reads the clock again.
3. Repeat r starts with side r mod 2, so the order of the two sides
   alternates at every repeat.

A sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``. Both sides run the same batch, so a ratio of
median rates is the inverse ratio of median sample times. After the last
sample, the latest output of the kernel must be within its declared bound
of the float64 reference, and the output of PyTorch within its sanity
bound, or the run stops.

Each point also records the bytes of a model with no cache for each side
(``TRAFFIC_MODEL``). A model value is not a measurement. Only the host timer
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
from .kernel_metal_geometry import vector_launch
from .memory_access_benchmark import timer_name
from .metal_fused_benchmark import side_rates
from .metal_fused_reference import float64_sha256
from .metal_launch_sweep_benchmark import time_shapes
from .metal_liger_config import LigerShape, compulsory_bytes
from .metal_liger_plan import TORCH_SIDES
from .metal_liger_reference import (
    BENCHMARK_STREAM,
    LigerLayout,
    LigerReference,
    liger_inputs,
    liger_reference,
)
from .metal_liger_source import (
    CROSS_ENTROPY,
    FAMILY_KERNELS,
    FLOAT32_BYTES,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
)

TRAFFIC_MODEL = {
    RMSNORM: (
        "device bytes of one call with no cache, for n elements in r rows, where "
        "a broadcast operand is read once for each output element: the kernel "
        "reads x and w and writes y, 12 n, plus 8 r of starts and lengths; the "
        "PyTorch sequence runs pow (8 n), mean (4 n + 4 r), the add of eps (8 r), "
        "rsqrt (8 r), x * inv (12 n) and * w (12 n), 40 n + 20 r"
    ),
    SWIGLU: (
        "device bytes of one call with no cache, for n elements in r rows: the "
        "kernel reads a and b and writes y, 12 n, plus 8 r of starts and lengths; "
        "the PyTorch sequence runs silu (8 n) and the multiply (12 n), 20 n"
    ),
    CROSS_ENTROPY: (
        "device bytes of one call with no cache, for n logits in r rows: the "
        "kernel reads each logit once, 4 n, and reads a 4-byte target, starts and "
        "lengths and writes a loss for each row, 16 r; the PyTorch call is "
        "modeled as log_softmax, which reads n logits and writes n "
        "log-probabilities, then nll_loss, which reads r of them, r 8-byte "
        "targets and writes r losses, 8 n + 16 r; PyTorch does not name its MPS "
        "kernels, so this side of the model is an assumption"
    ),
}
UNWRITTEN_SOFTMAX = (
    "4 n bytes for n logits: the float32 softmax or log-probabilities that the "
    "fused cross-entropy kernel never writes and never reads again"
)


def modeled_bytes(name: str, family: str, layout: LigerLayout) -> int:
    """Return the no-cache model's device bytes for one call of one side."""

    n, r = layout.elements, layout.rows
    if name == FAMILY_KERNELS.get(family):
        if family == CROSS_ENTROPY:
            return FLOAT32_BYTES * n + 16 * r
        return 3 * FLOAT32_BYTES * n + 8 * r
    if name == TORCH_SIDES.get(family):
        if family == RMSNORM:
            return 40 * n + 20 * r
        if family == SWIGLU:
            return 20 * n
        return 8 * n + 16 * r
    raise MetalLigerError(f"{name!r} is not a side of {family!r}")


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
    family: str,
    reference: LigerReference,
    size: int,
    label: str,
) -> dict[str, SideCheck]:
    """Compare both sides' latest outputs with the reference, or stop the run."""

    rounded = reference.output.astype(numpy.float32).view(numpy.uint32)
    checks = {}
    for name, bound in (
        (FAMILY_KERNELS[family], reference.bound),
        (TORCH_SIDES[family], reference.vendor_bound),
    ):
        output = outputs.get(name)
        if (
            not isinstance(output, numpy.ndarray)
            or output.dtype != numpy.float32
            or output.shape != (size,)
        ):
            raise MetalLigerError(f"{label}: {name} returned no packed float32 output")
        if not numpy.all(numpy.isfinite(output)):
            raise MetalLigerError(f"{label}: an output of {name} is not finite")
        error = numpy.abs(output.astype(numpy.float64) - reference.output)
        ratio = float(numpy.max(error / bound))
        if not ratio <= 1.0:
            raise MetalLigerError(f"{label}: {name} exceeds its bound by {ratio!r}")
        checks[name] = SideCheck(
            max_abs_error=float(error.max()),
            max_bound_ratio=ratio,
            differing_from_rounded_reference=int(
                numpy.count_nonzero(output.view(numpy.uint32) != rounded)
            ),
        )
    return checks


@dataclass(frozen=True)
class LigerPoint:
    """One timing point: both sides of one family timed on the same device rows."""

    family: str
    index: int
    shape: LigerShape
    digests: dict[str, str]
    reference_sha256: str
    timings: tuple[DeviceTiming, ...]
    checks: Mapping[str, SideCheck]

    @property
    def sides(self) -> tuple[str, ...]:
        """The timed sides in timing order: the kernel, then PyTorch."""

        return tuple(timing.implementation for timing in self.timings)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, rates, the model and the checks."""

        layout = self.shape.layout()
        family = self.family
        compulsory = compulsory_bytes(family, layout)
        size = layout.group_size(family)
        return {
            "family": family,
            "index": self.index,
            "shape": self.shape.to_mapping(),
            "layout": layout.to_mapping(family),
            "launch": vector_launch(layout.rows * size, size).to_mapping(),
            **self.digests,
            "reference_sha256": self.reference_sha256,
            "compulsory_bytes_per_call": compulsory,
            "unwritten_softmax_bytes": FLOAT32_BYTES * layout.elements
            if family == CROSS_ENTROPY
            else None,
            "side_order": list(self.sides),
            "sides": {
                timing.implementation: side_rates(
                    timing,
                    compulsory,
                    modeled_bytes(timing.implementation, family, layout),
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
    shape: LigerShape,
    inputs: str,
    warmup: int,
    repeats: int,
    target_bytes_per_sample: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> LigerPoint:
    """Time both sides of one point on one batch of device rows."""

    if not isinstance(shape, LigerShape):
        raise TypeError("a timing shape must be a LigerShape")
    family, layout = shape.family, shape.layout()
    values = liger_inputs(seed, family, BENCHMARK_STREAM, index, layout, inputs)
    workload = backend.workload(values)
    sides = tuple(workload.sides)
    if sides != (FAMILY_KERNELS[family], TORCH_SIDES[family]):
        raise MetalLigerError("a workload must bind the kernel and then PyTorch")
    calls = calls_per_sample(compulsory_bytes(family, layout), target_bytes_per_sample)
    samples = time_shapes(
        workload,
        calls_per_sample=calls,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"{family} point {index} ({layout.rows} x {layout.longest})"
    reference = liger_reference(values)
    size = layout.output_elements(family)
    checks = check_outputs(workload.outputs(), family, reference, size, label)
    timer = timer_name(clock)
    return LigerPoint(
        family=family,
        index=index,
        shape=shape,
        digests=values.digests(),
        reference_sha256=float64_sha256(reference.output),
        timings=tuple(
            DeviceTiming(name, calls, warmup, repeats, series, timer)
            for name, series in zip(sides, samples)
        ),
        checks=checks,
    )
