"""Time one call of every tiled-lab side on the three Day 75 clocks.

The Day 75 bottleneck rule needs three one-call times of each side: the
interval between two MPS events around the call, the host time until the
call completes, and the host time to issue it. The tiled report times
batches between two synchronizations, so it holds none of these. This run
times the same sides on the operands of the tiled run's points with the
gpu-timing lab's ``time_sides``: one launch and one clock per sample, with
the sides and clocks in rotating order.

The run times each side at a 1 x 1 x 1 floor first. That call does the
least work that a launch can do, so its host time bounds any fixed cost
that every call of the side pays. It then times every sweep shape of the
tiled config in order. After each shape, every output must agree with the
naive output within the tiled run's bound, because a timing of wrong
results is not a timing. A shape that fails stops the run.

A report seals the gpu-timing domain texts of its own schema_version. The
version 1 text calls the event interval device-side. New reports have
version 2, whose text says that the interval can also hold host time
between the two ``record()`` calls. A version 1 report keeps the words that
it was sealed with and still verifies.

Only ``TiledTimingRuntime`` with the host timer ``time.perf_counter_ns``
makes a measurement; any other backend or clock makes the report
synthetic. The budget starts after the kernels compile. The SHA-256 seal
follows ``memory_access_report``: it detects drift and tampering, but it is
not a signature and cannot prove that a timing was observed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
from .gpu_timing_benchmark import SCHEDULE, SideTiming, time_sides
from .gpu_timing_clocks import CLOCK_DEFINITIONS, DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION
from .gpu_timing_runtime import EventPair
from .kernel_metal_benchmark import MAX_POINTS, NS_PER_SECOND, RunBudget
from .kernel_metal_geometry import CheckedLaunch
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_benchmark import agreement_bound
from .matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_report import environment
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_metal_source import source_sha256
from .matmul_tiled_benchmark import AGREEMENT, _checks
from .matmul_tiled_config import MatmulTiledConfig
from .matmul_tiled_report import SWITCH_SOURCE
from .matmul_tiled_runtime import (
    MPS_SWITCHES,
    NAIVE_KERNEL,
    MatmulTiledRuntime,
    SideCheck,
)
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import TiledVariant, library_source, text_sha256
from .memory_access_benchmark import PERCENTILE_METHOD, TIMER, timer_name
from .memory_access_report import seal_report
from .memory_access_tiling import MatmulShape

REPORT_SCHEMA_VERSION = 2
# Each schema_version seals the gpu-timing domain texts of the same version,
# and the texts of a version never change. A report verifies only with the
# texts of its own version.
SCHEMA_VERSIONS = (1, 2)
REPORT_KIND = "matmul_tiled_one_call_timing"
FLOOR_SHAPE = (1, 1, 1)
# A sweep holds at most MAX_POINTS points, so this operand index is free.
FLOOR_INDEX = MAX_POINTS
FLOOR_RULE = (
    "every side first runs one 1 x 1 x 1 call on operands from case index "
    f"{FLOOR_INDEX}; its host time bounds a fixed cost that every call of the "
    "side pays"
)
OPERANDS = (
    "point i uses the operands of the tiled run's point i: seed, benchmark "
    "stream and case index i of the Day 74 generator"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "hardware_counters_used": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are one-call timings on the Apple GPU of the host that made this "
        "report, at the recorded load averages, on two clocks: MPS events in the "
        "command stream and perf_counter_ns on the host. An MPS event interval "
        "holds the signal latency of its events, so it is longer than the "
        "kernel. No CUDA kernel ran, no NVIDIA GPU was used, cuBLAS was not "
        "measured, and no hardware counter was read. These timings do not "
        "measure or predict CUDA, NVIDIA, cuBLAS, distributed or production "
        "performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "hardware_counters_used": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration in it is synthetic. They are not measurements of this host or "
        "of any other host. No CUDA kernel ran, no NVIDIA GPU was used, and "
        "cuBLAS was not measured."
    ),
}


def method_mapping(schema_version: int = REPORT_SCHEMA_VERSION) -> dict[str, object]:
    """Return a fresh copy of the method texts that a report of a version embeds.

    The versions differ only in the domain texts.
    """

    if type(schema_version) is not int or schema_version not in SCHEMA_VERSIONS:
        raise ValueError(f"no one-call timing schema_version {schema_version!r}")
    return {
        "clocks": dict(CLOCK_DEFINITIONS),
        "domains": dict(DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[schema_version]),
        "schedule": SCHEDULE,
        "floor": FLOOR_RULE,
        "operands": OPERANDS,
        "agreement": AGREEMENT,
    }


class MatmulTiledGapError(MatmulTiledError):
    """Raised when gap inputs are unreadable, invalid or do not pair."""


class MatmulTiledGapVerificationError(MatmulTiledGapError):
    """Raised when a timing or gap report, or one of its sources, fails its checks."""


class TiledTimingRuntime(MatmulTiledRuntime):
    """The tiled runtime plus the MPS event pairs that the event clock records."""

    def event_pair(self) -> EventPair:
        """Return a fresh start and end event for one side."""

        return EventPair(self._torch)


@dataclass(frozen=True)
class OneCallPoint:
    """One shape: every side timed one call at a time on the same operands."""

    index: int
    shape: MatmulShape
    timings: tuple[SideTiming, ...]
    checks: Mapping[str, SideCheck]

    def to_mapping(self) -> dict[str, object]:
        """Return the raw samples of every clock beside the output checks."""

        return {
            "index": self.index,
            **self.shape.to_mapping(),
            "flops_per_call": self.shape.flops,
            "side_order": [timing.implementation for timing in self.timings],
            "sides": {
                timing.implementation: timing.to_mapping() for timing in self.timings
            },
            "checks": {
                name: {
                    "same_bits_as_naive": check.same_bits_as_naive,
                    "disagreement": check.disagreement,
                }
                for name, check in self.checks.items()
            },
            "agreement_bound": agreement_bound(self.shape.k),
        }


def time_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    variants: tuple[TiledVariant, ...],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> OneCallPoint:
    """Time one call of every side at one shape, then check the outputs."""

    if not isinstance(shape, MatmulShape):
        raise TypeError("a timed shape must be a MatmulShape")
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, BENCHMARK_STREAM, index, shape)
    groups = {NAIVE_KERNEL: naive_group_size}
    groups.update({variant.name: variant.group_size for variant in variants})
    launches = {
        name: CheckedLaunch(
            matmul_launch(X_COLUMNS, shape, group), backend.limits[name]
        )
        for name, group in groups.items()
    }
    workload = backend.workload(left, right, launches)
    sides = tuple(workload.sides)
    if sides != (*launches, TORCH_MATMUL):
        raise MatmulTiledGapError(
            "a workload must bind every kernel and then torch.matmul"
        )
    timings = time_sides(
        dict(workload.sides),
        synchronize=workload.synchronize,
        event_pair=backend.event_pair,
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    label = f"point {index} ({shape.m} x {shape.n} x {shape.k})"
    checks = _checks(workload.check(), sides, label, shape.k)
    return OneCallPoint(index, shape, timings, checks)


def _kernels(backend: object, variants: tuple[TiledVariant, ...]) -> dict:
    return {
        "naive_source_sha256": source_sha256(),
        "tiled_source_sha256": text_sha256(library_source(variants)),
        "limits": {
            name: backend.limits[name].to_mapping()
            for name in (NAIVE_KERNEL, *(variant.name for variant in variants))
        },
        "admissions": [
            backend.admissions[variant.name].to_mapping() for variant in variants
        ],
    }


def _fastest(points: list[dict[str, object]], sides: list[str]) -> dict:
    """Name the smallest host median of each side; the first entry wins a tie."""

    fastest = {}
    for side in sides:
        best = None
        for point in points:
            host = point["sides"][side]["clocks"]["host_synchronized"]["timing"]
            if best is None or host["median_ns"] < best["median_host_ns"]:
                best = {
                    "m": point["m"],
                    "n": point["n"],
                    "k": point["k"],
                    "median_host_ns": host["median_ns"],
                }
        fastest[side] = best
    return fastest


def _summary(
    label: str,
    floor: dict[str, object],
    points: list[dict[str, object]],
    load: dict[str, object],
) -> dict[str, object]:
    timed = [floor, *points]
    sides = list(floor["side_order"])
    return {
        "label": label,
        "points": len(points),
        "sides": sides,
        "timed_samples": sum(
            len(clock["samples_ns"])
            for point in timed
            for side in point["sides"].values()
            for clock in side["clocks"].values()
        ),
        "fastest_host": _fastest(timed, sides),
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def build_timing_report(
    config: MatmulTiledConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Time the floor and every sweep shape; assemble the unsealed report.

    ``backend=None`` compiles the real ``TiledTimingRuntime``, which raises
    ``MatmulTiledUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MatmulTiledConfig):
        raise TypeError("config must be a MatmulTiledConfig")
    variants = config.tiled_variants()
    if backend is None:
        backend = TiledTimingRuntime(variants)
    real = type(backend) is TiledTimingRuntime
    if real and backend.tiled_source != library_source(variants):
        raise MatmulTiledGapError(
            "the runtime compiled another tiled library than the config names"
        )
    admitted = tuple(
        variant for variant in variants if backend.admissions[variant.name].admitted
    )
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    common = {
        "seed": config.seed,
        "naive_group_size": config.naive.group_size,
        "variants": admitted,
        "warmup": timing.warmup,
        "repeats": timing.repeats,
        "clock": clock,
        "budget": budget,
    }
    floor = time_point(
        backend, index=FLOOR_INDEX, shape=matmul_shape(*FLOOR_SHAPE), **common
    )
    points = [
        time_point(backend, index=index, shape=shape, **common).to_mapping()
        for index, shape in enumerate(timing.shapes)
    ]
    elapsed = budget.check()
    load = {
        "source": LOAD_AVERAGE_SOURCE,
        "before": before,
        "after": read_load_average(load_average),
    }
    timer = timer_name(clock)
    measured = real and timer == TIMER
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    floor_mapping = floor.to_mapping()
    host_samples = [
        sample
        for point in (floor_mapping, *points)
        for side in point["sides"].values()
        for name in ("host_synchronized", "host_dispatch")
        for sample in side["clocks"][name]["samples_ns"]
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "measured": measured,
        "measurement_scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
        "label": label,
        "evidence_boundary": dict(
            MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY
        ),
        "backend": BACKEND if real else INJECTED_BACKEND,
        "timer": timer,
        "percentile_method": PERCENTILE_METHOD,
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "environment": environment(backend),
        "torch_mps_switches": {
            "source": SWITCH_SOURCE,
            "values": {name: backend.mps_switches[name] for name in MPS_SWITCHES},
        },
        "host_load": load,
        "kernels": _kernels(backend, variants),
        "method": method_mapping(REPORT_SCHEMA_VERSION),
        "floor": floor_mapping,
        "points": points,
        "run": {
            "max_runtime_ns": timing.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "host_timed_ns": sum(host_samples),
        },
        "summary": _summary(label, floor_mapping, points, load),
    }


def measure_timing_report(
    config: MatmulTiledConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the one-call timing and return the report sealed with its digest."""

    return seal_report(
        build_timing_report(
            config, backend=backend, clock=clock, load_average=load_average
        )
    )
