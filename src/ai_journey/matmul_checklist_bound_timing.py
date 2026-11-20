"""Time one call of every checklist side on the three Day 75 clocks.

The Day 75 bottleneck rule needs three one-call times of each side: the
interval between two MPS events around the call, the host time until the
call completes, and the host time to issue it. The checklist report times
batches between two synchronizations, so it holds none of them. This run
times the same sides on the operands of the checklist run's points with the
gpu-timing lab's ``time_sides``, as the Day 76 gap timing does for the
tiles: one launch and one clock per sample, with the sides and the clocks in
rotating order.

The run times each side at a 1 x 1 x 1 floor first. That call does the least
work that a launch can do, so its host time bounds a fixed cost that every
call of the side pays. It then times every sweep shape of the checklist
config in order. After each shape, every output must agree with the S0
output within the bound of the checklist run, because a timing of wrong
results is not a timing. A shape that fails stops the run.

The report seals the version 2 domain texts of the gpu-timing lab, which say
that an event interval can also hold host time between the two ``record()``
calls. It captures Python warnings from before the runtime imports torch,
as the checklist run does. Only ``ChecklistTimingRuntime`` with the host
timer ``time.perf_counter_ns`` makes a measurement; any other backend or
clock makes the report synthetic. The budget starts after the kernels
compile. The SHA-256 seal detects drift and tampering, but it is not a
signature and cannot prove that a timing was observed.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
from .gpu_timing_benchmark import SCHEDULE, time_sides
from .gpu_timing_clocks import CLOCK_DEFINITIONS, DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION
from .gpu_timing_runtime import EventPair
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_checklist_bound_traffic import MatmulChecklistBoundError
from .matmul_checklist_config import MatmulChecklistConfig
from .matmul_checklist_correctness import runnable
from .matmul_checklist_runtime import MatmulChecklistRuntime, checked_launch
from .matmul_checklist_steps import REFERENCE
from .matmul_checklist_variants import (
    LIBRARY_VARIANTS,
    VARIANTS,
    admitted_names,
    checklist_library,
)
from .matmul_metal_mapping import matmul_shape
from .matmul_metal_reference import BENCHMARK_STREAM, matmul_inputs
from .matmul_metal_report import environment
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_metal_source import source_sha256
from .matmul_tiled_benchmark import AGREEMENT, _checks
from .matmul_tiled_gap_timing import (
    FLOOR_INDEX,
    FLOOR_RULE,
    FLOOR_SHAPE,
    OneCallPoint,
    _fastest,
)
from .matmul_tiled_report import SWITCH_SOURCE, warning_records
from .matmul_tiled_runtime import MPS_SWITCHES
from .matmul_tiled_variants import text_sha256
from .memory_access_benchmark import PERCENTILE_METHOD, TIMER, timer_name
from .memory_access_report import seal_report
from .memory_access_tiling import MatmulShape

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_checklist_one_call_timing"
# The gpu-timing domain texts that this report seals; version 2 says that an
# event interval can hold host time.
DOMAIN_SCHEMA_VERSION = 2
OPERANDS = (
    "point i uses the operands of the checklist run's point i: seed, benchmark "
    "stream and case index i of the Day 74 generator"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are one-call timings of the checklist kernels and torch.matmul "
        "on the Apple GPU of the host that made this report, at the recorded "
        "load averages, on two clocks: MPS events in the command stream and "
        "perf_counter_ns on the host. An MPS event interval holds the signal "
        "latency of its events and can hold host time between the two records, "
        "so it is longer than the kernel. No CUDA kernel ran, no NVIDIA GPU was "
        "used, cuBLAS was not measured, and no hardware counter was read. These "
        "timings do not measure or predict CUDA, NVIDIA, cuBLAS, distributed or "
        "production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
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


def method_mapping() -> dict[str, object]:
    """Return a fresh copy of the method texts that every timing report seals."""

    return {
        "clocks": dict(CLOCK_DEFINITIONS),
        "domains": dict(DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[DOMAIN_SCHEMA_VERSION]),
        "schedule": SCHEDULE,
        "floor": FLOOR_RULE,
        "operands": OPERANDS,
        "agreement": AGREEMENT,
    }


class ChecklistTimingRuntime(MatmulChecklistRuntime):
    """The checklist runtime plus the MPS event pairs that the event clock records."""

    def event_pair(self) -> EventPair:
        """Return a fresh start and end event for one side."""

        return EventPair(self._torch)


def time_point(
    backend: object,
    *,
    seed: int,
    index: int,
    shape: MatmulShape,
    naive_group_size: tuple[int, int],
    names: tuple[str, ...],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> OneCallPoint:
    """Time one call of every side at one shape, then check the outputs.

    ``names`` are the admitted kernels in registry order, so the S0 kernel
    comes first and ``torch.matmul`` follows the last kernel.
    """

    if not isinstance(shape, MatmulShape):
        raise TypeError("a timed shape must be a MatmulShape")
    names = runnable(backend, names)
    if not names or names[0] != REFERENCE:
        raise MatmulChecklistBoundError(f"a point times {REFERENCE} first")
    shape = matmul_shape(shape.m, shape.n, shape.k)
    left, right = matmul_inputs(seed, BENCHMARK_STREAM, index, shape)
    launches = {
        name: checked_launch(backend.limits, name, shape, naive_group_size)
        for name in names
    }
    workload = backend.workload(left, right, launches)
    sides = tuple(workload.sides)
    if sides != (*names, TORCH_MATMUL):
        raise MatmulChecklistBoundError(
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


def _kernels(backend: object) -> dict[str, object]:
    return {
        "naive_source_sha256": source_sha256(),
        "library_sha256": text_sha256(checklist_library()),
        "limits": {name: backend.limits[name].to_mapping() for name in VARIANTS},
        "admissions": [
            backend.admissions[name].to_mapping() for name in LIBRARY_VARIANTS
        ],
    }


def _measure(
    config: MatmulChecklistConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    common = {
        "seed": config.seed,
        "naive_group_size": config.naive.group_size,
        "names": admitted_names(backend.admissions),
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
    return {
        "floor": floor.to_mapping(),
        "points": points,
        "elapsed_ns": budget.check(),
        "load": {
            "source": LOAD_AVERAGE_SOURCE,
            "before": before,
            "after": read_load_average(load_average),
        },
    }


def _summary(
    label: str,
    floor: dict[str, object],
    points: list[dict[str, object]],
    refused: list[str],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    timed = [floor, *points]
    sides = list(floor["side_order"])
    return {
        "label": label,
        "points": len(points),
        "sides": sides,
        "refused_variants": refused,
        "timed_samples": sum(
            len(clock["samples_ns"])
            for point in timed
            for side in point["sides"].values()
            for clock in side["clocks"].values()
        ),
        "fastest_host": _fastest(timed, sides),
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def build_timing_report(
    config: MatmulChecklistConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Time the floor and every sweep shape; assemble the unsealed report.

    ``backend=None`` compiles the real ``ChecklistTimingRuntime``, which
    raises ``MatmulChecklistUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MatmulChecklistConfig):
        raise TypeError("config must be a MatmulChecklistConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = ChecklistTimingRuntime()
        real = type(backend) is ChecklistTimingRuntime
        # The report names the library by the registry's digest.
        if real and backend.library_source != checklist_library():
            raise MatmulChecklistBoundError(
                "the runtime compiled another checklist library than the registry"
            )
        run = _measure(config, backend, clock, load_average)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    measured = real and timer_name(clock) == TIMER
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    refused = [
        name for name in LIBRARY_VARIANTS if not backend.admissions[name].admitted
    ]
    host_samples = [
        sample
        for point in (run["floor"], *run["points"])
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
        "timer": timer_name(clock),
        "percentile_method": PERCENTILE_METHOD,
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "environment": environment(backend),
        "torch_mps_switches": {
            "source": SWITCH_SOURCE,
            "values": {name: backend.mps_switches[name] for name in MPS_SWITCHES},
        },
        "host_load": run["load"],
        "warnings": captured,
        "kernels": _kernels(backend),
        "method": method_mapping(),
        "floor": run["floor"],
        "points": run["points"],
        "run": {
            "max_runtime_ns": config.benchmark.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": run["elapsed_ns"],
            "host_timed_ns": sum(host_samples),
        },
        "summary": _summary(
            label, run["floor"], run["points"], refused, captured, run["load"]
        ),
    }


def measure_timing_report(
    config: MatmulChecklistConfig,
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
