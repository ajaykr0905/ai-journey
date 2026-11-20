"""Run the three-clock sweep under one budget, then seal a labeled report.

One run builds the runtime, reads the load averages, times the empty kernel
and the sweep under one ``RunBudget``, reads the load averages again, and
derives the analysis. The budget starts after the kernels compile, so the
time to import torch on a busy host does not count. The report records the
config and its digest, both shader digests, the clock and domain
definitions, the host environment without identifiers, the load averages,
the raw samples of every side on every clock, the analysis and a summary.

New reports seal schema version 2. It differs from version 1 only in the
event-domain text, which no longer calls the interval device-side;
``gpu_timing_clocks`` keeps the texts of both versions.

Labels keep the evidence honest. Only the real ``GpuTimingRuntime`` with
host timer ``time.perf_counter_ns`` makes a measurement, with the Day 73
scope ``apple_gpu_metal_this_host`` and label ``host-specific Apple GPU
(Metal) measurement``. Any other backend or clock makes the report
synthetic: backend or timer ``injected``, ``measured: false``, and a
boundary that denies any measurement.

The SHA-256 seal follows ``memory_access_report``: canonical JSON of every
field but the digest. It detects drift and tampering; it is not a signature
and cannot prove that a timing was observed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
from .gpu_timing_analysis import analyze
from .gpu_timing_benchmark import SCHEDULE
from .gpu_timing_clocks import (
    CLOCK_DEFINITIONS,
    CLOCK_DOMAINS,
    CLOCKS,
    DISPATCH,
    DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION,
    EVENT,
    SYNCHRONIZED,
    GpuTimingError,
)
from .gpu_timing_config import GpuTimingConfig
from .gpu_timing_fit import SIZE_DOMINANT
from .gpu_timing_runtime import GpuTimingRuntime
from .gpu_timing_source import EMPTY_KERNEL, EMPTY_SOURCE, empty_source_sha256
from .gpu_timing_sweep import measure_sweep
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_benchmark import AGREEMENT
from .matmul_metal_report import TORCH_EQUIVALENT, environment
from .matmul_metal_runtime import SIDES
from .matmul_metal_source import KERNELS, source_sha256
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "gpu_timing_three_clocks"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "hardware_counters_used": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are timings on the Apple GPU of the host that made this report, "
        "on two clocks: MPS events in the command stream and perf_counter_ns "
        "on the host. They are host-specific Apple GPU (Metal) measurements at "
        "the recorded load averages. No CUDA kernel ran, no NVIDIA GPU was "
        "used, and no hardware counter was read. A fixed overhead or a regime "
        "is an inference from a fitted line, and a fraction of the FMA ceiling "
        "compares with an earlier measurement on this host, not a vendor peak."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "hardware_counters_used": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, fit, split and rate in it is synthetic. They are not "
        "measurements of this host or of any other host. No CUDA kernel ran, "
        "and no NVIDIA GPU was used."
    ),
}


def _shaders(backend: object) -> dict[str, object]:
    return {
        "matmul_source_sha256": source_sha256(),
        "empty_source": EMPTY_SOURCE,
        "empty_source_sha256": empty_source_sha256(),
        "limits": {
            name: backend.limits[name].to_mapping() for name in (*KERNELS, EMPTY_KERNEL)
        },
    }


def _crossover(analysis: dict[str, object], side: str) -> int | None:
    """Return the first point index whose event median is size-dominant."""

    for point in analysis["points"]:
        if point["sides"][side]["clocks"][EVENT]["regime"] == SIZE_DOMINANT:
            return point["index"]
    return None


def _summary(
    label: str,
    analysis: dict[str, object],
    timed_samples: int,
    load: dict[str, object],
) -> dict[str, object]:
    largest = analysis["points"][-1]
    smallest = analysis["overhead_estimate"]["smallest_shape"]
    sides = {}
    for side in SIDES:
        fits = analysis["fits"][side]
        entry = largest["sides"][side]
        sides[side] = {
            "intercept_ns": {clock: fits[clock]["intercept_ns"] for clock in CLOCKS},
            "smallest_shape_median_ns": smallest["median_ns"][side],
            "first_size_dominant_index": _crossover(analysis, side),
            "largest_event_regime": entry["clocks"][EVENT]["regime"],
            "largest_event_gflops_per_second": entry["event_gflops_per_second"],
            "largest_event_fraction_of_fma_ceiling": entry[
                "event_fraction_of_fma_ceiling"
            ],
            "largest_dispatch_to_synchronized_ratio": entry[
                "dispatch_to_synchronized_ratio"
            ],
        }
    return {
        "label": label,
        "sweep_points": len(analysis["points"]),
        "timed_samples": timed_samples,
        "empty_kernel_median_ns": analysis["overhead_estimate"][
            "empty_kernel_median_ns"
        ],
        "largest_point": {key: largest[key] for key in ("index", "m", "n", "k")},
        "sides": sides,
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def build_report(
    config: GpuTimingConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and assemble the unsealed report.

    ``backend=None`` compiles the real ``GpuTimingRuntime``, which raises
    ``GpuTimingUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, GpuTimingConfig):
        raise TypeError("config must be a GpuTimingConfig")
    sweep = config.sweep
    backend = GpuTimingRuntime() if backend is None else backend
    before = read_load_average(load_average)
    budget = RunBudget(sweep.max_runtime_seconds, clock)
    empty, points = measure_sweep(
        backend,
        seed=config.seed,
        shapes=sweep.shapes,
        group_size=config.launch.group_size,
        warmup=sweep.warmup,
        repeats=sweep.repeats,
        clock=clock,
        budget=budget,
    )
    elapsed = budget.check()
    load = {
        "source": LOAD_AVERAGE_SOURCE,
        "before": before,
        "after": read_load_average(load_average),
    }
    timings = (empty.timing, *(item for point in points for item in point.timings))
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise GpuTimingError("every sample must share one known timer")
    real = type(backend) is GpuTimingRuntime
    measured = timers == {TIMER} and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    analysis = analyze(empty, points, config.reference.fma_ceiling_gflops_per_second)
    timed_samples = len(timings) * len(CLOCKS) * sweep.repeats
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
        "timer": timers.pop(),
        "percentile_method": PERCENTILE_METHOD,
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "environment": environment(backend),
        "host_load": load,
        "shaders": _shaders(backend),
        "clocks": {
            "order": list(CLOCKS),
            "definitions": dict(CLOCK_DEFINITIONS),
            "domains": dict(CLOCK_DOMAINS),
            "domain_definitions": dict(
                DOMAIN_DEFINITIONS_BY_SCHEMA_VERSION[REPORT_SCHEMA_VERSION]
            ),
            "schedule": SCHEDULE,
        },
        "planned_work": config.planned_work().to_mapping(),
        "benchmark": {
            "device": DEVICE,
            "torch_equivalent": TORCH_EQUIVALENT,
            "agreement": AGREEMENT,
            "empty": empty.to_mapping(),
            "points": [point.to_mapping() for point in points],
        },
        "analysis": analysis,
        "run": {
            "max_runtime_ns": sweep.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_host_ns": sum(
                sum(item.samples(which).samples_ns)
                for item in timings
                for which in (SYNCHRONIZED, DISPATCH)
            ),
        },
        "summary": _summary(label, analysis, timed_samples, load),
    }


def measure_report(
    config: GpuTimingConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``GpuTimingRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(config, backend=backend, clock=clock, load_average=load_average)
    )
