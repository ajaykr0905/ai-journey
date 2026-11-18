"""One sealed profiling snapshot of the simulator at one stage of the loop.

A snapshot is what one stage of the profiling loop observes on the current
tree. It runs these steps in order:

1. Read the load averages.
2. Run the misuse cases and require the digest pinned before round 1.
3. Profile one checked pass of each workload with cProfile.
4. Time ``repeats`` checked passes of each workload with no profiler.
5. Read the load averages again.

The report keeps every profiled function per workload, the merged profile of
all three workloads, its ranked hotspots, the raw timed samples and their
statistics. A stage is ``baseline`` or ``round-N``, and it must be a stage
of the config, so the snapshot after round N binds to a config that names
round N.

Only ``time.perf_counter_ns`` makes a measurement: scope
``cpu_wall_clock_this_host`` and the label ``host-specific CPU
measurement``. Any other clock makes the timings synthetic. The profiler
times are always the profiler's own and only rank functions. The SHA-256
seal follows ``kernel_launch_report``. It detects drift and tampering, and
it is not a signature.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from .kernel_launch_kernels import KernelRun
from .kernel_launch_report import seal_report
from .memory_access_benchmark import PERCENTILE_METHOD, TIMER
from .sim_profile_config import SimProfileConfig, SimProfileError
from .sim_profile_harness import (
    RANKING,
    FunctionProfile,
    merge_functions,
    profile_call,
    rank_hotspots,
)
from .sim_profile_semantics import checked_semantics
from .sim_profile_timing import (
    LOAD_AVERAGE_SOURCE,
    WorkloadTiming,
    host_environment,
    read_load_average,
    time_workload,
)
from .sim_profile_workloads import Workload, build_workloads

SCHEMA_VERSION = 1
SNAPSHOT_KIND = "sim_profile_snapshot"
MEASUREMENT_SCOPE = "cpu_wall_clock_this_host"
SYNTHETIC_SCOPE = "synthetic_injected_clock"
MEASUREMENT_LABEL = "host-specific CPU measurement"
SYNTHETIC_LABEL = "synthetic timing, not a measurement"
GETITEM = "ai_journey.kernel_launch_buffer:DeviceBuffer.__getitem__"
SETITEM = "ai_journey.kernel_launch_buffer:DeviceBuffer.__setitem__"
_COMMON_BOUNDARY = {
    "pure_python_simulator": True,
    "gpu_executed": False,
    "cuda_compiled": False,
    "nvidia_gpu_used": False,
    "hardware_counters_used": False,
    "production_claim": False,
    "portable_performance_claim": False,
}
MEASUREMENT_BOUNDARY = {
    **_COMMON_BOUNDARY,
    "cpu_wall_clock_this_host": True,
    "host_specific": True,
    "statement": (
        "These timings are wall-clock CPU measurements of the pure-Python "
        "launch simulator on the host that made this report. They depend on "
        "that host, its Python and its load. A hotspot is an inference from "
        "profiler self time and is not read from hardware counters. No GPU ran, "
        "no CUDA source was compiled, and no NVIDIA GPU was used."
    ),
}
SYNTHETIC_BOUNDARY = {
    **_COMMON_BOUNDARY,
    "cpu_wall_clock_this_host": False,
    "synthetic_timing": True,
    "statement": (
        "An injected clock timed the passes, so every timed sample in this "
        "report is synthetic and measures no host. The profiler counts and "
        "the output checks are real. No GPU ran, no CUDA source was compiled, "
        "and no NVIDIA GPU was used."
    ),
}


def _profiled(workload: Workload) -> tuple[tuple[FunctionProfile, ...], KernelRun]:
    """Profile one checked pass; its buffer calls must match its access log."""

    functions, result = profile_call(workload.run)
    workload.check(result)
    calls = {record.function: record.calls for record in functions}
    log = result.result.log
    if (
        calls.get(GETITEM, 0) != log.reads + log.out_of_bounds_reads
        or calls.get(SETITEM, 0) != log.writes + log.out_of_bounds_writes
    ):
        raise SimProfileError(f"the {workload.name} profile missed buffer accesses")
    return functions, result


def _workload_entry(
    workload: Workload,
    functions: tuple[FunctionProfile, ...],
    result: KernelRun,
    timing: WorkloadTiming,
) -> dict[str, object]:
    log = result.result.log
    return {
        "name": workload.name,
        "launch": workload.config.geometry(),
        "output_dtype": workload.output_dtype,
        "reference_sha256": workload.reference_sha256,
        "output_sha256": result.output_sha256,
        "reads": log.reads,
        "writes": log.writes,
        "profile": _profile_totals(functions),
        "timing": timing.to_mapping(),
    }


def _profile_totals(functions: tuple[FunctionProfile, ...]) -> dict[str, object]:
    return {
        "calls": sum(record.calls for record in functions),
        "primitive_calls": sum(record.primitive_calls for record in functions),
        "total_ns": sum(record.total_ns for record in functions),
        "functions": [record.to_mapping() for record in functions],
    }


def build_snapshot(
    config: SimProfileConfig,
    stage: str,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Profile and time the current tree at one stage; return an unsealed report."""

    if type(config) is not SimProfileConfig:
        raise TypeError("config must be SimProfileConfig")
    if type(stage) is not str or stage not in config.stages:
        raise SimProfileError(f"stage must be one of the config stages {config.stages}")
    before = read_load_average(load_average)
    semantics = checked_semantics()
    workloads = build_workloads(config)
    profiled = [_profiled(workload) for workload in workloads]
    timings = [
        time_workload(
            workload, warmup=config.warmup, repeats=config.repeats, clock=clock
        )
        for workload in workloads
    ]
    after = read_load_average(load_average)
    timer = timings[0].timer
    measured = timer == TIMER
    merged = merge_functions(
        record for functions, _ in profiled for record in functions
    )
    hotspots = rank_hotspots(merged, config.top_functions)
    medians = [timing.summary.median_ns for timing in timings]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "stage": stage,
        "config": config.snapshot_mapping(),
        "workload_sha256": config.workload_sha256,
        "environment": host_environment(),
        "timer": timer,
        "measured": measured,
        "scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
        "label": MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL,
        "percentile_method": PERCENTILE_METHOD,
        "ranking": RANKING,
        "load_average_source": LOAD_AVERAGE_SOURCE,
        "load_average_before": before,
        "load_average_after": after,
        "semantics": semantics,
        "workloads": [
            _workload_entry(workload, functions, result, timing)
            for workload, (functions, result), timing in zip(
                workloads, profiled, timings
            )
        ],
        "profile": {
            **_profile_totals(merged),
            "hotspots": [hotspot.to_mapping() for hotspot in hotspots],
        },
        "summary": {
            "total_median_ns": sum(medians),
            "top_hotspot": hotspots[0].function,
            "top_hotspot_share_ppm": hotspots[0].share_ppm,
        },
        "evidence_boundary": dict(
            MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY
        ),
    }


def snapshot(
    config: SimProfileConfig,
    stage: str,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Build and seal one snapshot."""

    return seal_report(
        build_snapshot(config, stage, clock=clock, load_average=load_average)
    )
