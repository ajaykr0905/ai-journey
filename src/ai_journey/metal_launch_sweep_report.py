"""Run the whole sweep under one budget, then seal a labeled report.

After the kernels compile, one run does these steps in order:

1. It records the limits of the three pipelines and the runtime's SIMD rule.
2. It admits or rejects each matmul shape against the matmul pipeline, and
   each vector group size against the Day 73 ``vector_add`` pipeline.
3. It checks the matmul mapping of each admitted shape in the simulator.
4. It probes the SIMD layout of each admitted shape.
5. It runs every correctness case with every admitted shape.
6. It times every sweep point.

One ``RunBudget`` on the run's clock covers steps 3 to 6, so the torch
import and the compile on a busy host do not count. The report records the
config and its digest, the source and its digest, the host environment
without identifiers, the 1, 5 and 15 minute load averages before and after
the run, the planned work, the elapsed and timed nanoseconds, and a summary.

Only the real ``MetalLaunchSweepRuntime`` timed with ``time.perf_counter_ns``
makes a measurement, with the Day 73 scope and label of a host-specific
Apple GPU (Metal) measurement. Every such timing carries the load averages,
so a host that gives none stops the run before its first launch. Its
boundary rules out CUDA, NVIDIA, distributed, production, vendor-peak and
scheduler claims. Any other backend or clock makes the report synthetic, and
a synthetic run may go without load averages. The SHA-256 seal follows
``memory_access_report``: it detects drift and tampering, but it is not a
signature and cannot prove that a timing was observed.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
    host_environment,
)
from .kernel_metal_runtime import BACKEND
from .kernel_metal_source import KERNELS, VECTOR_ADD
from .kernel_metal_source import source_sha256 as day_73_source_sha256
from .memory_access_benchmark import (
    INJECTED_TIMER,
    PERCENTILE_METHOD,
    TIMER,
    timer_name,
)
from .memory_access_report import seal_report
from .metal_launch_sweep_benchmark import (
    SweepPoint,
    measure_matmul_point,
    measure_vector_point,
)
from .metal_launch_sweep_config import MetalLaunchSweepConfig
from .metal_launch_sweep_correctness import (
    BOUND,
    REFERENCE,
    check_matmul_case,
    check_vector_sizes,
)
from .metal_launch_sweep_layout import observe_simd_layout
from .metal_launch_sweep_mapping import MAPPING_CASE, check_mapping
from .metal_launch_sweep_runtime import PIPELINES, MetalLaunchSweepRuntime
from .metal_launch_sweep_shapes import (
    REJECTION_REASONS,
    MetalLaunchSweepError,
    admit_shapes,
)
from .metal_launch_sweep_source import (
    MATMUL,
    SWEEP_SOURCE,
    kernel_bindings,
    source_sha256,
)

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "metal_launch_sweep"
DEVICE = "mps"
GRID_RULE = "whole threadgroups on every axis: ceil(extent / group) groups per axis"
SYNCHRONIZATION = (
    "torch.mps.synchronize() before the first and before the second clock "
    "reading of every sample; untimed warmups for each shape first"
)
TURN_ORDER = (
    "repeat r runs every admitted shape once, starting with shape r mod S; "
    "one batch size and one set of input buffers per point"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "vendor_peak_claim": False,
    "scheduler_internals_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report. The kernels came from torch.mps.compile_shader. The "
        "timings are host-specific Apple GPU (Metal) measurements. No CUDA "
        "kernel ran, and no NVIDIA GPU was used. A best rate is the highest "
        "median rate that this run measured, not a vendor peak. The layout "
        "probe shows which SIMD group and lane ran each thread, not how the "
        "GPU schedules SIMD groups. These timings do not measure or predict "
        "CUDA, NVIDIA, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "vendor_peak_claim": False,
    "scheduler_internals_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration and rate in it is synthetic. They are not measurements of "
        "this host or of any other host. The checks ran on the outputs that "
        "the backend returned. No CUDA kernel ran, and no NVIDIA GPU was used."
    ),
}


def load_average() -> list[float] | None:
    """Return the 1, 5 and 15 minute load averages, or ``None`` where unknown.

    Other jobs on the host change the timings, so the report keeps the load
    next to them.
    """

    try:
        values = os.getloadavg()
    except (AttributeError, OSError):
        return None
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        return None
    return [round(float(value), 2) for value in values]


def _host_load(measuring: bool) -> list[float] | None:
    """Read the load averages; a measurement cannot be labeled without them.

    The check runs before the first launch and again after the last sample,
    so a host that gives no load averages stops before anything is timed.
    """

    loads = load_average()
    if measuring and loads is None:
        raise MetalLaunchSweepError(
            "a measurement must record the load averages of the host, and this "
            "host gives none"
        )
    return loads


def _summary_point(point: dict[str, object]) -> dict[str, object]:
    rate = (
        "median_gflops_per_second"
        if point["kernel"] == MATMUL
        else "median_gb_per_second"
    )
    extent = {key: point[key] for key in ("m", "n", "k", "length") if key in point}
    return {
        **extent,
        "fastest": point["ranking"][0],
        "slowest": point["ranking"][-1],
        "spread": point["spread"],
        "separated": point["separated"],
        f"best_{rate}": max(shape[rate] for shape in point["shapes"]),
    }


def _summary(
    label: str,
    admission: dict[str, list[dict[str, object]]],
    matmul_cases: list[dict[str, object]],
    vector_cases: list[dict[str, object]],
    matmul: list[dict[str, object]],
    vector: list[dict[str, object]],
    counts: tuple[int, int],
) -> dict[str, object]:
    entries = [*admission["matmul"], *admission["vector"]]
    ratios = [case["max_bound_ratio"] for case in matmul_cases] + [
        point["accuracy"]["max_bound_ratio"] for point in matmul
    ]
    matmul_rows = [_summary_point(point) for point in matmul]
    vector_rows = [_summary_point(point) for point in vector]
    return {
        "label": label,
        "admitted_shapes": sum(entry["admitted"] for entry in entries),
        "rejected_shapes": sum(not entry["admitted"] for entry in entries),
        "mapping_checks": counts[0],
        "layout_probes": counts[1],
        "correctness_cases": len(matmul_cases) + len(vector_cases),
        "matmul_max_bound_ratio": max(ratios),
        "benchmark_points": len(matmul) + len(vector),
        "timed_samples": sum(
            len(shape["samples_ns"])
            for point in (*matmul, *vector)
            for shape in point["shapes"]
        ),
        "matmul": matmul_rows,
        "vector": vector_rows,
        "best_matmul_gflops_per_second": max(
            row["best_median_gflops_per_second"] for row in matmul_rows
        ),
        "best_vector_gb_per_second": max(
            row["best_median_gb_per_second"] for row in vector_rows
        ),
    }


def build_report(
    config: MetalLaunchSweepConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the sweep and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalLaunchSweepRuntime``, which
    raises ``KernelMetalUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MetalLaunchSweepConfig):
        raise TypeError("config must be a MetalLaunchSweepConfig")
    backend = MetalLaunchSweepRuntime() if backend is None else backend
    real = type(backend) is MetalLaunchSweepRuntime
    measuring = real and timer_name(clock) == TIMER
    before = _host_load(measuring)
    timing = config.timing
    budget = RunBudget(timing.max_runtime_seconds, clock)
    rule = backend.requires_whole_simd_groups
    if type(rule) is not bool:
        raise MetalLaunchSweepError("the runtime must state its SIMD rule as a bool")
    limits = {name: backend.limits[name] for name in PIPELINES}
    matmul_admissions = admit_shapes(
        config.matmul.group_shapes, limits[MATMUL], requires_whole_simd_groups=rule
    )
    vector_admissions = admit_shapes(
        tuple((size,) for size in config.vector.group_sizes),
        limits[VECTOR_ADD],
        requires_whole_simd_groups=rule,
    )
    mapping = [
        check_mapping(item.group_size, order_seed=config.seed)
        for item in matmul_admissions
        if item.admitted
    ]
    budget.check()
    layouts = [
        observe_simd_layout(backend, item)
        for item in (*matmul_admissions, *vector_admissions)
        if item.admitted
    ]
    budget.check()
    matmul_cases = []
    for index, shape in enumerate(config.matmul.correctness_cases):
        matmul_cases.append(
            check_matmul_case(
                backend,
                seed=config.seed,
                index=index,
                shape=shape,
                admissions=matmul_admissions,
            )
        )
        budget.check()
    vector_cases = check_vector_sizes(
        backend,
        seed=config.seed,
        length=config.vector.correctness_length,
        admissions=vector_admissions,
    )
    budget.check()
    common = {
        "seed": config.seed,
        "warmup": timing.warmup,
        "repeats": timing.repeats,
        "clock": clock,
        "budget": budget,
    }
    points: list[SweepPoint] = [
        measure_matmul_point(
            backend,
            index=index,
            shape=size,
            admissions=matmul_admissions,
            target_flops_per_sample=config.matmul.target_flops_per_sample,
            **common,
        )
        for index, size in enumerate(config.matmul.sizes)
    ]
    points += [
        measure_vector_point(
            backend,
            index=index,
            length=length,
            admissions=vector_admissions,
            target_bytes_per_sample=config.vector.target_bytes_per_sample,
            **common,
        )
        for index, length in enumerate(config.vector.lengths)
    ]
    elapsed = budget.check()
    after = _host_load(measuring)
    timers = {point.timer for point in points}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MetalLaunchSweepError("every sample must share one known timer")
    measured = timers == {TIMER} and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    admission = {
        "matmul": [item.to_mapping() for item in matmul_admissions],
        "vector": [item.to_mapping() for item in vector_admissions],
    }
    matmul = [point.to_mapping() for point in points if point.kernel == MATMUL]
    vector = [point.to_mapping() for point in points if point.kernel == VECTOR_ADD]
    matmul_records = [case.to_mapping() for case in matmul_cases]
    vector_records = [case.to_mapping() for case in vector_cases]
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
        "environment": host_environment(backend.torch_version),
        "shaders": {
            "source": SWEEP_SOURCE,
            "source_sha256": source_sha256(),
            "kernels": kernel_bindings(),
            "day_73_source_sha256": day_73_source_sha256(),
            "day_73_kernel": KERNELS[VECTOR_ADD].to_mapping(),
        },
        "pipelines": {name: limits[name].to_mapping() for name in PIPELINES},
        "dispatch": {
            "grid": GRID_RULE,
            "requires_whole_simd_groups": rule,
            "rules": list(REJECTION_REASONS),
        },
        "admission": admission,
        "mapping": {
            "case": list(MAPPING_CASE),
            "checks": [check.to_mapping() for check in mapping],
        },
        "simd_layout": [layout.to_mapping() for layout in layouts],
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": REFERENCE,
            "bound": BOUND,
            "matmul": matmul_records,
            "vector_add": vector_records,
        },
        "benchmark": {
            "device": DEVICE,
            "synchronization": SYNCHRONIZATION,
            "turn_order": TURN_ORDER,
            "matmul": matmul,
            "vector": vector,
        },
        "run": {
            "max_runtime_ns": timing.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_ns": sum(
                sum(shape.samples_ns) for point in points for shape in point.timings
            ),
            "load_average_before": before,
            "load_average_after": after,
        },
        "summary": _summary(
            label,
            admission,
            matmul_records,
            vector_records,
            matmul,
            vector,
            (len(mapping), len(layouts)),
        ),
    }


def measure_report(
    config: MetalLaunchSweepConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the sweep and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalLaunchSweepRuntime``, or any clock other
    than the host timer, makes the report synthetic.
    """

    return seal_report(build_report(config, backend=backend, clock=clock))
