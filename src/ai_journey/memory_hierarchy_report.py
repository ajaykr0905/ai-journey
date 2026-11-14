"""Run every section under one budget and seal the CPU measurement report.

One run measures, in order, the STREAM working-set sweep, the access-pattern
contrast and the matmul compute ceiling, each with the config's dtype, seed,
warmups and repeats, all against one ``RunBudget`` on the run's clock. It
then derives the roofline and the Little's-law view from those samples and
records the planned work, the host environment, the run's elapsed and timed
nanoseconds, and a summary. Only samples taken with ``time.perf_counter_ns``
make a measurement: a report built with an injected clock is labeled
synthetic (timer ``injected``, ``measured: false``, scope
``synthetic_injected_clock`` and a boundary that denies any host
measurement), although its kernels and checks really ran.

The SHA-256 seal follows ``memory_access_report``: canonical JSON (sorted
keys, compact separators, ASCII escapes) of every field but the digest. It
detects drift and tampering; it is not a signature and cannot prove that a
timing was observed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report
from .memory_hierarchy_access import (
    INDEX_BYTES,
    access_contrast,
    measure_access_patterns,
)
from .memory_hierarchy_config import MemoryHierarchyConfig
from .memory_hierarchy_environment import host_environment
from .memory_hierarchy_littles_law import littles_law_section
from .memory_hierarchy_matmul import measure_matmul
from .memory_hierarchy_roofline import measured_roofline
from .memory_hierarchy_stream import (
    STREAM_SCALAR,
    MemoryHierarchyError,
    stream_kernel,
)
from .memory_hierarchy_sweep import curve_summary, measure_stream_sweep, stream_curves
from .memory_hierarchy_timing import NS_PER_SECOND, RunBudget

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "memory_hierarchy_cpu_measurement"
MEASUREMENT_SCOPE = "cpu_wall_clock_this_host"
SYNTHETIC_SCOPE = "synthetic_injected_clock"
REPORT_DIGEST_FIELD = DIGEST_FIELD

MEASUREMENT_BOUNDARY = {
    "cpu_measurement_this_host": True,
    "gpu_executed": False,
    "cuda_kernel_executed": False,
    "hardware_counters_read": False,
    "cache_sizes_detected": False,
    "latency_measured": False,
    "portable_performance_claim": False,
    "statement": (
        "Wall-clock timings of NumPy kernels on the CPU host that built this "
        "report. Bandwidths count the operand bytes each NumPy call reads and "
        "writes, not hardware traffic, and include NumPy call overhead. The "
        "matmul runs on NumPy's BLAS, which may use several threads; the other "
        "kernels use one. Little's-law rows multiply measured rates by a stated "
        "latency that was not measured. No cache size is detected or claimed, "
        "and nothing here measures or predicts GPU, CUDA, distributed, or "
        "production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "cpu_measurement_this_host": False,
    "synthetic_timing": True,
    "gpu_executed": False,
    "cuda_kernel_executed": False,
    "hardware_counters_read": False,
    "cache_sizes_detected": False,
    "latency_measured": False,
    "portable_performance_claim": False,
    "statement": (
        "Durations came from a clock injected by the caller instead of "
        "time.perf_counter_ns, so every duration and rate here is synthetic and "
        "not a measurement of this or any host. The NumPy kernels and their "
        "checks did run on the CPU; no GPU or CUDA kernel was executed."
    ),
}


def _summary(
    stream_section: dict[str, object],
    access_section: dict[str, object],
    roofline: dict[str, object],
    samples: int,
) -> dict[str, object]:
    triad = stream_section["curve_summaries"]["triad"]
    return {
        "stream_measurements": len(stream_section["measurements"]),
        "access_patterns": len(access_section["measurements"]),
        "timed_samples": samples,
        "triad_peak_working_set_bytes": triad["peak_working_set_bytes"],
        "triad_peak_gb_per_second": triad["peak_median_gb_per_second"],
        "triad_largest_working_set_bytes": triad["largest_working_set_bytes"],
        "triad_largest_gb_per_second": triad["largest_median_gb_per_second"],
        "random_to_sequential_time_ratio": access_section["contrast"][
            "random_to_sequential_time_ratio"
        ],
        "compute_ceiling_flops_per_second": roofline["compute_ceiling"][
            "flops_per_second"
        ],
        "ridge_point_flops_per_byte": roofline["ridge_point_flops_per_byte"],
    }


def build_report(
    config: MemoryHierarchyConfig, *, clock: Callable[[], int] = time.perf_counter_ns
) -> dict[str, object]:
    """Measure every section under one budget and assemble the unsealed report.

    Sections run one after another and release their arrays, so the run's
    memory peak is the largest section's, as ``planned_work`` bounded it.
    """

    if not isinstance(config, MemoryHierarchyConfig):
        raise TypeError("config must be a MemoryHierarchyConfig")
    controls = config.timing
    budget = RunBudget(controls.max_runtime_seconds, clock)
    common = {
        "dtype": config.dtype,
        "seed": config.seed,
        "warmup": controls.warmup,
        "repeats": controls.repeats,
        "clock": clock,
        "budget": budget,
    }
    stream = measure_stream_sweep(
        kernels=config.stream.kernels,
        elements=config.stream.elements,
        target_bytes_per_sample=controls.target_bytes_per_sample,
        **common,
    )
    access = measure_access_patterns(
        elements=config.access.elements,
        stride_elements=config.access.stride_elements,
        target_bytes_per_sample=controls.target_bytes_per_sample,
        **common,
    )
    matmul = measure_matmul(
        m=config.matmul.m, n=config.matmul.n, k=config.matmul.k, **common
    )
    elapsed = budget.check()
    timings = [item.timing for item in (*stream, *access, matmul)]
    timers = {timing.timer for timing in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MemoryHierarchyError("every measurement must share one known timer")
    measured = timers == {TIMER}
    curves = stream_curves(stream)
    stream_section = {
        "dtype": config.dtype,
        "element_bytes": config.element_bytes,
        "scalar": STREAM_SCALAR,
        "accounting": [
            stream_kernel(name).to_mapping() for name in config.stream.kernels
        ],
        "measurements": [item.to_mapping() for item in stream],
        "curves": curves,
        "curve_summaries": {
            name: curve_summary(curve) for name, curve in curves.items()
        },
    }
    access_section = {
        "dtype": config.dtype,
        "elements": config.access.elements,
        "stride_elements": config.access.stride_elements,
        "stride_bytes": config.access.stride_elements * config.element_bytes,
        "source_bytes": config.access.elements * config.element_bytes,
        "index_bytes": INDEX_BYTES,
        "measurements": [item.to_mapping() for item in access],
        "contrast": access_contrast(access),
    }
    roofline = measured_roofline(stream, matmul)
    samples = sum(timing.repeats for timing in timings)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "measured": measured,
        "measurement_scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
        "evidence_boundary": dict(
            MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY
        ),
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "timer": timers.pop(),
        "percentile_method": PERCENTILE_METHOD,
        "environment": host_environment(),
        "planned_work": config.planned_work().to_mapping(),
        "stream": stream_section,
        "access": access_section,
        "matmul": matmul.to_mapping(),
        "roofline": roofline,
        "littles_law": littles_law_section(config.littles_law, stream, access),
        "run": {
            "max_runtime_ns": controls.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_ns": sum(sum(timing.samples_ns) for timing in timings),
        },
        "summary": _summary(stream_section, access_section, roofline, samples),
    }


def measure_report(
    config: MemoryHierarchyConfig, *, clock: Callable[[], int] = time.perf_counter_ns
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any ``clock`` other than the host timer makes the report synthetic.
    """

    return seal_report(build_report(config, clock=clock))
