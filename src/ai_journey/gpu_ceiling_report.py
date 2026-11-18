"""Run every check and sweep under one budget, then seal a labeled report.

One run compiles the kernels, records each pipeline's limits, checks every
kernel on a guarded launch, times the FMA, stream and bank sweeps, and
derives the ceilings. One ``RunBudget`` on the run's clock covers the checks
and the sweeps; it starts after the kernels compile, so the time to import
torch on a busy host does not count. The report records the config and its
digest, the shader source and its digest, the host environment without
identifiers, the 1, 5 and 15 minute load averages that ``uptime`` prints,
read before and after the run, the planned work, and a summary.

Labels keep the evidence honest. Only the real ``CeilingRuntime`` timed with
``time.perf_counter_ns`` makes a measurement: scope
``apple_gpu_metal_this_host``, label ``host-specific Apple GPU (Metal)
measurement``, and a boundary that rules out vendor peaks and CUDA, NVIDIA,
distributed and production claims. Any other backend or clock makes the
report synthetic: backend or timer ``injected``, ``measured: false``, scope
``synthetic_backend_or_clock``, and a boundary that denies any measurement.

The SHA-256 seal follows ``memory_access_report``: canonical JSON of every
field but the digest. It detects drift and tampering; it is not a signature
and cannot prove that a timing was observed.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

from .gpu_ceiling_analysis import (
    bandwidth_ceiling,
    bank_analysis,
    fma_ceiling,
    fma_scaling,
    ridge_point,
)
from .gpu_ceiling_benchmark import benchmark_banks, benchmark_fma, benchmark_streams
from .gpu_ceiling_config import GpuCeilingConfig
from .gpu_ceiling_correctness import check_bank, check_fma, check_stream
from .gpu_ceiling_reference import FMA_COEFFICIENTS, MAX_REDUCE_VALUE
from .gpu_ceiling_runtime import STREAM_GUARD_FLOAT4, CeilingRuntime
from .gpu_ceiling_source import (
    BANK_ROW_WORDS,
    KERNELS,
    SHADER_SOURCE,
    STREAM_KERNELS,
    GpuCeilingError,
    source_sha256,
)
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .kernel_metal_report import host_environment
from .kernel_metal_runtime import BACKEND
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "gpu_ceiling_measurement"
MEASUREMENT_SCOPE = "apple_gpu_metal_this_host"
SYNTHETIC_SCOPE = "synthetic_backend_or_clock"
MEASUREMENT_LABEL = "host-specific Apple GPU (Metal) measurement"
SYNTHETIC_LABEL = "synthetic timing, not a measurement"
INJECTED_BACKEND = "injected"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
LOAD_AVERAGE_SOURCE = (
    "os.getloadavg(): the 1, 5 and 15 minute load averages that uptime prints, "
    "read before the checks and after the last sample"
)
REFERENCE = (
    "NumPy on the CPU, rebuilt from the seed: exact fused FMA chains, copies, "
    "exact integer partials and wrapping lane sums"
)
FMA_RULE = (
    "x = fma(x, a, b) with a = (2^24 - 4173) * 2^-24 and b = 12561923 * 2^-35, "
    "from starts m * 2^-23 in [1, 2); float64 holds each exact x * a + b, so one "
    "rounding to float32 equals the fused result, and x stays in [1, 2)"
)
CEILING_DEFINITION = (
    "Each ceiling is the best median rate that these kernels reached on this "
    "host in this run. It is not a vendor peak and not a hardware limit."
)
SYNCHRONIZATION = (
    "torch.mps.synchronize() before the first and before the second clock "
    "reading of every sample; untimed warmups first; samples rotate over the "
    "points of a sweep and the order reverses every repeat"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "vendor_peak_used": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report. The kernels came from torch.mps.compile_shader. Each "
        "ceiling is the best median rate that these kernels reached in this "
        "run. It is not a vendor peak and not a hardware limit. The timings are "
        "host-specific Apple GPU (Metal) measurements. No CUDA kernel ran, and "
        "no NVIDIA GPU was used. Bandwidth counts the bytes that each kernel "
        "must read and write, not the hardware traffic. These timings do not "
        "measure or predict CUDA, NVIDIA, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "vendor_peak_used": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate and ceiling in it is synthetic. They are not "
        "measurements of this host or of any other host. The correctness "
        "checks ran on the outputs that the backend returned. No CUDA kernel "
        "ran, and no NVIDIA GPU was used."
    ),
}


def read_load_average(
    reader: Callable[[], tuple[float, float, float]],
) -> list[float]:
    """Read the three load averages; refuse anything but three finite numbers."""

    values = reader()
    if (
        type(values) is not tuple
        or len(values) != 3
        or not all(
            type(value) is float and math.isfinite(value) and value >= 0
            for value in values
        )
    ):
        raise GpuCeilingError("the load average must be three nonnegative floats")
    return list(values)


def _shaders(backend: object) -> dict[str, object]:
    return {
        "source": SHADER_SOURCE,
        "source_sha256": source_sha256(),
        "kernels": [
            {**spec.to_mapping(), "limits": backend.limits[name].to_mapping()}
            for name, spec in KERNELS.items()
        ],
    }


def _correctness(
    config: GpuCeilingConfig, backend: object, budget: RunBudget
) -> dict[str, object]:
    cases = config.correctness
    fma = []
    for chains in config.fma.chain_counts:
        fma.append(
            check_fma(
                backend,
                seed=config.seed,
                chains=chains,
                threads=cases.threads,
                group_size=config.fma.group_size,
                fma_steps=cases.fma_steps,
            )
        )
        budget.check()
    stream = []
    for kernel in STREAM_KERNELS:
        stream.append(
            check_stream(
                backend,
                seed=config.seed,
                kernel=kernel,
                float4_count=cases.stream_float4,
                float4_per_thread=config.bandwidth.per_thread(kernel),
                group_size=config.bandwidth.group_size,
            )
        )
        budget.check()
    bank = []
    for row_words in BANK_ROW_WORDS:
        for stride in config.bank.strides:
            bank.append(
                check_bank(
                    backend,
                    row_words=row_words,
                    stride=stride,
                    threads=cases.threads,
                    group_size=config.bank.group_size,
                    trips=cases.bank_trips,
                )
            )
            budget.check()
    return {
        "reference": REFERENCE,
        "fma_rule": FMA_RULE,
        "fma_coefficients": list(FMA_COEFFICIENTS),
        "reduce_max_value": MAX_REDUCE_VALUE,
        "sentinel_bits": FLOAT32_SENTINEL_BITS,
        "stream_guard_float4": STREAM_GUARD_FLOAT4,
        "fma": [case.to_mapping() for case in fma],
        "stream": [case.to_mapping() for case in stream],
        "bank": [case.to_mapping() for case in bank],
    }


def _summary(
    label: str,
    correctness: dict[str, object],
    benchmark: dict[str, object],
    ceilings: dict[str, object],
    timed_samples: int,
    load: dict[str, object],
) -> dict[str, object]:
    fma, bandwidth, bank = ceilings["fma"], ceilings["bandwidth"], ceilings["bank"]
    return {
        "label": label,
        "correctness_cases": sum(
            len(correctness[family]) for family in ("fma", "stream", "bank")
        ),
        "fma_points": len(benchmark["fma"]),
        "stream_points": len(benchmark["stream"]),
        "bank_points": len(benchmark["bank"]),
        "timed_samples": timed_samples,
        "fma_ceiling_gflops_per_second": fma["gflops_per_second"],
        "fma_ceiling_chains": fma["chains"],
        "fma_ceiling_fma_steps": fma["fma_steps"],
        "fma_fixed_share_of_ceiling_call": fma["fixed_share_of_ceiling_call"],
        "fma_scales_with_work": fma["scales_with_work"],
        "bandwidth_ceiling_gb_per_second": bandwidth["gb_per_second"],
        "bandwidth_ceiling_kernel": bandwidth["kernel"],
        "bandwidth_ceiling_buffer_bytes": bandwidth["buffer_bytes"],
        "ridge_point_flops_per_byte": ceilings["ridge_point_flops_per_byte"],
        "bank_finding": bank["finding"],
        "bank_largest_unpadded_ratio": bank["largest_unpadded_ratio"],
        "bank_largest_padded_ratio": bank["largest_padded_ratio"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def build_report(
    config: GpuCeilingConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and assemble the unsealed report.

    ``backend=None`` compiles the real ``CeilingRuntime``, which raises
    ``GpuCeilingUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, GpuCeilingConfig):
        raise TypeError("config must be a GpuCeilingConfig")
    timing = config.timing
    backend = CeilingRuntime() if backend is None else backend
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    correctness = _correctness(config, backend, budget)
    common = {
        "warmup": timing.warmup,
        "repeats": timing.repeats,
        "clock": clock,
        "budget": budget,
    }
    fma_points = benchmark_fma(
        backend,
        seed=config.seed,
        threads=config.fma.threads,
        group_size=config.fma.group_size,
        chain_counts=config.fma.chain_counts,
        fma_steps=config.fma.fma_steps,
        **common,
    )
    stream_points = benchmark_streams(
        backend,
        seed=config.seed,
        group_size=config.bandwidth.group_size,
        float4_counts=config.bandwidth.float4_counts,
        copy_float4_per_thread=config.bandwidth.copy_float4_per_thread,
        reduce_float4_per_thread=config.bandwidth.reduce_float4_per_thread,
        target_bytes_per_sample=config.bandwidth.target_bytes_per_sample,
        **common,
    )
    bank_points = benchmark_banks(
        backend,
        threads=config.bank.threads,
        group_size=config.bank.group_size,
        row_words=BANK_ROW_WORDS,
        strides=config.bank.strides,
        trips=config.bank.trips,
        **common,
    )
    elapsed = budget.check()
    load = {
        "source": LOAD_AVERAGE_SOURCE,
        "before": before,
        "after": read_load_average(load_average),
    }
    points = (*fma_points, *stream_points, *bank_points)
    timers = {point.timing.timer for point in points}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise GpuCeilingError("every sample must share one known timer")
    real = type(backend) is CeilingRuntime
    measured = timers == {TIMER} and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    benchmark = {
        "device": DEVICE,
        "synchronization": SYNCHRONIZATION,
        "fma": [point.to_mapping() for point in fma_points],
        "stream": [point.to_mapping() for point in stream_points],
        "bank": [point.to_mapping() for point in bank_points],
    }
    fma = fma_ceiling(benchmark["fma"])
    bandwidth = bandwidth_ceiling(benchmark["stream"])
    ceilings = {
        "definition": CEILING_DEFINITION,
        "fma": fma,
        "fma_scaling": fma_scaling(benchmark["fma"]),
        "bandwidth": bandwidth,
        "ridge_point_flops_per_byte": ridge_point(fma, bandwidth),
        "bank": bank_analysis(benchmark["bank"]),
    }
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
        "host_load": load,
        "shaders": _shaders(backend),
        "planned_work": config.planned_work().to_mapping(),
        "correctness": correctness,
        "benchmark": benchmark,
        "ceilings": ceilings,
        "run": {
            "max_runtime_ns": timing.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_ns": sum(sum(point.timing.samples_ns) for point in points),
        },
        "summary": _summary(
            label,
            correctness,
            benchmark,
            ceilings,
            timing.repeats * len(points),
            load,
        ),
    }


def measure_report(
    config: GpuCeilingConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``CeilingRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(config, backend=backend, clock=clock, load_average=load_average)
    )
