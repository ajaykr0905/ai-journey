"""Run every case and sweep point under one budget, then seal a labeled report.

One run compiles the kernels, records each pipeline's limits and the
contraction probe, runs every correctness case, and times every sweep point.
One ``RunBudget`` on the run's clock covers the probe, the cases and the
sweep; it starts after the kernels compile, so the time to import torch on a
busy host does not count. The report records the config and its digest, the
shader source and its digest, the host environment without identifiers, the
planned work, the run's elapsed and timed nanoseconds, and a summary.

Labels keep the evidence honest. Only the real ``MetalRuntime`` timed with
``time.perf_counter_ns`` makes a measurement: scope
``apple_gpu_metal_this_host``, label ``host-specific Apple GPU (Metal)
measurement``, and a boundary that rules out CUDA, NVIDIA, distributed and
production claims. Any other backend or clock makes the report synthetic:
backend or timer ``injected``, ``measured: false``, scope
``synthetic_backend_or_clock``, and a boundary that denies any measurement.

The SHA-256 seal follows ``memory_access_report``: canonical JSON of every
field but the digest. It detects drift and tampering; it is not a signature
and cannot prove that a timing was observed.
"""

from __future__ import annotations

import platform
import re
import time
from collections.abc import Callable

import numpy

from .kernel_metal_benchmark import (
    NS_PER_SECOND,
    BenchmarkPoint,
    RunBudget,
    benchmark_rgb_to_gray,
    benchmark_vector_add,
)
from .kernel_metal_config import KernelMetalConfig
from .kernel_metal_correctness import (
    METAL_KERNEL,
    GrayCase,
    check_rgb_to_gray,
    check_vector_add,
)
from .kernel_metal_geometry import KernelMetalError
from .kernel_metal_reference import (
    FLOAT32_SENTINEL_BITS,
    GRAY_WEIGHTS,
    MAX_LEVEL_DIFFERENCE,
    UINT8_SENTINEL,
)
from .kernel_metal_runtime import BACKEND, MetalRuntime
from .kernel_metal_source import (
    KERNELS,
    PROBE_EXPRESSION,
    PROBE_FUSED_RESULT,
    PROBE_INPUTS,
    PROBE_UNFUSED_RESULT,
    SHADER_SOURCE,
    source_sha256,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "kernel_metal_correctness_and_benchmark"
MEASUREMENT_SCOPE = "apple_gpu_metal_this_host"
SYNTHETIC_SCOPE = "synthetic_backend_or_clock"
MEASUREMENT_LABEL = "host-specific Apple GPU (Metal) measurement"
SYNTHETIC_LABEL = "synthetic timing, not a measurement"
INJECTED_BACKEND = "injected"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
ROUNDING_RULE = (
    "y = 0.2989 R + 0.5870 G + 0.1140 B in float32, left to right; "
    "gray = clamp(floor(y + 0.5), 0, 255): round half up, then clamp to 0..255"
)
TORCH_EQUIVALENTS = {
    "vector_add": "torch.add(a, b, out=out)",
    "rgb_to_gray": (
        "p = rgb.to(float32); y = p[..., 0] * 0.2989 + p[..., 1] * 0.5870; "
        "y = y + p[..., 2] * 0.1140; floor(y + 0.5).clamp_(0, 255).to(uint8)"
    ),
}
SYNCHRONIZATION = (
    "torch.mps.synchronize() before the first and before the second clock "
    "reading of every sample; untimed warmups first; the two sides alternate "
    "and the order flips every repeat"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report. The Metal kernels came from torch.mps.compile_shader, and "
        "their PyTorch equivalents ran on the MPS device. The timings are "
        "host-specific Apple GPU (Metal) measurements. No CUDA kernel ran, and "
        "no NVIDIA GPU was used. Effective bandwidth counts the bytes that each "
        "kernel must read and write, not the hardware traffic. These timings do "
        "not measure or predict CUDA, NVIDIA, distributed or production "
        "performance."
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
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration and rate in it is synthetic. They are not measurements of "
        "this host or of any other host. The correctness checks ran on the "
        "outputs that the backend returned. No CUDA kernel ran, and no NVIDIA "
        "GPU was used."
    ),
}
ENVIRONMENT_FIELDS = (
    "python_version",
    "python_implementation",
    "numpy_version",
    "torch_version",
    "platform_system",
    "platform_machine",
    "macos_version",
)
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")


def _token(value: object) -> str:
    if isinstance(value, str) and _TOKEN.fullmatch(value) is not None:
        return value
    return "unknown"


def host_environment(torch_version: object) -> dict[str, str]:
    """Describe the interpreter, libraries and platform without naming the host.

    Only version and platform tokens are copied; host names, user names,
    paths, serial numbers and the GPU's model name are never recorded, and
    any value that is not a short token becomes ``unknown``.
    """

    return {
        "python_version": _token(platform.python_version()),
        "python_implementation": _token(platform.python_implementation()),
        "numpy_version": _token(numpy.__version__),
        "torch_version": _token(torch_version),
        "platform_system": _token(platform.system()),
        "platform_machine": _token(platform.machine()),
        "macos_version": _token(platform.mac_ver()[0]),
    }


def _shaders(backend: object, observed: float) -> dict[str, object]:
    return {
        "source": SHADER_SOURCE,
        "source_sha256": source_sha256(),
        "kernels": [
            {**spec.to_mapping(), "limits": backend.limits[name].to_mapping()}
            for name, spec in KERNELS.items()
        ],
        "contraction_probe": {
            "expression": PROBE_EXPRESSION,
            "inputs": list(PROBE_INPUTS),
            "unfused_result": PROBE_UNFUSED_RESULT,
            "fused_result": PROBE_FUSED_RESULT,
            "observed": observed,
            "fused": observed == PROBE_FUSED_RESULT,
        },
    }


def _largest(points: tuple[BenchmarkPoint, ...]) -> dict[str, object]:
    mapping = points[-1].to_mapping()
    shape = {
        key: mapping[key] for key in ("length", "height", "width") if key in mapping
    }
    return {
        **shape,
        "metal_kernel_gb_per_second": mapping[METAL_KERNEL]["median_gb_per_second"],
        "torch_gb_per_second": mapping["torch"]["median_gb_per_second"],
        "torch_to_metal_time_ratio": mapping["torch_to_metal_time_ratio"],
    }


def _summary(
    label: str,
    vector_cases: int,
    gray_cases: tuple[GrayCase, ...],
    fused: bool,
    vector_points: tuple[BenchmarkPoint, ...],
    gray_points: tuple[BenchmarkPoint, ...],
    repeats: int,
) -> dict[str, object]:
    points = (*vector_points, *gray_points)
    return {
        "label": label,
        "correctness_cases": vector_cases + len(gray_cases),
        "vector_add_exact_cases": vector_cases,
        "rgb_to_gray_max_abs_difference_levels": max(
            case.max_abs_difference for case in gray_cases
        ),
        "rgb_to_gray_metal_mismatched_pixels": sum(
            case.outputs[0].comparison.mismatched_pixels for case in gray_cases
        ),
        "contraction_fused": fused,
        "benchmark_points": len(points),
        "timed_samples": 2 * len(points) * repeats,
        "vector_add_largest": _largest(vector_points),
        "rgb_to_gray_largest": _largest(gray_points),
        "peak_metal_kernel_gb_per_second": max(
            point.to_mapping()[METAL_KERNEL]["median_gb_per_second"] for point in points
        ),
    }


def build_report(
    config: KernelMetalConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the lab and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalRuntime``, which raises
    ``KernelMetalUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, KernelMetalConfig):
        raise TypeError("config must be a KernelMetalConfig")
    timing = config.benchmark
    backend = MetalRuntime() if backend is None else backend
    budget = RunBudget(timing.max_runtime_seconds, clock)
    observed = backend.contraction_probe()
    if type(observed) is not float or observed not in (
        PROBE_UNFUSED_RESULT,
        PROBE_FUSED_RESULT,
    ):
        raise KernelMetalError(
            f"the contraction probe returned {observed!r}, neither the fused nor "
            "the unfused result"
        )
    group = config.launch.vector_add_group_size
    vector_cases = []
    for index, length in enumerate(config.correctness.vector_add_lengths):
        vector_cases.append(
            check_vector_add(
                backend, seed=config.seed, index=index, length=length, group_size=group
            )
        )
        budget.check()
    gray_cases = []
    for index, shape, pattern in config.correctness.gray_cases():
        gray_cases.append(
            check_rgb_to_gray(
                backend,
                seed=config.seed,
                index=index,
                shape=shape,
                group_size=config.launch.rgb_to_gray_group_size,
                pattern=pattern,
            )
        )
        budget.check()
    common = {
        "seed": config.seed,
        "warmup": timing.warmup,
        "repeats": timing.repeats,
        "target_bytes_per_sample": timing.target_bytes_per_sample,
        "clock": clock,
        "budget": budget,
    }
    vector_points = benchmark_vector_add(
        backend, lengths=timing.vector_add_lengths, group_size=group, **common
    )
    gray_points = benchmark_rgb_to_gray(
        backend,
        shapes=timing.rgb_to_gray_shapes,
        group_size=config.launch.rgb_to_gray_group_size,
        **common,
    )
    elapsed = budget.check()
    timings = [
        timing
        for point in (*vector_points, *gray_points)
        for timing in (point.metal, point.torch)
    ]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise KernelMetalError("every sample must share one known timer")
    real = type(backend) is MetalRuntime
    measured = timers == {TIMER} and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
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
        "shaders": _shaders(backend, observed),
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": "NumPy float32 on the CPU, rebuilt from the seed",
            "rounding": ROUNDING_RULE,
            "weights": list(GRAY_WEIGHTS),
            "bound_levels": MAX_LEVEL_DIFFERENCE,
            "sentinels": {
                "float32_bits": FLOAT32_SENTINEL_BITS,
                "uint8": UINT8_SENTINEL,
            },
            "vector_add": [case.to_mapping() for case in vector_cases],
            "rgb_to_gray": [case.to_mapping() for case in gray_cases],
        },
        "benchmark": {
            "device": DEVICE,
            "torch_equivalents": dict(TORCH_EQUIVALENTS),
            "synchronization": SYNCHRONIZATION,
            "vector_add": [point.to_mapping() for point in vector_points],
            "rgb_to_gray": [point.to_mapping() for point in gray_points],
        },
        "run": {
            "max_runtime_ns": timing.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_ns": sum(sum(item.samples_ns) for item in timings),
        },
        "summary": _summary(
            label,
            len(vector_cases),
            tuple(gray_cases),
            observed == PROBE_FUSED_RESULT,
            vector_points,
            gray_points,
            timing.repeats,
        ),
    }


def measure_report(
    config: KernelMetalConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalRuntime``, or any clock other than the host
    timer, makes the report synthetic.
    """

    return seal_report(build_report(config, backend=backend, clock=clock))
