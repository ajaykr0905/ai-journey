"""Run every case and sweep point under one budget, then seal a labeled report.

One run compiles both kernels, records each pipeline's limits, runs every
correctness case, and times every sweep point. One ``RunBudget`` on the
run's clock covers the cases and the sweep; it starts after the kernels
compile, so the time to import torch on a busy host does not count. The
report records the config and its digest, the shader source and its digest,
the host environment without identifiers, the planned work, the run's
elapsed and timed nanoseconds, and a summary that compares each naive
kernel with the measured ceiling.

Labels keep the evidence honest. Only the real ``MatmulMetalRuntime`` timed
with ``time.perf_counter_ns`` makes a measurement: scope
``apple_gpu_metal_this_host``, label ``host-specific Apple GPU (Metal)
measurement``, and a boundary that rules out CUDA, NVIDIA, vendor-peak,
distributed and production claims. Any other backend or clock makes the
report synthetic: backend or timer ``injected``, ``measured: false``, scope
``synthetic_backend_or_clock``, and a boundary that denies any measurement.
The scopes and labels are the Day 73 ones, so both Metal labs read alike.

The SHA-256 seal follows ``memory_access_report``: canonical JSON of every
field but the digest. It detects drift and tampering; it is not a signature
and cannot prove that a timing was observed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
    _token,
    host_environment,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_benchmark import (
    AGREEMENT,
    MatmulPoint,
    benchmark_sweep,
    ceiling_summary,
)
from .matmul_metal_config import MatmulMetalConfig
from .matmul_metal_correctness import IMPLEMENTATIONS, MatmulCase, check_case
from .matmul_metal_mapping import MatmulMetalError
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    REFERENCE,
    RELATIVE_ERROR,
)
from .matmul_metal_runtime import SIDES, TORCH_MATMUL, MatmulMetalRuntime
from .matmul_metal_source import (
    KERNELS,
    LOOP_STATEMENT,
    SHADER_SOURCE,
    source_sha256,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_metal_correctness_and_gflops"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
MATMUL_PRECISION_FIELD = "torch_float32_matmul_precision"
REPLAY = (
    f"each naive output equals the CPU replay of `{LOOP_STATEMENT}` for k = 0 "
    "to K - 1, one float32 rounding per step, bit for bit"
)
FLOPS_PER_CALL = "2 * M * N * K: one multiply and one add per term of each output"
GFLOPS = "FLOPs of one sample divided by the median sample duration in ns"
TORCH_EQUIVALENT = "torch.matmul(a, b, out=c) on the MPS device, float32"
SYNCHRONIZATION = (
    "torch.mps.synchronize() before the first and before the second clock "
    "reading of every sample; untimed warmups first; the three sides rotate "
    "and the order shifts by one side every repeat"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report. The naive kernels came from torch.mps.compile_shader, "
        "and torch.matmul ran on the MPS device. The timings are host-specific "
        "Apple GPU (Metal) measurements. No CUDA kernel ran, and no NVIDIA GPU "
        "was used. The measured ceiling is the best median rate of torch.matmul "
        "in this run, not a hardware or vendor peak. These timings do not "
        "measure or predict CUDA, NVIDIA, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "vendor_peak_claimed": False,
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


def environment(backend: object) -> dict[str, str]:
    """Describe the host as the Day 73 lab does, plus PyTorch's matmul precision.

    ``torch.get_float32_matmul_precision()`` decides whether PyTorch may trade
    float32 accuracy for speed, so a report of float32 rates names it.
    """

    return {
        **host_environment(backend.torch_version),
        MATMUL_PRECISION_FIELD: _token(backend.float32_matmul_precision),
    }


def _shaders(backend: object) -> dict[str, object]:
    return {
        "source": SHADER_SOURCE,
        "source_sha256": source_sha256(),
        "kernels": [
            {**spec.to_mapping(), "limits": backend.limits[name].to_mapping()}
            for name, spec in KERNELS.items()
        ],
    }


def _summary(
    label: str,
    cases: tuple[MatmulCase, ...],
    points: tuple[MatmulPoint, ...],
    repeats: int,
) -> dict[str, object]:
    keys = [f"{name}/{device}" for name, device in IMPLEMENTATIONS]
    largest = max(points, key=lambda point: point.shape.flops)
    mapping = largest.to_mapping()
    ceiling = ceiling_summary(points)
    return {
        "label": label,
        "correctness_cases": len(cases),
        "naive_outputs_replay_exactly": True,
        "max_relative_error": {
            key: max(case.outputs[position].max_relative_error for case in cases)
            for position, key in enumerate(keys)
        },
        "max_bound_ratio": max(
            output.max_relative_error / case.bound
            for case in cases
            for output in case.outputs
        ),
        "torch_cases_with_other_bits": {
            key: sum(case.outputs[position].differing_elements > 0 for case in cases)
            for position, key in enumerate(keys)
            if key.startswith(TORCH_MATMUL)
        },
        "benchmark_points": len(points),
        "timed_samples": len(SIDES) * len(points) * repeats,
        "measured_ceiling_gflops_per_second": ceiling[
            "measured_ceiling_gflops_per_second"
        ],
        "naive_fraction_of_measured_ceiling": {
            name: entry["fraction_of_measured_ceiling"]
            for name, entry in ceiling["naive"].items()
        },
        "largest_point": {
            **largest.shape.to_mapping(),
            "median_gflops_per_second": {
                side: mapping["sides"][side]["median_gflops_per_second"]
                for side in SIDES
            },
            "naive_to_torch_gflops_ratio": mapping["naive_to_torch_gflops_ratio"],
        },
    }


def build_report(
    config: MatmulMetalConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the lab and assemble the unsealed report.

    ``backend=None`` compiles the real ``MatmulMetalRuntime``, which raises
    ``MatmulMetalUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MatmulMetalConfig):
        raise TypeError("config must be a MatmulMetalConfig")
    timing = config.benchmark
    backend = MatmulMetalRuntime() if backend is None else backend
    budget = RunBudget(timing.max_runtime_seconds, clock)
    cases = []
    for index, shape in enumerate(config.correctness.shapes):
        cases.append(
            check_case(
                backend,
                seed=config.seed,
                index=index,
                shape=shape,
                group_size=config.launch.group_size,
            )
        )
        budget.check()
    points = benchmark_sweep(
        backend,
        seed=config.seed,
        shapes=timing.shapes,
        group_size=config.launch.group_size,
        warmup=timing.warmup,
        repeats=timing.repeats,
        target_flops_per_sample=timing.target_flops_per_sample,
        clock=clock,
        budget=budget,
    )
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MatmulMetalError("every sample must share one known timer")
    real = type(backend) is MatmulMetalRuntime
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
        "environment": environment(backend),
        "shaders": _shaders(backend),
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": REFERENCE,
            "replay": REPLAY,
            "relative_error": RELATIVE_ERROR,
            "bound": ERROR_BOUND,
            "gamma": GAMMA,
            "bound_source": BOUND_SOURCE,
            "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
            "cases": [case.to_mapping() for case in cases],
        },
        "benchmark": {
            "device": DEVICE,
            "flops_per_call": FLOPS_PER_CALL,
            "gflops": GFLOPS,
            "torch_equivalent": TORCH_EQUIVALENT,
            "synchronization": SYNCHRONIZATION,
            "agreement": AGREEMENT,
            "points": [point.to_mapping() for point in points],
            "ceiling": ceiling_summary(points),
        },
        "run": {
            "max_runtime_ns": timing.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": elapsed,
            "timed_ns": sum(sum(item.samples_ns) for item in timings),
        },
        "summary": _summary(label, tuple(cases), points, timing.repeats),
    }


def measure_report(
    config: MatmulMetalConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MatmulMetalRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(build_report(config, backend=backend, clock=clock))
