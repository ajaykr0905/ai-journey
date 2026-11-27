"""Run every case and every size under one budget, then seal a Day 82 report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, checks every correctness
case, times every size, reads the load averages again and derives each
verdict and each share of the two ceilings. One ``RunBudget`` on the
run's clock covers the cases and the points, and a run past it stops
without a report.

The warning capture is the Day 76 to 81 capture. The record states whether
torch was imported when the capture began. In a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process.

The report seals the declared plan with its digest, and the config must
name that digest. It records the Metal language version and the support of
M3 and of the bfloat16 PyTorch call as the runtime found them. Only the
real ``MetalTcRuntime`` timed with ``time.perf_counter_ns`` makes a
measurement, and a measurement needs a measured Day 74 ceiling report from
the same interpreter, libraries and platform. Any other backend or clock
makes the report synthetic. The SHA-256 seal follows
``memory_access_report``. It detects drift and tampering; it is not a
signature and cannot prove that a timing was observed.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
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
from .matmul_tiled_report import SWITCH_SOURCE, SYNCHRONIZATION, warning_records
from .matmul_tiled_runtime import MPS_SWITCHES
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_benchmark import timer_name
from .memory_access_report import DIGEST_FIELD, seal_report
from .metal_flash_ceiling import FlashCeiling
from .metal_tc_benchmark import ARITHMETIC, COMPULSORY, benchmark_point
from .metal_tc_config import MetalTcConfig
from .metal_tc_correctness import RULES, check_case
from .metal_tc_plan import SIDES, TORCH_CALL, plan_mapping, plan_sha256
from .metal_tc_reference import (
    BOUND,
    DISTRIBUTION,
    GUARD_BITS,
    REFERENCE,
    RELATIVE_ERROR,
    SENTINEL_BITS,
    TAIL,
    VENDOR_BOUND,
)
from .metal_tc_runtime import SUPPORT_RULE, MetalTcRuntime, admit
from .metal_tc_source import (
    KERNELS,
    LIBRARIES,
    M1,
    M2,
    S1,
    VARIANTS,
    MetalTcError,
    library_sha256,
    library_source,
)
from .metal_tc_speedup import speedup

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "metal_matrix_precision"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
CHECK_RULE = (
    "a timing point checks check_rows rows of C, spread evenly from the first "
    "row to the last, at every column; a correctness case checks every output"
)
_FALSE_CLAIMS = {
    "tensor_cores_used": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "triton_gpu_kernel_executed": False,
    "ptx_executed": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
}
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    **_FALSE_CLAIMS,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made this "
        "report, at the recorded load averages. Apple GPUs have no NVIDIA Tensor "
        "Cores. The simdgroup_matrix path is the matrix path of the Apple GPU, "
        "and this report times it against scalar fma on the same tiles. S1, M1, "
        "M2 and M3 came from torch.mps.compile_shader, and torch.matmul ran on "
        "the MPS device. PyTorch does not report which kernels it ran, so its "
        "numbers name the call, not a kernel. Both ceilings come from a separate "
        "gpu-ceilings run with its own load averages. No hardware counter was "
        "read, and each byte and operation count is a model. No CUDA kernel ran, "
        "no NVIDIA GPU was used, no PTX ran, and no Triton kernel ran on a GPU. "
        "These numbers do not measure or predict CUDA, NVIDIA, Tensor Core, "
        "Triton, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    **_FALSE_CLAIMS,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate, ratio and fraction in it is synthetic. They are not "
        "measurements of this host or of any other host. The correctness checks "
        "ran on the outputs that the backend returned. No CUDA kernel ran, no "
        "NVIDIA GPU or Tensor Core was used, and no Triton kernel ran on a GPU."
    ),
}
RECOMPUTED = (
    "the config, its digest and the planned work",
    "the declared plan, every kernel digest and both library digests",
    "every operand digest in each input type, reference digest, scale digest "
    "and bound digest, from the seed",
    "each launch, and each admission from the recorded limits",
    "each sample statistic, TFLOP/s, GB/s, intensity, roofline, ceiling "
    "fraction, ratio, verdict and held hypothesis, from the raw samples",
    "the run totals, the summary and the SHA-256 seal",
)
GPU_ONLY = (
    "the samples_ns of each side: checked for form only, never observed again",
    "each kernel output digest, max_abs_error, max_bound_ratio and "
    "max_relative_error: max_bound_ratio is checked to be at most 1, and the "
    "errors are checked against the recomputed bounds and scale, not recomputed",
    "identical_repeats, m1_same_bits_as_s1 and torch_float32_same_bits_as_s1: "
    "checked for form and for agreement with the digests and errors only",
    "the Metal language version and the support of M3 and of the bfloat16 "
    "PyTorch call: checked for form only",
    "the pipeline limits: checked for form, then used to rebuild the admissions",
    "the PyTorch side checks, the warnings, the MPS switches and the load "
    "averages: checked for form only",
)


def verification_scope() -> dict[str, list[str]]:
    """State what the CPU verifier recomputes and what only the GPU run saw."""

    return {"recomputed": list(RECOMPUTED), "gpu_only": list(GPU_ONLY)}


def _shaders(backend: object, admissions: dict) -> dict[str, object]:
    return {
        "libraries": {name: library_sha256(name) for name in LIBRARIES},
        "kernels": [
            {
                **kernel.to_mapping(),
                "limits": backend.limits[name].to_mapping()
                if name in backend.limits
                else None,
            }
            for name, kernel in KERNELS.items()
        ],
        "admissions": [
            admissions[name].to_mapping() for name in KERNELS if name in admissions
        ],
    }


def _support(backend: object) -> dict[str, object]:
    return {
        "rule": SUPPORT_RULE,
        "metal_language_version": backend.metal_version,
        "sides": {
            name: record.to_mapping() for name, record in backend.support.items()
        },
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled other libraries than the registry's."""

    if type(backend) is MetalTcRuntime and backend.library_sources != {
        name: library_source(name) for name in LIBRARIES
    }:
        raise MetalTcError("the runtime compiled other libraries than the registry")


def _require_ceiling(ceiling: object, measured: bool, environment: dict) -> None:
    if not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    if measured and not ceiling.measured:
        raise MetalTcError("a measured run needs a measured ceiling report")
    if measured and not ceiling.same_environment(environment):
        raise MetalTcError(
            "the ceiling report comes from another interpreter, library or platform"
        )


def summarize(
    label: str,
    cases: list[dict[str, object]],
    support: dict[str, object],
    points: list[dict[str, object]],
    gains: dict[str, object],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    """Reduce a run to the facts that a reader checks first."""

    ran = [variant for variant in VARIANTS if variant in cases[0]["outputs"]]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "metal_language_version": support["metal_language_version"],
        "supported": {
            name: record["supported"] for name, record in support["sides"].items()
        },
        "max_bound_ratio": {
            variant: max(case["outputs"][variant]["max_bound_ratio"] for case in cases)
            for variant in ran
        },
        "max_relative_error": {
            variant: max(
                case["outputs"][variant]["max_relative_error"] for case in cases
            )
            for variant in ran
        },
        "identical_repeats": {
            variant: all(
                case["outputs"][variant]["identical_repeats"] for case in cases
            )
            for variant in ran
        },
        "m1_same_bits_as_s1_cases": sum(case["m1_same_bits_as_s1"] for case in cases),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "index": row["index"],
                "size": row["size"],
                "tflops_per_second": {
                    side: entry["tflops_per_second"]
                    for side, entry in row["rates"].items()
                },
                "fraction_of_fma": {
                    side: entry["fraction_of_fma"]
                    for side, entry in row["rates"].items()
                },
                "verdicts": {
                    step["step"]: step["verdict"] if step["ran"] else None
                    for step in row["steps"]
                },
                "m1_same_bits_as_s1": point["m1_same_bits_as_s1"],
                "torch_float32_same_bits_as_s1": point["torch_float32_same_bits_as_s1"],
            }
            for row, point in zip(gains["points"], points, strict=True)
        ],
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "not_run": len(record["not_run_at"]),
            }
            for record in gains["hypotheses"]
        },
        "bandwidth_gb_per_second": gains["bandwidth_gb_per_second"],
        "fma_gflops_per_second": gains["fma_gflops_per_second"],
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _measure(
    config: MetalTcConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    admissions = admit(backend.limits)
    refused = [name for name, verdict in admissions.items() if not verdict.admitted]
    if refused:
        raise MetalTcError(f"a pipeline refused {', '.join(refused)}")
    for variant in (S1, M1, M2):
        if not backend.supported(variant):
            raise MetalTcError(f"{variant} must run on every host of this lab")
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    cases = []
    for index, case in enumerate(config.correctness.cases):
        cases.append(
            check_case(
                backend,
                seed=config.seed,
                index=index,
                case=case,
                repeats=config.correctness.repeats,
            )
        )
        budget.check()
    points = [
        benchmark_point(
            backend,
            seed=config.seed,
            point=point,
            check_rows=timing.check_rows,
            warmup=timing.warmup,
            repeats=timing.repeats,
            target_flops_per_sample=timing.target_flops_per_sample,
            clock=clock,
            budget=budget,
        )
        for point in timing.points()
    ]
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MetalTcError("every sample must share one known timer")
    return {
        "admissions": admissions,
        "cases": cases,
        "points": [point.to_mapping() for point in points],
        "timer": timers.pop(),
        "elapsed_ns": elapsed,
        "timed_ns": sum(sum(item.samples_ns) for item in timings),
        "load": {
            "source": LOAD_AVERAGE_SOURCE,
            "before": before,
            "after": read_load_average(load_average),
        },
    }


def build_report(
    config: MetalTcConfig,
    ceiling: FlashCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalTcRuntime``, which raises
    ``MetalTcUnavailableError`` on a host without MPS, before anything is
    measured or written.
    """

    if not isinstance(config, MetalTcConfig):
        raise TypeError("config must be a MetalTcConfig")
    if not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MetalTcRuntime()
        _require_compiled(backend)
        real = type(backend) is MetalTcRuntime
        host = host_environment(backend.torch_version)
        _require_ceiling(ceiling, real and timer_name(clock) == TIMER, host)
        run = _measure(config, backend, clock, load_average)
        support = _support(backend)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    measured = run["timer"] == TIMER and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    gains = speedup(
        run["points"],
        ceiling.bandwidth.gb_per_second,
        ceiling.fma_gflops_per_second,
        config.comparison.min_faster_ratio,
        host["torch_version"],
    )
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
        "timer": run["timer"],
        "percentile_method": PERCENTILE_METHOD,
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "plan": {**plan_mapping(), "sha256": plan_sha256()},
        "environment": host,
        "torch_mps_switches": {
            "source": SWITCH_SOURCE,
            "values": {name: backend.mps_switches[name] for name in MPS_SWITCHES},
        },
        "host_load": run["load"],
        "warnings": captured,
        "shaders": _shaders(backend, run["admissions"]),
        "support": support,
        "planned_work": config.planned_work(),
        "ceiling": ceiling.bandwidth.to_mapping(),
        "compute_ceiling": ceiling.compute_mapping(),
        "correctness": {
            "distribution": DISTRIBUTION,
            "reference": REFERENCE,
            "bound": BOUND,
            "relative_error": RELATIVE_ERROR,
            "tail": TAIL,
            "guard_float32_bits": GUARD_BITS,
            "sentinel_float32_bits": SENTINEL_BITS,
            "rules": RULES,
            "cases": run["cases"],
        },
        "benchmark": {
            "device": DEVICE,
            "synchronization": SYNCHRONIZATION,
            "sides": list(SIDES),
            "torch_call": TORCH_CALL,
            "arithmetic": ARITHMETIC,
            "compulsory": COMPULSORY,
            "vendor_bound": VENDOR_BOUND,
            "check_rule": CHECK_RULE,
            "points": run["points"],
        },
        "speedup": gains,
        "verification": verification_scope(),
        "run": {
            "max_runtime_ns": config.benchmark.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": run["elapsed_ns"],
            "timed_ns": run["timed_ns"],
        },
        "summary": summarize(
            label,
            run["cases"],
            support,
            run["points"],
            gains,
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MetalTcConfig,
    ceiling: FlashCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalTcRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(
            config, ceiling, backend=backend, clock=clock, load_average=load_average
        )
    )
