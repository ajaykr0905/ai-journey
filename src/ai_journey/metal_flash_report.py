"""Run the probe, every case and every point under one budget, then seal a report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, measures the exp probe, checks
every correctness case, measures the memory and the timing of every point,
reads the load averages again and derives each verdict and each share of
the two ceilings. One ``RunBudget`` on the run's clock covers the probe,
the cases and the points, and a run past it stops without a report.

The warning capture is the Day 76 to 80 capture. The record states whether
torch was imported when the capture began. In a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process.

The report seals the declared plan with its digest, and the config must
name that digest. Only the real ``MetalFlashRuntime`` timed with
``time.perf_counter_ns`` makes a measurement, and a measurement needs a
measured Day 74 ceiling report from the same interpreter, libraries and
platform. Any other backend or clock makes the report synthetic. The
SHA-256 seal follows ``memory_access_report``. It detects drift and
tampering; it is not a signature and cannot prove that a timing was
observed.
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
from .metal_flash_benchmark import ARITHMETIC, TRAFFIC_MODEL, benchmark_point
from .metal_flash_ceiling import FlashCeiling
from .metal_flash_config import COMPULSORY, MetalFlashConfig
from .metal_flash_correctness import RULES, check_case, check_probe
from .metal_flash_memory import CONSTRUCTION, COUNTERS_CANNOT_SHOW, COUNTERS_MEASURE
from .metal_flash_plan import SIDES, TORCH_CALL, plan_mapping, plan_sha256
from .metal_flash_reference import (
    BOUND,
    CORRECTNESS_GAP,
    DISTRIBUTIONS,
    REFERENCE,
    VENDOR_BOUND,
)
from .metal_flash_runtime import MetalFlashRuntime, admit
from .metal_flash_source import (
    KERNELS,
    MODES,
    VARIANTS,
    MetalFlashError,
    flash_library,
    library_sha256,
)
from .metal_flash_speedup import speedup
from .metal_fused_reference import GAP_INPUT, SENTINEL_BITS

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "metal_flash_attention"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
CHECK_RULE = (
    "a timing point checks check_rows query rows of each sequence, spread "
    "evenly from the first row to the last; a correctness case checks every row"
)
_FALSE_CLAIMS = {
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "triton_gpu_kernel_executed": False,
    "ptx_executed": False,
    "backward_pass_measured": False,
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
        "These are wall-clock timings and allocator readings on the Apple GPU of "
        "the host that made this report, at the recorded load averages. N0 and "
        "F1 came from torch.mps.compile_shader, and the PyTorch call ran on the "
        "MPS device. PyTorch does not report which kernels it ran, so its numbers "
        "name the call, not a kernel. Both ceilings come from a separate "
        "gpu-ceilings run with its own load averages. No hardware counter was "
        "read, and each byte and operation count is a model. No CUDA kernel ran, "
        "no NVIDIA GPU was used, no PTX ran, and no Triton kernel ran on a GPU. "
        "Only forward passes were timed. These numbers do not measure or predict "
        "CUDA, NVIDIA, Triton, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    **_FALSE_CLAIMS,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate, ratio, fraction and memory reading in it is synthetic. "
        "They are not measurements of this host or of any other host. The "
        "correctness checks ran on the outputs that the backend returned. No "
        "CUDA kernel ran, no NVIDIA GPU was used, and no Triton kernel ran on a "
        "GPU."
    ),
}
RECOMPUTED = (
    "the config, its digest and the planned work",
    "the declared plan, every kernel digest and the library digest",
    "every input digest, reference digest and bound digest, from the seed",
    "each layout and launch, and each admission from the recorded limits",
    "the exp probe arguments and the declared probe assumption",
    "each sample statistic, GB/s and GFLOP/s rate, model byte count, ratio, "
    "verdict, ceiling fraction and held hypothesis, from the raw samples",
    "the bytes of each side by construction and each memory difference, from "
    "the recorded readings",
    "the run totals, the summary and the SHA-256 seal",
)
GPU_ONLY = (
    "the samples_ns of each side: checked for form only, never observed again",
    "each kernel output digest, max_abs_error and differing count, unless every "
    "value equals the rounded reference: max_bound_ratio is checked to be at "
    "most 1, not recomputed",
    "identical_repeats: the CPU cannot rerun the GPU",
    "the exp probe results digest and measured error: checked against the rule",
    "the pipeline limits: checked for form, then used to rebuild the admissions",
    "every allocator reading and poll count: checked for form only",
    "the PyTorch side checks, the warnings, the MPS switches and the load "
    "averages: checked for form only",
)


def verification_scope() -> dict[str, list[str]]:
    """State what the CPU verifier recomputes and what only the GPU run saw."""

    return {"recomputed": list(RECOMPUTED), "gpu_only": list(GPU_ONLY)}


def _shaders(backend: object, admissions: dict) -> dict[str, object]:
    return {
        "library_sha256": library_sha256(),
        "kernels": [
            {**kernel.to_mapping(), "limits": backend.limits[name].to_mapping()}
            for name, kernel in KERNELS.items()
        ],
        "admissions": [admissions[name].to_mapping() for name in KERNELS],
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled another library than the registry's."""

    if type(backend) is MetalFlashRuntime and backend.library_source != (
        flash_library()
    ):
        raise MetalFlashError("the runtime compiled another library than the registry")


def _require_ceiling(ceiling: object, measured: bool, environment: dict) -> None:
    if not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    if measured and not ceiling.measured:
        raise MetalFlashError("a measured run needs a measured ceiling report")
    if measured and not ceiling.same_environment(environment):
        raise MetalFlashError(
            "the ceiling report comes from another interpreter, library or platform"
        )


def summarize(
    label: str,
    cases: list[dict[str, object]],
    probe: dict[str, object],
    points: list[dict[str, object]],
    gains: dict[str, object],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    """Reduce a run to the facts that a reader checks first."""

    outputs = [
        (variant, case["modes"][mode]["outputs"][variant])
        for case in cases
        for mode in MODES
        for variant in VARIANTS
    ]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "max_bound_ratio": {
            variant: max(
                entry["max_bound_ratio"] for name, entry in outputs if name == variant
            )
            for variant in VARIANTS
        },
        "identical_repeats": {
            variant: all(
                entry["identical_repeats"] for name, entry in outputs if name == variant
            )
            for variant in VARIANTS
        },
        "unscaled_exp_overflow_cases": [
            case["index"]
            for case in cases
            if any(case["modes"][mode]["unscaled_exp_overflows"] for mode in MODES)
        ],
        "probe_max_ulps": probe["max_ulps"],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "index": row["index"],
                "sequences": row["sequences"],
                "length": row["length"],
                "mode": row["mode"],
                "median_ms_per_call": {
                    side: row["rates"][side]["median_ns_per_call"] / 1e6
                    for side in SIDES
                },
                "gflops_per_second": {
                    side: row["rates"][side]["gflops_per_second"] for side in SIDES
                },
                "verdicts": {step["step"]: step["verdict"] for step in row["steps"]},
                "peak_driver_over_before_bytes": {
                    side: point["memory"][side]["peak_driver_over_before_bytes"]
                    for side in SIDES
                },
            }
            for row, point in zip(gains["points"], points)
        ],
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
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
    config: MetalFlashConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    admissions = admit(backend.limits)
    refused = [name for name, verdict in admissions.items() if not verdict.admitted]
    if refused:
        raise MetalFlashError(f"a pipeline refused {', '.join(refused)}")
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    probe = check_probe(backend)
    budget.check()
    cases = []
    for index, case in enumerate(config.correctness.cases):
        record = check_case(
            backend,
            seed=config.seed,
            index=index,
            case=case,
            repeats=config.correctness.repeats,
        )
        cases.append(record.to_mapping())
        budget.check()
    points = [
        benchmark_point(
            backend,
            seed=config.seed,
            point=point,
            inputs=timing.inputs,
            check_rows=timing.check_rows,
            warmup=timing.warmup,
            repeats=timing.repeats,
            target_bytes_per_sample=timing.target_bytes_per_sample,
            clock=clock,
            budget=budget,
        )
        for point in timing.points()
    ]
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MetalFlashError("every sample must share one known timer")
    return {
        "admissions": admissions,
        "probe": probe,
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
    config: MetalFlashConfig,
    ceiling: FlashCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalFlashRuntime``, which raises
    ``MetalFlashUnavailableError`` on a host without MPS, before anything is
    measured or written.
    """

    if not isinstance(config, MetalFlashConfig):
        raise TypeError("config must be a MetalFlashConfig")
    if not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MetalFlashRuntime()
        _require_compiled(backend)
        real = type(backend) is MetalFlashRuntime
        host = host_environment(backend.torch_version)
        _require_ceiling(ceiling, real and timer_name(clock) == TIMER, host)
        run = _measure(config, backend, clock, load_average)
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
        "planned_work": config.planned_work(),
        "ceiling": ceiling.bandwidth.to_mapping(),
        "compute_ceiling": ceiling.compute_mapping(),
        "exp_probe": run["probe"],
        "correctness": {
            "reference": REFERENCE,
            "bound": BOUND,
            "distributions": dict(DISTRIBUTIONS),
            "gap": CORRECTNESS_GAP,
            "gap_input": GAP_INPUT,
            "sentinel_float32_bits": SENTINEL_BITS,
            "rules": RULES,
            "cases": run["cases"],
        },
        "benchmark": {
            "device": DEVICE,
            "inputs": config.benchmark.inputs,
            "synchronization": SYNCHRONIZATION,
            "sides": list(SIDES),
            "torch_call": TORCH_CALL,
            "compulsory": COMPULSORY,
            "arithmetic": ARITHMETIC,
            "traffic_model": dict(TRAFFIC_MODEL),
            "vendor_bound": VENDOR_BOUND,
            "check_rule": CHECK_RULE,
            "points": run["points"],
        },
        "memory_method": {
            "measures": COUNTERS_MEASURE,
            "cannot_show": COUNTERS_CANNOT_SHOW,
            "construction": dict(CONSTRUCTION),
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
            run["probe"],
            run["points"],
            gains,
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MetalFlashConfig,
    ceiling: FlashCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalFlashRuntime``, or any clock other than
    the host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(
            config, ceiling, backend=backend, clock=clock, load_average=load_average
        )
    )
