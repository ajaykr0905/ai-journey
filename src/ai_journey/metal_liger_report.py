"""Run the probes, every case and every point under one budget, then seal a report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, measures the three math
probes, checks every correctness case of every family, times every point,
reads the load averages again and derives each speedup and each share of
the bandwidth ceiling. One ``RunBudget`` on the run's clock covers the
probes, the cases and the points, and a run past it stops without a report.

The warning capture is the Day 76 and Day 79 capture:
``warnings.catch_warnings`` with ``simplefilter("always")`` around the whole
run, equal warnings kept once with a count and the home directory as ``~``.
The record states whether torch was imported when the capture began. In a
fresh CLI process it was not, so the capture holds every warning that torch
raised in the process. The list is sealed even when it is empty, beside the
two MPS switches.

The report seals the declared plan with its digest, and the config must
name that digest. Only the real ``MetalLigerRuntime`` timed with
``time.perf_counter_ns`` makes a measurement, and a measurement needs a
measured Day 74 ceiling report from the same interpreter, libraries and
platform; the Day 79 ``BandwidthCeiling`` carries that join. Any other
backend or clock makes the report synthetic. The SHA-256 seal follows
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
from .metal_fused_ceiling import BandwidthCeiling
from .metal_fused_reference import DISTRIBUTIONS, GAP_INPUT, SENTINEL_BITS
from .metal_liger_benchmark import TRAFFIC_MODEL, UNWRITTEN_SOFTMAX, benchmark_point
from .metal_liger_config import COMPULSORY, MetalLigerConfig
from .metal_liger_correctness import RULES, check_case, check_probes
from .metal_liger_plan import TORCH_SEQUENCES, plan_mapping, plan_sha256
from .metal_liger_reference import BOUNDS, CORRECTNESS_GAP, REFERENCE, VENDOR_BOUNDS
from .metal_liger_runtime import MetalLigerRuntime, admit
from .metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    FAMILY_KERNELS,
    KERNELS,
    MetalLigerError,
    liger_library,
    library_sha256,
)
from .metal_liger_speedup import speedup

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "metal_liger_correctness_and_bandwidth"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
_FALSE_CLAIMS = {
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "triton_gpu_kernel_executed": False,
    "liger_kernel_code_executed": False,
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
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The three forward kernels "
        "came from torch.mps.compile_shader, and each PyTorch sequence ran on "
        "the MPS device. PyTorch does not report which kernels it ran, so each "
        "vendor number names the sequence, not a kernel. The bandwidth ceiling "
        "comes from a separate gpu-ceilings run with its own load averages. No "
        "hardware counter was read, and each byte count is a model. No CUDA "
        "kernel ran, no NVIDIA GPU was used, no Triton kernel ran on a GPU, and "
        "no code of the Liger Kernel project ran. Only forward passes were "
        "timed. These timings do not measure or predict CUDA, NVIDIA, Triton, "
        "Liger Kernel, distributed or production performance."
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
        "NVIDIA GPU was used, and no Triton kernel ran on a GPU."
    ),
}
RECOMPUTED = (
    "the config, its digest and the planned work",
    "the declared plan and every kernel digest",
    "every input digest, reference digest and bound digest, from the seed",
    "each layout and launch, and each admission from the recorded limits",
    "the probe argument rules and the declared probe assumptions",
    "each sample statistic, GB/s rate, model byte count, ratio, verdict, "
    "ceiling fraction and held hypothesis, from the raw samples",
    "the run totals, the summary and the SHA-256 seal",
)
GPU_ONLY = (
    "the samples_ns of each side: checked for form only, never observed again",
    "each kernel output digest, max_abs_error and differing count, unless every "
    "value equals the rounded reference: max_bound_ratio is checked to be at "
    "most 1, not recomputed",
    "identical_repeats: the CPU cannot rerun the GPU",
    "each probe results digest and measured error: checked against the rule",
    "the pipeline limits: checked for form, then used to rebuild the admissions",
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

    if type(backend) is MetalLigerRuntime and backend.library_source != (
        liger_library()
    ):
        raise MetalLigerError("the runtime compiled another library than the registry")


def _require_ceiling(ceiling: object, measured: bool, environment: dict) -> None:
    if not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    if measured and not ceiling.measured:
        raise MetalLigerError("a measured run needs a measured ceiling report")
    if measured and not ceiling.same_environment(environment):
        raise MetalLigerError(
            "the ceiling report comes from another interpreter, library or platform"
        )


def summarize(
    label: str,
    cases: list[dict[str, object]],
    probes: dict[str, dict[str, object]],
    points: list[dict[str, object]],
    gains: dict[str, object],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    """Reduce a run to the facts that a reader checks first."""

    by_family = {
        family: [case for case in cases if case["family"] == family]
        for family in FAMILIES
    }
    return {
        "label": label,
        "correctness_cases": len(cases),
        "max_bound_ratio": {
            family: max(case["output"]["max_bound_ratio"] for case in group)
            for family, group in by_family.items()
        },
        "identical_repeats": {
            family: all(case["output"]["identical_repeats"] for case in group)
            for family, group in by_family.items()
        },
        "naive_float32_failed_cases": [
            case["index"]
            for case in by_family[CROSS_ENTROPY]
            if case["naive_float32_finite"] is False
        ],
        "probe_max_ulps": {name: probe["max_ulps"] for name, probe in probes.items()},
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "family": row["family"],
                "index": row["index"],
                "shape": dict(row["shape"]),
                "kernel_gb_per_second": row["kernel"]["effective_gb_per_second"],
                "torch_gb_per_second": row["torch"]["effective_gb_per_second"],
                "ratio": row["to_torch"]["ratio"],
                "verdict": row["to_torch"]["verdict"],
                "held": row["held"],
            }
            for row in gains["points"]
        ],
        "hypotheses": {
            record["family"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
            }
            for record in gains["hypotheses"]
        },
        "ceiling_gb_per_second": gains["ceiling_gb_per_second"],
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _measure(
    config: MetalLigerConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    required = config.required_threads()
    admissions = admit(backend.limits, required)
    refused = [name for name, verdict in admissions.items() if not verdict.admitted]
    if refused:
        raise MetalLigerError(f"a pipeline refused {', '.join(refused)}")
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    probes = check_probes(backend)
    budget.check()
    cases = []
    for family in FAMILIES:
        for index, case in enumerate(config.correctness.cases[family]):
            record = check_case(
                backend,
                seed=config.seed,
                index=index,
                case=case,
                repeats=config.correctness.repeats,
                required=required,
            )
            cases.append(record.to_mapping())
            budget.check()
    points = [
        benchmark_point(
            backend,
            seed=config.seed,
            index=index,
            shape=shape,
            inputs=timing.inputs,
            warmup=timing.warmup,
            repeats=timing.repeats,
            target_bytes_per_sample=timing.target_bytes_per_sample,
            clock=clock,
            budget=budget,
        )
        for family in FAMILIES
        for index, shape in enumerate(timing.shapes[family])
    ]
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MetalLigerError("every sample must share one known timer")
    return {
        "admissions": admissions,
        "probes": probes,
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
    config: MetalLigerConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalLigerRuntime``, which raises
    ``MetalLigerUnavailableError`` on a host without MPS, before anything is
    measured or written.
    """

    if not isinstance(config, MetalLigerConfig):
        raise TypeError("config must be a MetalLigerConfig")
    if not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MetalLigerRuntime()
        _require_compiled(backend)
        real = type(backend) is MetalLigerRuntime
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
        ceiling.gb_per_second,
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
        "ceiling": ceiling.to_mapping(),
        "math_probes": run["probes"],
        "correctness": {
            "reference": dict(REFERENCE),
            "bounds": dict(BOUNDS),
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
            "kernels": dict(FAMILY_KERNELS),
            "torch_sequences": dict(TORCH_SEQUENCES),
            "compulsory": dict(COMPULSORY),
            "traffic_model": dict(TRAFFIC_MODEL),
            "unwritten_softmax": UNWRITTEN_SOFTMAX,
            "vendor_bounds": dict(VENDOR_BOUNDS),
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
            run["probes"],
            run["points"],
            gains,
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MetalLigerConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalLigerRuntime``, or any clock other than
    the host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(
            config, ceiling, backend=backend, clock=clock, load_average=load_average
        )
    )
