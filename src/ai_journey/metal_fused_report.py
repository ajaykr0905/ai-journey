"""Run the probe, every case and every point under one budget, then seal a report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, measures the exp probe, checks
every correctness case on every admitted step, times every point, reads the
load averages again and derives the speedup of each step and its share of
the bandwidth ceiling. One ``RunBudget`` on the run's clock covers the
probe, the cases and the points, and a run past it stops without a report.
The budget starts after the library compiles, so the time to import torch on
a busy host does not count.

The warning capture is the Day 76 capture: ``warnings.catch_warnings`` with
``simplefilter("always")`` around the whole run, equal warnings kept once
with a count and the home directory as ``~``. The record states whether
torch was imported when the capture began. In a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process.
The list is sealed even when it is empty, beside the two MPS switches.

The report seals the declared plan with its digest, and the config must
name that digest, so a run cannot test other hypotheses than the declared
ones. Labels keep the evidence honest. Only the real ``MetalFusedRuntime``
timed with ``time.perf_counter_ns`` makes a measurement, with the Day 73
scope ``apple_gpu_metal_this_host``, and a measurement needs a measured
ceiling report from the same interpreter, libraries and platform. Any
other backend or clock makes the report synthetic. The SHA-256 seal
follows ``memory_access_report``: canonical JSON of every field but the
digest. It detects drift and tampering; it is not a signature and cannot
prove that a timing was observed.
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
from .metal_fused_benchmark import (
    COMPULSORY,
    TRAFFIC_MODEL,
    VENDOR_BOUND,
    benchmark_point,
)
from .metal_fused_ceiling import BandwidthCeiling
from .metal_fused_config import MetalFusedConfig
from .metal_fused_correctness import RULES, check_case, check_exp_probe
from .metal_fused_plan import PLAN, plan_mapping, plan_sha256
from .metal_fused_reference import (
    BOUND,
    CORRECTNESS_GAP,
    DISTRIBUTIONS,
    GAP_INPUT,
    ORDER_TERMS,
    REFERENCE,
    ROW_SUM_BOUND,
    SENTINEL_BITS,
)
from .metal_fused_runtime import TORCH_EQUIVALENT, MetalFusedRuntime, admit
from .metal_fused_source import (
    KERNELS,
    STEP_VARIANTS,
    VARIANTS,
    MetalFusedError,
    fused_library,
    library_sha256,
)
from .metal_fused_speedup import speedup

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "metal_fused_softmax_correctness_and_bandwidth"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "triton_gpu_kernel_executed": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The softmax kernels came "
        "from torch.mps.compile_shader, and torch.softmax ran on the MPS device. "
        "PyTorch does not report which kernel torch.softmax ran, so the vendor "
        "number names the call, not a kernel. The bandwidth ceiling comes from a "
        "separate gpu-ceilings run with its own load averages. No hardware "
        "counter was read, so no cache hit, stall or occupancy was observed, and "
        "each byte count is a model. No CUDA kernel ran, no NVIDIA GPU was used, "
        "and no Triton kernel ran on a GPU. Each ratio compares two sides of one "
        "run on this host. These timings do not measure or predict CUDA, NVIDIA, "
        "Triton, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "triton_gpu_kernel_executed": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate, ratio and fraction in it is synthetic. They are not "
        "measurements of this host or of any other host. The correctness checks "
        "ran on the outputs that the backend returned. No CUDA kernel ran, no "
        "NVIDIA GPU was used, and no Triton kernel ran on a GPU."
    ),
}


def _shaders(backend: object, admissions: dict) -> dict[str, object]:
    return {
        "library_sha256": library_sha256(),
        "kernels": [
            {**kernel.to_mapping(), "limits": backend.limits[name].to_mapping()}
            for name, kernel in KERNELS.items()
        ],
        "steps": [variant.to_mapping() for variant in VARIANTS.values()],
        "admissions": [admissions[name].to_mapping() for name in STEP_VARIANTS],
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled another library than the registry's.

    The report names the library by the registry's digest, so a measured
    report must come from a runtime that compiled exactly that text.
    """

    if type(backend) is MetalFusedRuntime and backend.library_source != (
        fused_library()
    ):
        raise MetalFusedError("the runtime compiled another library than the registry")


def _require_ceiling(ceiling: object, measured: bool, environment: dict) -> None:
    if not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    if measured and not ceiling.measured:
        raise MetalFusedError("a measured run needs a measured ceiling report")
    if measured and not ceiling.same_environment(environment):
        raise MetalFusedError(
            "the ceiling report comes from another interpreter, library or platform"
        )


def _same_bits_at(cases: list[dict], variant: str, same: bool) -> list[int]:
    return [
        case["index"]
        for case in cases
        for item in case["same_bits"]
        if item["variant"] == variant and item["same"] is same
    ]


def _summary(
    label: str,
    names: list[str],
    refused: list[str],
    cases: list[dict[str, object]],
    probe: dict[str, object],
    points: list[dict[str, object]],
    gains: dict[str, object],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    outputs = [output for case in cases for output in case["outputs"]]
    rows = []
    for point, entry in zip(points, gains["points"]):
        rows.append(
            {
                "index": point["index"],
                "shape": dict(point["shape"]),
                "best": None if entry["best"] is None else entry["best"]["variant"],
                "effective_gb_per_second": {
                    name: side["effective_gb_per_second"]
                    for name, side in point["sides"].items()
                },
                "to_base": {
                    row["variant"]: None
                    if row["to_base"] is None
                    else row["to_base"]["verdict"]
                    for row in entry["rows"]
                },
            }
        )
    return {
        "label": label,
        "steps_checked": names,
        "refused_steps": refused,
        "correctness_cases": len(cases),
        "max_bound_ratio": max(output["max_bound_ratio"] for output in outputs),
        "max_row_sum_ratio": max(output["max_row_sum_ratio"] for output in outputs),
        "identical_repeats": {
            name: all(
                output["identical_repeats"]
                for output in outputs
                if output["kernel"] == name
            )
            for name in names
        },
        "same_bits": [
            {
                "variant": step.variant,
                "as": step.same_bits_as,
                "same_at": _same_bits_at(cases, step.variant, True),
                "differ_at": _same_bits_at(cases, step.variant, False),
            }
            for step in PLAN
            if step.same_bits_as is not None
        ],
        "naive_float32_failed_cases": [
            case["index"] for case in cases if not case["naive_float32_finite"]
        ],
        "exp_probe_max_ulps": probe["max_ulps"],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": rows,
        "hypotheses": {
            record["variant"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
            }
            for record in gains["hypotheses"]
        },
        "ceiling_gb_per_second": gains["ceiling_gb_per_second"],
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _measure(
    config: MetalFusedConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    admissions = admit(backend.limits, config.largest_group())
    names = [name for name in STEP_VARIANTS if admissions[name].admitted]
    if not names:
        raise MetalFusedError("every step was refused by its pipeline")
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    probe = check_exp_probe(backend).to_mapping()
    budget.check()
    cases = []
    for index, case in enumerate(config.correctness.cases):
        record = check_case(
            backend,
            seed=config.seed,
            index=index,
            case=case,
            names=tuple(names),
            repeats=config.correctness.repeats,
            required=config.largest_group(),
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
            names=tuple(names),
            warmup=timing.warmup,
            repeats=timing.repeats,
            target_bytes_per_sample=timing.target_bytes_per_sample,
            clock=clock,
            budget=budget,
        )
        for index, shape in enumerate(timing.shapes)
    ]
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise MetalFusedError("every sample must share one known timer")
    return {
        "admissions": admissions,
        "names": names,
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
    config: MetalFusedConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MetalFusedRuntime``, which raises
    ``MetalFusedUnavailableError`` on a host without MPS, before anything
    is measured or written.
    """

    if not isinstance(config, MetalFusedConfig):
        raise TypeError("config must be a MetalFusedConfig")
    if not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MetalFusedRuntime()
        _require_compiled(backend)
        real = type(backend) is MetalFusedRuntime
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
    admissions = run["admissions"]
    refusals = [
        admissions[name].to_mapping()
        for name in STEP_VARIANTS
        if not admissions[name].admitted
    ]
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
        "shaders": _shaders(backend, admissions),
        "refusals": refusals,
        "planned_work": config.planned_work().to_mapping(),
        "ceiling": ceiling.to_mapping(),
        "exp_probe": run["probe"],
        "correctness": {
            "reference": REFERENCE,
            "bound": BOUND,
            "order_terms": dict(ORDER_TERMS),
            "row_sum_bound": ROW_SUM_BOUND,
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
            "compulsory": COMPULSORY,
            "traffic_model": TRAFFIC_MODEL,
            "torch_equivalent": TORCH_EQUIVALENT,
            "vendor_bound": VENDOR_BOUND,
            "points": run["points"],
        },
        "speedup": gains,
        "run": {
            "max_runtime_ns": config.benchmark.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": run["elapsed_ns"],
            "timed_ns": run["timed_ns"],
        },
        "summary": _summary(
            label,
            run["names"],
            [entry["variant"] for entry in refusals],
            run["cases"],
            run["probe"],
            run["points"],
            gains,
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MetalFusedConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MetalFusedRuntime``, or any clock other than
    the host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(
            config, ceiling, backend=backend, clock=clock, load_average=load_average
        )
    )
