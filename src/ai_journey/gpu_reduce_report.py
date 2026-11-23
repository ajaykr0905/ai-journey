"""Run every reduction case and size under one budget, then seal a labeled report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, checks every correctness case
on every kernel, times every size, reads the load averages again and
derives the speedup of each step. One ``RunBudget`` on the run's clock
covers the cases and the sweep, and a run past it stops without a report.
The budget starts after the library compiles, so the time to import torch
on a busy host does not count.

The warning capture is the Day 76 capture: ``warnings.catch_warnings`` with
``simplefilter("always")`` around the whole run, equal warnings kept once
with a count and the home directory as ``~``. The record states whether
torch was imported when the capture began. In a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process,
from its start to the end of the run. The list is sealed even when it is
empty, and so are the two PyTorch MPS switches.

The report seals the declared plan with its digest, and the config must
name that digest. The bandwidth ceiling comes from a verified Day 74
report. A measured run refuses a synthetic ceiling or one from another
environment before it runs, so every fraction in a measured report divides
two measurements of one environment. Only the real ``GpuReduceRuntime``
timed with ``time.perf_counter_ns`` makes a measurement, with the label
``host-specific Apple GPU (Metal) measurement``. Any other backend or clock
makes the report synthetic. The SHA-256 seal follows
``memory_access_report``: it detects drift and tampering; it is not a
signature and cannot prove that a timing was observed.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable

import numpy

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
from .gpu_reduce_benchmark import GB_PER_SECOND, benchmark_sweep
from .gpu_reduce_bound import (
    BOUND_SOURCE,
    DEPTH_RULE,
    EXACT_RULE,
    VENDOR_RULE,
    combine_launch,
)
from .gpu_reduce_ceiling import FRACTION_RULE, BandwidthCeiling
from .gpu_reduce_config import GpuReduceConfig
from .gpu_reduce_correctness import check_cases
from .gpu_reduce_plan import PLAN, plan_mapping, plan_sha256
from .gpu_reduce_runtime import SIDES, TORCH_EQUIVALENT, GpuReduceRuntime
from .gpu_reduce_source import (
    COMBINE,
    KERNELS,
    STEP_KERNELS,
    GpuReduceError,
    library_sha256,
    library_source,
)
from .gpu_reduce_speedup import speedup
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_geometry import CheckedLaunch
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
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
from .memory_access_benchmark import (
    INJECTED_TIMER,
    PERCENTILE_METHOD,
    TIMER,
    timer_name,
)
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "gpu_reduce_correctness_and_bandwidth"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
REFERENCE = (
    "the exact sum of the float32 inputs, from their integer significands, "
    "rounded once to float64"
)
UNIT_ROUNDOFF = "2^-24"
CORRECTNESS_RULES = (
    "every partial slot is written, the partial and output tails keep their "
    "sentinels, and the result is finite, within the bound of its order and "
    "equal to the exact sum when every order is exact; same_bits_every_run and "
    "same_order are measured, not required"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "warp_level_claim": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The reduction kernels came "
        "from torch.mps.compile_shader, and torch.sum ran on the MPS device. "
        "PyTorch does not report which kernel torch.sum ran, so the vendor number "
        "names the call, not a kernel. simd_shuffle_down and simd_sum are the "
        "Metal counterpart of CUDA warp shuffles, not the same instructions. No "
        "hardware counter was read, so a limit named from these timings is an "
        "inference. The bandwidth ceiling comes from a separate Day 74 run on "
        "this host, and it is not a vendor peak. No CUDA kernel ran, and no "
        "NVIDIA GPU was used. These timings do not measure or predict CUDA, "
        "NVIDIA, warp-level, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "warp_level_claim": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate, fraction and ratio in it is synthetic. They are not "
        "measurements of this host or of any other host. The correctness checks "
        "ran on the outputs that the backend returned. No CUDA kernel ran, and "
        "no NVIDIA GPU was used."
    ),
}


def _bits(value: float) -> bytes:
    return numpy.float32(value).tobytes()


def _shaders(backend: object) -> dict[str, object]:
    return {
        "library_sha256": library_sha256(),
        "kernels": [
            {
                **kernel.to_mapping(),
                "limits": backend.limits[name].to_mapping(),
                "admission": backend.admissions[name].to_mapping(),
            }
            for name, kernel in KERNELS.items()
        ],
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled another library than the registry's."""

    if type(backend) is GpuReduceRuntime and backend.library_source != library_source():
        raise GpuReduceError("the runtime compiled another library than the registry")


def _require_ceiling(
    ceiling: BandwidthCeiling, environment: dict[str, str], measured: bool
) -> None:
    """A measured run takes only a measured ceiling of its own environment."""

    if measured and not (ceiling.measured and ceiling.same_environment(environment)):
        raise GpuReduceError(
            "a measured run needs a measured gpu-ceilings report from the same "
            "interpreter, libraries and platform"
        )


def _measure(
    config: GpuReduceConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    cases = check_cases(
        backend, seed=config.seed, controls=config.correctness, budget=budget
    )
    points = benchmark_sweep(
        backend,
        seed=config.seed,
        sizes=timing.sizes,
        warmup=timing.warmup,
        repeats=timing.repeats,
        target_bytes_per_sample=timing.target_bytes_per_sample,
        clock=clock,
        budget=budget,
    )
    elapsed = budget.check()
    timings = [item for point in points for item in point.timings]
    timers = {item.timer for item in timings}
    if len(timers) != 1 or not timers <= {TIMER, INJECTED_TIMER}:
        raise GpuReduceError("every sample must share one known timer")
    return {
        "cases": [case.to_mapping() for case in cases],
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


def same_order_record(
    cases: list[dict[str, object]], points: list[dict[str, object]]
) -> list[dict[str, object]]:
    """For each step that keeps an earlier order, where its bits differed."""

    records = []
    for step in PLAN:
        if step.same_order_as is None:
            continue
        position = STEP_KERNELS.index(step.kernel)
        earlier = STEP_KERNELS.index(step.same_order_as)
        records.append(
            {
                "kernel": step.kernel,
                "same_order_as": step.same_order_as,
                "cases_with_other_bits": [
                    case["index"]
                    for case in cases
                    if _bits(case["outputs"][position]["output"])
                    != _bits(case["outputs"][earlier]["output"])
                ],
                "points_with_other_bits": [
                    point["index"]
                    for point in points
                    if _bits(point["checks"][step.kernel]["output"])
                    != _bits(point["checks"][step.same_order_as]["output"])
                ],
            }
        )
    return records


def summarize(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    gains: dict[str, object],
    captured: dict[str, object],
    load: dict[str, object],
    ceiling: dict[str, object],
    fraction_measured: bool,
) -> dict[str, object]:
    """Reduce the sections to the facts a reader checks first."""

    kernels = list(STEP_KERNELS)
    rows = []
    for entry in gains["points"]:
        best = entry["best"]
        rows.append(
            {
                "n": entry["n"],
                "best_kernel": best["kernel"],
                "best_gb_per_second": best["median_gb_per_second"],
                "best_fraction_of_ceiling": best["fraction_of_ceiling"],
                "best_over_r0": None
                if best["to_r0"] is None
                else best["to_r0"]["ratio"],
                "torch_sum_gb_per_second": entry["torch_sum"]["median_gb_per_second"],
                "torch_sum_fraction_of_ceiling": entry["torch_sum"][
                    "fraction_of_ceiling"
                ],
            }
        )
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "exact_cases": [case["index"] for case in cases if case["exact_in_any_order"]],
        "max_bound_ratio": {
            name: max(case["outputs"][position]["bound_ratio"] for case in cases)
            for position, name in enumerate(kernels)
        },
        "cases_with_changed_bits": {
            name: [
                case["index"]
                for case in cases
                if not case["outputs"][position]["same_bits_every_run"]
            ]
            for position, name in enumerate(kernels)
        },
        "points_with_changed_bits": {
            name: [
                point["index"]
                for point in points
                if not point["checks"][name]["same_bits_on_repeat"]
            ]
            for name in SIDES
        },
        "same_order": same_order_record(cases, points),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": rows,
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
            }
            for record in gains["hypotheses"]
        },
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
        "ceiling_gb_per_second": ceiling["gb_per_second"],
        "ceiling_report_sha256": ceiling["report_sha256"],
        "fraction_measured": fraction_measured,
    }


def build_report(
    config: GpuReduceConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``GpuReduceRuntime``, which raises
    ``GpuReduceUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, GpuReduceConfig):
        raise TypeError("config must be a GpuReduceConfig")
    if not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = GpuReduceRuntime()
        _require_compiled(backend)
        real = type(backend) is GpuReduceRuntime
        host = host_environment(backend.torch_version)
        _require_ceiling(ceiling, host, real and timer_name(clock) == TIMER)
        run = _measure(config, backend, clock, load_average)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    measured = run["timer"] == TIMER and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    same_environment = ceiling.same_environment(host)
    section = {
        **ceiling.to_mapping(),
        "same_environment": same_environment,
        "fraction_rule": FRACTION_RULE,
    }
    gains = speedup(
        run["points"],
        config.comparison.min_faster_ratio,
        host["torch_version"],
        ceiling.gb_per_second,
    )
    fraction_measured = measured and ceiling.measured and same_environment
    combine = CheckedLaunch(combine_launch(), backend.limits[COMBINE])
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
        "shaders": _shaders(backend),
        "ceiling": section,
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": REFERENCE,
            "unit_roundoff": UNIT_ROUNDOFF,
            "bound_source": BOUND_SOURCE,
            "depth_rule": DEPTH_RULE,
            "exact_rule": EXACT_RULE,
            "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
            "rules": CORRECTNESS_RULES,
            "combine_launch": combine.to_mapping(),
            "cases": run["cases"],
        },
        "benchmark": {
            "device": DEVICE,
            "gb_per_second": GB_PER_SECOND,
            "torch_equivalent": TORCH_EQUIVALENT,
            "vendor_rule": VENDOR_RULE,
            "synchronization": SYNCHRONIZATION,
            "points": run["points"],
        },
        "speedup": gains,
        "run": {
            "max_runtime_ns": config.benchmark.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": run["elapsed_ns"],
            "timed_ns": run["timed_ns"],
        },
        "summary": summarize(
            label,
            run["cases"],
            run["points"],
            gains,
            captured,
            run["load"],
            section,
            fraction_measured,
        ),
    }


def measure_report(
    config: GpuReduceConfig,
    ceiling: BandwidthCeiling,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``GpuReduceRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(
            config, ceiling, backend=backend, clock=clock, load_average=load_average
        )
    )
