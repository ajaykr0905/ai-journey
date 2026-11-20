"""Run every checklist case and point under one budget, then seal a labeled report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, checks every correctness case
on every admitted kernel, times every sweep point, reads the load averages
again and derives the speedup of each checklist change. One ``RunBudget``
on the run's clock covers the cases and the sweep, and a run past it stops
without a report. The budget starts after the kernels compile, so the time
to import torch on a busy host does not count.

The warning capture is the Day 76 capture: ``warnings.catch_warnings`` with
``simplefilter("always")`` around the whole run, equal warnings kept once
with a count and the home directory as ``~``. The record states whether
torch was imported when the capture began. In a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process
up to the end of the run. The list is sealed even when it is empty.

The report seals the declared checklist with its digest, and the config must
name that digest, so a run cannot test other hypotheses than the declared
ones. Labels keep the evidence honest. Only the
real ``MatmulChecklistRuntime`` timed with ``time.perf_counter_ns`` makes a
measurement, with the Day 73 scope ``apple_gpu_metal_this_host`` and the
label ``host-specific Apple GPU (Metal) measurement``. Any other backend or
clock makes the report synthetic. The SHA-256 seal follows
``memory_access_report``: canonical JSON of every field but the digest. It
detects drift and tampering; it is not a signature and cannot prove that a
timing was observed.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from collections.abc import Callable

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE, read_load_average
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_checklist_benchmark import TRAFFIC_MODEL, benchmark_sweep
from .matmul_checklist_config import MatmulChecklistConfig
from .matmul_checklist_correctness import check_case
from .matmul_checklist_runtime import MatmulChecklistRuntime
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_speedup import speedup
from .matmul_checklist_steps import checklist_mapping, checklist_sha256
from .matmul_checklist_variants import (
    VARIANTS,
    admitted_names,
    checklist_library,
)
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    RELATIVE_ERROR,
)
from .matmul_metal_reference import REFERENCE as FLOAT64_REFERENCE
from .matmul_metal_report import FLOPS_PER_CALL, GFLOPS, TORCH_EQUIVALENT, environment
from .matmul_metal_source import source_sha256
from .matmul_tiled_benchmark import AGREEMENT
from .matmul_tiled_report import (
    CORRECTNESS_RULES,
    SWITCH_SOURCE,
    SYNCHRONIZATION,
    warning_records,
)
from .matmul_tiled_runtime import MPS_SWITCHES
from .matmul_tiled_variants import text_sha256
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_checklist_correctness_and_speedup"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
REPLAY = (
    "replay_bits compares an output with the CPU replay of sum = fma(a[i, k], "
    "b[k, j], sum) for k = 0 to K - 1, one float32 rounding per step; every "
    "checklist kernel runs that chain, one fma per k in increasing order"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The checklist kernels came "
        "from torch.mps.compile_shader, and torch.matmul ran on the MPS device. "
        "PyTorch does not report which kernel torch.matmul ran, so the vendor "
        "number names the call, not a kernel. No hardware counter was read, so "
        "the occupancy step records times and pipeline limits, not occupancy. "
        "No CUDA kernel ran, no NVIDIA GPU was used, and cuBLAS was not "
        "measured. Each ratio compares two sides of one run on this host. These "
        "timings do not measure or predict CUDA, NVIDIA, cuBLAS, distributed or "
        "production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "A stand-in backend or an injected clock made this report, so each "
        "duration, rate and ratio in it is synthetic. They are not measurements "
        "of this host or of any other host. The correctness checks ran on the "
        "outputs that the backend returned. No CUDA kernel ran, no NVIDIA GPU "
        "was used, and cuBLAS was not measured."
    ),
}


def _shaders(backend: object) -> dict[str, object]:
    return {
        "naive_source_sha256": source_sha256(),
        "library_sha256": text_sha256(checklist_library()),
        "variants": [
            {
                **variant.to_mapping(),
                "limits": backend.limits[name].to_mapping(),
                "admission": None
                if name not in backend.admissions
                else backend.admissions[name].to_mapping(),
            }
            for name, variant in VARIANTS.items()
        ],
    }


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    gains: dict[str, object],
    refused: list[str],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    kernels = [output["kernel"] for output in cases[0]["outputs"]] if cases else []
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "refused_variants": refused,
        "cases_with_other_bits": {
            name: [
                case["index"]
                for case in cases
                if not case["outputs"][position]["replay_bits"]
            ]
            for position, name in enumerate(kernels)
        },
        "max_bound_ratio": max(
            output["bound_ratio"] for case in cases for output in case["outputs"]
        ),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "best": [
            {
                "m": entry["m"],
                "n": entry["n"],
                "k": entry["k"],
                "variant": entry["best"]["variant"],
                "over_s0": entry["best"]["to_s0"]["ratio"],
                "fraction_of_torch": entry["best"]["fraction_of_torch"],
                "torch_gflops_per_second": entry["torch"]["median_gflops_per_second"],
            }
            for entry in gains["points"]
        ],
        "hypotheses": {
            record["variant"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
            }
            for record in gains["hypotheses"]
        },
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled another library than the registry's.

    The report names the library by the registry's digest, so a measured
    report must come from a runtime that compiled exactly that text.
    """

    if (
        type(backend) is MatmulChecklistRuntime
        and backend.library_source != checklist_library()
    ):
        raise MatmulChecklistError(
            "the runtime compiled another checklist library than the registry"
        )


def _measure(
    config: MatmulChecklistConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    names = admitted_names(backend.admissions)
    cases = []
    for index, shape in enumerate(config.correctness.shapes):
        case = check_case(
            backend,
            seed=config.seed,
            index=index,
            shape=shape,
            naive_group_size=config.naive.group_size,
            names=names,
        )
        cases.append(case.to_mapping())
        budget.check()
    points = benchmark_sweep(
        backend,
        seed=config.seed,
        shapes=timing.shapes,
        naive_group_size=config.naive.group_size,
        names=names,
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
        raise MatmulChecklistError("every sample must share one known timer")
    return {
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
    config: MatmulChecklistConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MatmulChecklistRuntime``, which
    raises ``MatmulChecklistUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MatmulChecklistConfig):
        raise TypeError("config must be a MatmulChecklistConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MatmulChecklistRuntime()
        _require_compiled(backend)
        run = _measure(config, backend, clock, load_average)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    real = type(backend) is MatmulChecklistRuntime
    measured = run["timer"] == TIMER and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    host = environment(backend)
    gains = speedup(
        run["points"], config.comparison.min_faster_ratio, host["torch_version"]
    )
    refusals = [
        entry.to_mapping()
        for entry in backend.admissions.values()
        if not entry.admitted
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
        "checklist": {**checklist_mapping(), "sha256": checklist_sha256()},
        "environment": host,
        "torch_mps_switches": {
            "source": SWITCH_SOURCE,
            "values": {name: backend.mps_switches[name] for name in MPS_SWITCHES},
        },
        "host_load": run["load"],
        "warnings": captured,
        "shaders": _shaders(backend),
        "refusals": refusals,
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": FLOAT64_REFERENCE,
            "replay": REPLAY,
            "relative_error": RELATIVE_ERROR,
            "bound": ERROR_BOUND,
            "gamma": GAMMA,
            "bound_source": BOUND_SOURCE,
            "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
            "rules": CORRECTNESS_RULES,
            "cases": run["cases"],
        },
        "benchmark": {
            "device": DEVICE,
            "flops_per_call": FLOPS_PER_CALL,
            "gflops": GFLOPS,
            "torch_equivalent": TORCH_EQUIVALENT,
            "synchronization": SYNCHRONIZATION,
            "agreement": AGREEMENT,
            "traffic_model": TRAFFIC_MODEL,
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
            run["cases"],
            run["points"],
            gains,
            [entry["name"] for entry in refusals],
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MatmulChecklistConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MatmulChecklistRuntime``, or any clock other
    than the host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(config, backend=backend, clock=clock, load_average=load_average)
    )
