"""Run every case and point under one budget, then seal a labeled tiled matmul report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, checks every correctness case
on the naive kernel and the admitted variants, times every sweep point,
reads the load averages again, and derives the three numbers of each point.
One ``RunBudget`` on the run's clock covers the cases and the sweep, and a
run past it stops without a report. The budget starts after the kernels
compile, so the time to import torch on a busy host does not count.

The capture is ``warnings.catch_warnings(record=True)`` with
``simplefilter("always")`` around the whole run. It thus holds the one-time
warnings that PyTorch raises on first use, such as an MPS matmul that falls
back to a Metal kernel. Equal warnings are kept once with a count, a message
keeps at most ``MAX_WARNING_TEXT`` characters with the home directory as
``~``, and a file keeps only its base name. The record states whether torch
was already imported when the capture began; in a fresh CLI process it was
not, so the capture holds every warning that torch raised in the process.
The list is sealed even when it is empty.

Labels keep the evidence honest. Only the real ``MatmulTiledRuntime`` timed
with ``time.perf_counter_ns`` makes a measurement, with the Day 73 scope
``apple_gpu_metal_this_host`` and label ``host-specific Apple GPU (Metal)
measurement``. Any other backend or clock makes the report synthetic. The
SHA-256 seal follows ``memory_access_report``: canonical JSON of every field
but the digest. It detects drift and tampering; it is not a signature and
cannot prove that a timing was observed.
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
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    REFERENCE,
    RELATIVE_ERROR,
)
from .matmul_metal_report import FLOPS_PER_CALL, GFLOPS, TORCH_EQUIVALENT, environment
from .matmul_metal_source import KERNELS, source_sha256
from .matmul_tiled_benchmark import AGREEMENT, LOADS_PER_OUTPUT, benchmark_sweep
from .matmul_tiled_comparison import comparison
from .matmul_tiled_config import MatmulTiledConfig
from .matmul_tiled_correctness import check_case
from .matmul_tiled_runtime import MPS_SWITCHES, MatmulTiledRuntime
from .matmul_tiled_source import LOOP_STATEMENT, MatmulTiledError
from .matmul_tiled_variants import library_source, text_sha256
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_tiled_correctness_and_gflops"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
MAX_WARNINGS = 32
MAX_WARNING_TEXT = 2000
WARNING_CAPTURE = (
    "warnings.catch_warnings(record=True) with simplefilter('always'), from "
    "before the runtime imports torch to the end of the run; equal warnings "
    "are kept once with a count"
)
SWITCH_SOURCE = (
    "os.environ, read before the runtime imports torch; null means unset, and "
    "a value that is not a short token is 'unknown'"
)
CORRECTNESS_RULES = (
    "every output keeps its sentinel guard tail and stays within the bound; "
    "replay_bits is measured: true only when no element differs from the "
    "replay"
)
REPLAY = (
    "replay_bits compares an output with the CPU replay of sum = fma(a[i, k], "
    "b[k, j], sum) for k = 0 to K - 1, one float32 rounding per step; the "
    f"tiled kernels run that chain as `{LOOP_STATEMENT}`"
)
SYNCHRONIZATION = (
    "torch.mps.synchronize() before the first and before the second clock "
    "reading of every sample; untimed warmups of each side first; repeat r "
    "starts with side r mod S, so the order shifts by one side every repeat"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The naive and tiled "
        "kernels came from torch.mps.compile_shader, and torch.matmul ran on "
        "the MPS device. PyTorch does not report which kernel torch.matmul ran, "
        "so the vendor number names the call, not a kernel. No CUDA kernel ran, "
        "no NVIDIA GPU was used, and cuBLAS was not measured. Each ratio "
        "compares two sides of one run on this host. These timings do not "
        "measure or predict CUDA, NVIDIA, cuBLAS, distributed or production "
        "performance."
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


def _text(value: str, home: str) -> str:
    if home and home != os.sep:
        value = value.replace(home, "~")
    return value[:MAX_WARNING_TEXT]


def warning_records(
    caught: list[warnings.WarningMessage], *, torch_imported: bool, home: str
) -> dict[str, object]:
    """Group the captured warnings into the sealed record, without host paths."""

    if type(torch_imported) is not bool:
        raise TypeError("torch_imported must be a bool")
    grouped: dict[tuple[str, str, str, int], int] = {}
    for item in caught:
        key = (
            item.category.__name__,
            _text(str(item.message), home),
            os.path.basename(str(item.filename))[:MAX_WARNING_TEXT],
            int(item.lineno or 0),
        )
        grouped[key] = grouped.get(key, 0) + 1
    records = [
        {
            "category": category,
            "message": message,
            "file": name,
            "line": line,
            "count": count,
        }
        for (category, message, name, line), count in grouped.items()
    ]
    return {
        "capture": WARNING_CAPTURE,
        "torch_imported_before_capture": torch_imported,
        "count": len(caught),
        "distinct": len(records),
        "records": records[:MAX_WARNINGS],
    }


def _shaders(backend: object, config: MatmulTiledConfig) -> dict[str, object]:
    variants = config.tiled_variants()
    source = library_source(variants)
    return {
        "naive": {
            "source_sha256": source_sha256(),
            "kernels": [
                {**spec.to_mapping(), "limits": backend.limits[name].to_mapping()}
                for name, spec in KERNELS.items()
            ],
        },
        "tiled": {
            "source": source,
            "source_sha256": text_sha256(source),
            "variants": [
                {
                    **variant.to_mapping(),
                    "limits": backend.limits[variant.name].to_mapping(),
                    "admission": backend.admissions[variant.name].to_mapping(),
                }
                for variant in variants
            ],
        },
    }


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    numbers: dict[str, object],
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
        "three_numbers": [
            {
                "m": entry["m"],
                "n": entry["n"],
                "k": entry["k"],
                "naive_gflops_per_second": entry["naive"]["median_gflops_per_second"],
                "tiled_side": None
                if entry["tiled"] is None
                else entry["tiled"]["side"],
                "tiled_gflops_per_second": None
                if entry["tiled"] is None
                else entry["tiled"]["median_gflops_per_second"],
                "vendor_gflops_per_second": entry["vendor"]["median_gflops_per_second"],
                "tiled_over_naive": None
                if entry["tiled_over_naive"] is None
                else entry["tiled_over_naive"]["verdict"],
                "vendor_over_tiled": None
                if entry["vendor_over_tiled"] is None
                else entry["vendor_over_tiled"]["verdict"],
            }
            for entry in numbers["points"]
        ],
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _require_compiled(backend: object, config: MatmulTiledConfig) -> None:
    """Refuse a real runtime whose tiled library is not the config's library.

    The report records the library text that the config names, so a measured
    report must come from a runtime that compiled exactly that text.
    """

    if type(backend) is MatmulTiledRuntime and backend.tiled_source != library_source(
        config.tiled_variants()
    ):
        raise MatmulTiledError(
            "the runtime compiled another tiled library than the config names"
        )


def _measure(
    config: MatmulTiledConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    variants = config.tiled_variants()
    admitted = tuple(
        variant for variant in variants if backend.admissions[variant.name].admitted
    )
    cases = []
    for index, shape in enumerate(config.correctness.shapes):
        case = check_case(
            backend,
            seed=config.seed,
            index=index,
            shape=shape,
            naive_group_size=config.naive.group_size,
            variants=admitted,
        )
        cases.append(case.to_mapping())
        budget.check()
    points = benchmark_sweep(
        backend,
        seed=config.seed,
        shapes=timing.shapes,
        naive_group_size=config.naive.group_size,
        variants=admitted,
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
        raise MatmulTiledError("every sample must share one known timer")
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
    config: MatmulTiledConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``MatmulTiledRuntime``, which raises
    ``MatmulTiledUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, MatmulTiledConfig):
        raise TypeError("config must be a MatmulTiledConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = MatmulTiledRuntime(config.tiled_variants())
        _require_compiled(backend, config)
        run = _measure(config, backend, clock, load_average)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    real = type(backend) is MatmulTiledRuntime
    measured = run["timer"] == TIMER and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    host = environment(backend)
    tiles = {variant.name: variant.tile for variant in config.tiled_variants()}
    numbers = comparison(
        run["points"],
        config.variants,
        tiles,
        config.comparison.min_faster_ratio,
        host["torch_version"],
    )
    refusals = [
        entry.to_mapping()
        for entry in (backend.admissions[name] for name in config.variants)
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
        "environment": host,
        "torch_mps_switches": {
            "source": SWITCH_SOURCE,
            "values": {name: backend.mps_switches[name] for name in MPS_SWITCHES},
        },
        "host_load": run["load"],
        "warnings": captured,
        "shaders": _shaders(backend, config),
        "refusals": refusals,
        "planned_work": config.planned_work().to_mapping(),
        "correctness": {
            "reference": REFERENCE,
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
            "loads_per_output": LOADS_PER_OUTPUT,
            "points": run["points"],
        },
        "comparison": numbers,
        "run": {
            "max_runtime_ns": config.benchmark.max_runtime_seconds * NS_PER_SECOND,
            "elapsed_ns": run["elapsed_ns"],
            "timed_ns": run["timed_ns"],
        },
        "summary": _summary(
            label,
            run["cases"],
            run["points"],
            numbers,
            [entry["name"] for entry in refusals],
            captured,
            run["load"],
        ),
    }


def measure_report(
    config: MatmulTiledConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``MatmulTiledRuntime``, or any clock other than the
    host timer, makes the report synthetic.
    """

    return seal_report(
        build_report(config, backend=backend, clock=clock, load_average=load_average)
    )
