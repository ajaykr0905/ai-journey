"""Run every scan case and sweep point under one budget, then seal a labeled report.

One run captures Python warnings from before the runtime imports torch,
builds the runtime, reads the load averages, checks every correctness case
on every admitted kernel and ``torch.cumsum``, times every sweep point,
reads the load averages again and derives the rates and ratios. One
``RunBudget`` on the run's clock covers the cases and the sweep, and a run
past it stops without a report. The budget starts after the kernels compile,
so the time to import torch on a busy host does not count.

The warning capture is the Day 76 capture. The report seals the declared
plan with its digest, and the config must name that digest, so a run cannot
test other hypotheses or other kernel texts than the declared ones.

Labels keep the evidence honest. Only the real ``ScanRuntime`` timed with
``time.perf_counter_ns`` makes a measurement, with the Day 73 scope
``apple_gpu_metal_this_host`` and the label ``host-specific Apple GPU (Metal)
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
from .gpu_ceiling_source import STREAM_COPY
from .gpu_ceiling_source import source_sha256 as ceiling_source_sha256
from .gpu_scan_benchmark import CHECK_RULE, TRAFFIC_MODEL, benchmark_sweep
from .gpu_scan_config import BYTES_PER_ELEMENT, GpuScanConfig
from .gpu_scan_correctness import check_case
from .gpu_scan_plan import COPY, VENDOR, plan_mapping, plan_sha256
from .gpu_scan_reference import BOUND_RULE, BOUND_SOURCE, ERROR_RULE, PATTERNS
from .gpu_scan_runtime import (
    COPY_FLOAT4_PER_THREAD,
    COPY_GROUP,
    ScanRuntime,
    admitted_names,
)
from .gpu_scan_source import (
    ADD_OFFSETS,
    KERNELS,
    GpuScanError,
    add_mapping,
    library_sha256,
    library_source,
)
from .gpu_scan_speedup import speedup
from .kernel_metal_benchmark import NS_PER_SECOND, RunBudget
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
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "gpu_scan_correctness_and_bandwidth"
REPORT_DIGEST_FIELD = DIGEST_FIELD
DEVICE = "mps"
REFERENCE = (
    "numpy.cumsum of the float64 inputs in order, rebuilt from the seed; S0 and "
    "S1 also against their float32 CPU replay, S2 and torch.cumsum against no "
    "replay"
)
CORRECTNESS_RULES = (
    "every kernel run keeps its sentinel guard tail and stays within its bound, "
    "and on small_integers equals the exact prefix sums, or the run stops; "
    "deterministic, replay_bits and every torch.cumsum value are measured"
)
BYTES_RULE = (
    "GB/s = bytes of one sample / median sample time in ns; 8 bytes for each "
    "element and call for the scans and torch.cumsum, 32 bytes for each float4 "
    "and call for the copy"
)
COPY_TEXT = (
    "stream_copy from the Day 74 ceiling source, unchanged: ceil(N / 4) float4 "
    "values from a zero-padded copy of the input, 4 for each thread in groups of "
    "256, as the pinned Day 74 config runs it"
)
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "warp_shuffle_executed": False,
    "vendor_kernel_known": False,
    "vendor_peak_claimed": False,
    "hardware_counters_used": False,
    "occupancy_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These are wall-clock timings on the Apple GPU of the host that made "
        "this report, at the recorded load averages. The scan kernels and the "
        "Day 74 copy came from torch.mps.compile_shader, and torch.cumsum ran on "
        "the MPS device. PyTorch does not report which kernel torch.cumsum ran, "
        "so the vendor number names the call, not a kernel. The SIMD-group "
        "functions of S2 are the Metal counterpart of CUDA warp shuffles, not "
        "the same instruction. No hardware counter was read, so a bound name is "
        "an inference from the timings and the same-run copy rate. No CUDA kernel "
        "ran, and no NVIDIA GPU was used. Each ratio compares two sides of one "
        "run on this host. These timings do not measure or predict CUDA, NVIDIA, "
        "distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "warp_shuffle_executed": False,
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
        "outputs that the backend returned. No CUDA kernel ran, and no NVIDIA "
        "GPU was used."
    ),
}


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
        "add": {**add_mapping(), "limits": backend.limits[ADD_OFFSETS].to_mapping()},
        "copy": {
            "kernel": STREAM_COPY,
            "text": COPY_TEXT,
            "source_sha256": ceiling_source_sha256(),
            "group_size": COPY_GROUP,
            "float4_per_thread": COPY_FLOAT4_PER_THREAD,
            "limits": backend.limits[COPY].to_mapping(),
        },
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
    sides = [output["side"] for output in cases[0]["outputs"]]
    outputs = [(case["index"], output) for case in cases for output in case["outputs"]]
    kernels = [name for name in sides if name != VENDOR]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "refused_kernels": refused,
        "max_bound_ratio": max(
            (
                output["bound_ratio"]
                for _, output in outputs
                if output["side"] != VENDOR
            ),
            default=None,
        ),
        "cases_with_other_bits": {
            name: [
                index
                for index, output in outputs
                if output["side"] == name and output["replay_bits"] is False
            ]
            for name in kernels
            if KERNELS[name].replay
        },
        "nondeterministic_cases": {
            name: [
                index
                for index, output in outputs
                if output["side"] == name and not output["deterministic"]
            ]
            for name in sides
        },
        "vendor_max_relative_error": max(
            output["max_relative_error"]
            for _, output in outputs
            if output["side"] == VENDOR
        ),
        "vendor_missed_exact_at": [
            index
            for index, output in outputs
            if output["side"] == VENDOR and output["exact_bits"] is False
        ],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "length": entry["length"],
                "best": None if entry["best"] is None else entry["best"]["kernel"],
                "best_to_vendor": None
                if entry["best"] is None
                else entry["best"]["to_vendor"]["ratio"],
                "best_fraction_of_copy": None
                if entry["best"] is None
                else entry["best"]["fraction_of_copy"],
                "vendor_gb_per_second": entry["vendor"]["median_gb_per_second"],
                "copy_gb_per_second": entry["copy"]["median_gb_per_second"],
            }
            for entry in gains["points"]
        ],
        "hypotheses": {
            record["kernel"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
                "no_prediction": len(record["no_prediction_at"]),
            }
            for record in gains["hypotheses"]
        },
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _require_compiled(backend: object) -> None:
    """Refuse a real runtime that compiled another library than the registry's."""

    if type(backend) is ScanRuntime and backend.library_source != library_source():
        raise GpuScanError(
            "the runtime compiled another scan library than the registry"
        )


def _measure(
    config: GpuScanConfig,
    backend: object,
    clock: Callable[[], int],
    load_average: Callable[[], tuple[float, float, float]],
) -> dict[str, object]:
    timing = config.benchmark
    before = read_load_average(load_average)
    budget = RunBudget(timing.max_runtime_seconds, clock)
    names = admitted_names(backend.admissions)
    cases = []
    for index, (length, pattern) in enumerate(config.correctness.cases):
        case = check_case(
            backend,
            seed=config.seed,
            index=index,
            length=length,
            pattern=pattern,
            names=names,
            runs=config.correctness.runs,
        )
        cases.append(case.to_mapping())
        budget.check()
    points = benchmark_sweep(
        backend,
        seed=config.seed,
        lengths=timing.lengths,
        names=names,
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
        raise GpuScanError("every sample must share one known timer")
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
    config: GpuScanConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab inside the warning capture and assemble the unsealed report.

    ``backend=None`` compiles the real ``ScanRuntime``, which raises
    ``GpuScanUnavailableError`` on a host without MPS.
    """

    if not isinstance(config, GpuScanConfig):
        raise TypeError("config must be a GpuScanConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch_imported = sys.modules.get("torch") is not None
        if backend is None:
            backend = ScanRuntime()
        _require_compiled(backend)
        run = _measure(config, backend, clock, load_average)
    captured = warning_records(
        caught, torch_imported=torch_imported, home=os.path.expanduser("~")
    )
    real = type(backend) is ScanRuntime
    measured = run["timer"] == TIMER and real
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    host = host_environment(backend.torch_version)
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
        "plan": {**plan_mapping(), "sha256": plan_sha256()},
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
            "reference": REFERENCE,
            "patterns": list(PATTERNS),
            "error_rule": ERROR_RULE,
            "bound_rule": BOUND_RULE,
            "bound_source": BOUND_SOURCE,
            "sentinel_float32_bits": FLOAT32_SENTINEL_BITS,
            "rules": CORRECTNESS_RULES,
            "runs": config.correctness.runs,
            "cases": run["cases"],
        },
        "benchmark": {
            "device": DEVICE,
            "bytes_per_element": BYTES_PER_ELEMENT,
            "bytes_rule": BYTES_RULE,
            "synchronization": SYNCHRONIZATION,
            "traffic_model": TRAFFIC_MODEL,
            "check_rule": CHECK_RULE,
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
    config: GpuScanConfig,
    *,
    backend: object | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
    load_average: Callable[[], tuple[float, float, float]] = os.getloadavg,
) -> dict[str, object]:
    """Run the lab and return the report sealed with its SHA-256 digest.

    Any backend other than ``ScanRuntime``, or any clock other than the host
    timer, makes the report synthetic.
    """

    return seal_report(
        build_report(config, backend=backend, clock=clock, load_average=load_average)
    )
