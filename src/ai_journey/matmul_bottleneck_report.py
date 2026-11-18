"""The sealed bottleneck report of the naive Metal matmul.

The report names the top bottleneck of each naive kernel at each Day 74 sweep
point, and of each kernel over the sweep. It holds every input of each
decision: the steady-state sample time and batch of the Day 74 gap, the
one-call medians of the timing input, the two measured ceilings and the
intensity of the pinned traffic model. It also holds the full config, so a
verifier can repeat each decision without the sources. It adds the warp
model of both mappings at each shape and the verdict on x over rows.

A report is a measurement only when the gap and the timing input both are,
and it keeps the load averages of the ceiling run and of the timing run. Its
evidence boundary states that the names are inferences: no hardware counter,
Nsight or Instruments trace took part. The SHA-256 seal covers canonical JSON
of every other field; it detects drift and tampering but is not a signature.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .kernel_metal_report import (
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .matmul_bottleneck_config import BottleneckConfig, MatmulBottleneckError
from .matmul_bottleneck_mapping import MODEL_SCOPE, VERDICTS, compare_mappings
from .matmul_bottleneck_rule import BOTTLENECKS, INDETERMINATE, REASONS, classify
from .matmul_bottleneck_sources import Day74Sources, JoinedPoint, join_points
from .matmul_bottleneck_timing import TIMING_KIND, TimingInput
from .matmul_metal_source import KERNELS
from .matmul_roofline_gap import CEILING_SOURCE, MATMUL_SOURCE, NAIVE_SIDES
from .matmul_roofline_gap import REPORT_KIND as GAP_KIND
from .matmul_metal_report import REPORT_KIND as MATMUL_KIND
from .memory_access_report import DIGEST_FIELD, seal_report

REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_bottleneck"
GAP_SOURCE = "matmul_roofline_gap"
TIMING_SOURCE = "timing"
TIE = "tie"
DEFINITIONS = {
    "steady_ns_per_call": (
        "the Day 74 median sample time divided by its calls per sample. With many "
        "calls per sample it approaches the time per call of back-to-back calls; "
        "with one call it also holds that call's synchronization"
    ),
    "rate_gflops_per_second": "flops_per_sample divided by median_sample_ns",
    "overhead_share": "1 - steady_ns_per_call / median_host_ns",
    "host_outside_gpu_share": "1 - median_gpu_ns / median_host_ns",
    "gpu_excess_share": "(median_gpu_ns - steady_ns_per_call) / median_host_ns",
    "dispatch_share": "median_dispatch_ns / median_host_ns",
    "fixed_share_bound": (
        "the smallest median_host_ns of the kernel over the whole timing input, "
        "divided by median_host_ns: the largest share that a fixed one-call cost "
        "can take if it does not depend on the shape"
    ),
    "gpu_to_steady_ratio": "median_gpu_ns / steady_ns_per_call",
    "bandwidth_roof_gflops_per_second": (
        "the measured bandwidth ceiling in GB/s times the intensity of the "
        "config's traffic model in FLOPs per byte"
    ),
    "lower_roof_share": (
        "the rate divided by the lower of the FMA ceiling and the bandwidth roof"
    ),
    "threadgroups": "the threadgroups of each kernel's launch, x by y",
    "model_sector_ratio": (
        "32-byte sectors of the x-over-rows probe warp divided by those of the "
        "x-over-columns probe warp, in the Day 74 per-warp model"
    ),
    "top_bottleneck": (
        "the name that most points of a kernel get; a tie between names gives "
        "indeterminate"
    ),
}
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "inference_from_timings": True,
    "hardware_counters_used": False,
    "nsight_used": False,
    "instruments_used": False,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "Each bottleneck name is an inference from host-specific Apple GPU "
        "(Metal) timings and from the ceilings that the Day 74 ceiling lab "
        "measured on the same host. No hardware counters, Nsight or Instruments "
        "were used, so no stall, cache hit or occupancy was observed. The "
        "mapping explanation comes from the Day 74 per-warp access model, not "
        "from measured traffic. No CUDA kernel ran, and no NVIDIA GPU was used. "
        "These names do not describe CUDA, NVIDIA, distributed or production "
        "performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "inference_from_timings": True,
    "hardware_counters_used": False,
    "nsight_used": False,
    "instruments_used": False,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "At least one input is synthetic, so each name in this report comes "
        "from synthetic numbers. It describes no host. No hardware counters, "
        "Nsight or Instruments were used. No CUDA kernel ran, and no NVIDIA GPU "
        "was used."
    ),
}


def reasons_mapping() -> dict[str, dict[str, str]]:
    """Return the reason codes of the rule, in the order of its steps."""

    return {
        code: {"bottleneck": name, "explanation": text}
        for code, (name, text) in REASONS.items()
    }


def top_bottleneck(names: list[str]) -> dict[str, object]:
    """Count the names of one kernel and pick the most common; a tie claims none."""

    counts = Counter(names)
    most = max(counts.values())
    leaders = sorted(name for name in counts if counts[name] == most)
    return {
        "counts": {name: counts.get(name, 0) for name in BOTTLENECKS},
        "top": leaders[0] if len(leaders) == 1 else INDETERMINATE,
        "top_points": most,
        "reason": "most_points" if len(leaders) == 1 else TIE,
    }


def _sources(
    sources: Day74Sources, timing: TimingInput, used: int
) -> dict[str, object]:
    gap = sources.gap_report
    ceiling = gap["sources"][CEILING_SOURCE]
    return {
        MATMUL_SOURCE: {
            "kind": MATMUL_KIND,
            "report_sha256": sources.matmul_report[DIGEST_FIELD],
        },
        CEILING_SOURCE: {
            "kind": ceiling["kind"],
            "report_sha256": ceiling["report_sha256"],
            "load_average_before": list(ceiling["load_average_before"]),
            "load_average_after": list(ceiling["load_average_after"]),
        },
        GAP_SOURCE: {
            "kind": GAP_KIND,
            "report_sha256": gap[DIGEST_FIELD],
            "measured": gap["measured"],
            "label": gap["label"],
        },
        TIMING_SOURCE: {
            "kind": TIMING_KIND,
            "input_sha256": timing.digest(),
            "measured": timing.measured,
            "label": timing.label,
            "producer": timing.producer,
            "repeats": timing.repeats,
            "load_average_before": list(timing.load_average_before),
            "load_average_after": list(timing.load_average_after),
            "entries": len(timing.entries),
            "entries_used": used,
            "fastest_calls": {
                kernel: timing.fastest_call(kernel).to_mapping()
                for kernel in NAIVE_SIDES
            },
        },
    }


def _kernel_entry(item: JoinedPoint, decision: Mapping[str, object]) -> dict:
    return {
        "mapping": KERNELS[item.kernel].mapping,
        "inputs": {
            "median_gpu_ns": item.timing.gpu_ns,
            "median_host_ns": item.timing.host_ns,
            "median_dispatch_ns": item.timing.dispatch_ns,
        },
        "decision": decision,
    }


def _point(
    gap_point: Mapping[str, object],
    matmul_point: Mapping[str, object],
    items: list[JoinedPoint],
    sources: Day74Sources,
    config: BottleneckConfig,
) -> dict[str, object]:
    model = config.rule.intensity_model
    shape = items[0].shape
    sides = gap_point["sides"]
    widths = sources.simd_widths
    if len(set(widths.values())) != 1:
        raise MatmulBottleneckError("the two naive pipelines report other SIMD widths")
    (width,) = set(widths.values())
    return {
        "index": gap_point["index"],
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "flops_per_call": gap_point["flops_per_call"],
        "calls_per_sample": gap_point["calls_per_sample"],
        "intensity": {
            "model": model,
            "flops_per_byte": dict(gap_point["intensities"][model]["flops_per_byte"]),
        },
        "threadgroups": {
            kernel: list(matmul_point["launches"][KERNELS[kernel].mapping]["groups"])
            for kernel in NAIVE_SIDES
        },
        "median_sample_ns": {
            kernel: sides[kernel]["median_sample_ns"] for kernel in NAIVE_SIDES
        },
        "kernels": {
            item.kernel: _kernel_entry(
                item, classify(item.inputs, config.rule).to_mapping()
            )
            for item in items
        },
        "mapping": compare_mappings(
            shape,
            sources.group_size,
            width,
            {
                kernel: sides[kernel]["median_gflops_per_second"]
                for kernel in NAIVE_SIDES
            },
            config.mapping,
        ),
    }


def _summary(label: str, points: list[dict[str, object]]) -> dict[str, object]:
    verdicts = Counter(point["mapping"]["verdict"] for point in points)
    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            kernel: top_bottleneck(
                [point["kernels"][kernel]["decision"]["bottleneck"] for point in points]
            )
            for kernel in NAIVE_SIDES
        },
        "mapping_verdicts": {verdict: verdicts.get(verdict, 0) for verdict in VERDICTS},
    }


def build_report(
    sources: Day74Sources, timing: TimingInput, config: BottleneckConfig
) -> dict[str, object]:
    """Classify every naive kernel at every gap point and seal the report."""

    if type(config) is not BottleneckConfig:
        raise TypeError("config must be a BottleneckConfig")
    joined = join_points(sources, timing, config.rule)
    gap = sources.gap_report
    matmul_points = sources.matmul_report["benchmark"]["points"]
    points = []
    for gap_point in gap["points"]:
        index = gap_point["index"]
        items = [item for item in joined if item.index == index]
        points.append(_point(gap_point, matmul_points[index], items, sources, config))
    measured = gap["measured"] is True and timing.measured
    label = MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL
    ceilings = gap["ceilings"]
    return seal_report(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "measured": measured,
            "measurement_scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
            "label": label,
            "evidence_boundary": dict(
                MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY
            ),
            "config": config.to_mapping(),
            "config_sha256": config.fingerprint(),
            "sources": _sources(sources, timing, len(joined)),
            "environment": dict(gap["environment"]),
            "ceilings": {
                "fma_gflops_per_second": ceilings["fma_gflops_per_second"],
                "bandwidth_gb_per_second": ceilings["bandwidth_gb_per_second"],
                "fma_scales_with_work": ceilings["fma_scales_with_work"],
            },
            "geometry": {
                "group_size": list(sources.group_size),
                "simd_widths": sources.simd_widths,
            },
            "model_scope": MODEL_SCOPE,
            "definitions": dict(DEFINITIONS),
            "reasons": reasons_mapping(),
            "points": points,
            "summary": _summary(label, points),
        }
    )
