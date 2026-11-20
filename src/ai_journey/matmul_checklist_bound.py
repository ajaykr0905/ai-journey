"""Is each checklist kernel memory bound or compute bound? The measured answer.

Day 77 asks which limit each matmul meets. This module joins three verified
reports of one host and never uses a vendor number:

- the checklist report: the median, p25 and p75 sample times and the rate of
  each side at each sweep shape;
- the gpu-ceilings report: the FMA and bandwidth ceilings that the ceiling
  lab's own kernels reached;
- the one-call timing report: the event, host and dispatch medians of one
  call of each side, and the fastest host call of each side.

For each sweep shape and side the report gives the share of the measured
FMA ceiling, the modeled device and threadgroup traffic of the kernel from
``matmul_checklist_bound_traffic``, the attainable rate of
``MeasuredRoofline`` at the modeled intensity and at the ideal-reuse
intensity, and the Day 75 name from ``matmul_bottleneck_rule.classify`` with
the ideal-reuse roof, as the Day 76 gap does. No kernel moves fewer bytes
than ideal reuse, so that roof bounds every side. PyTorch does not name the
kernel of ``torch.matmul``, so no traffic model applies to that side.

For each checklist step and each kernel that it adds, the report then puts
the modeled change beside the measured one: the ratios of device bytes and
load operations and the threadgroup reads for each fma of the base and the
new kernel, beside the step ratio and verdict of the Day 76 rule.

A share of a modeled roof above 1 shows that the kernel moved fewer device
bytes than the model counts, probably because a cache served repeated
loads, or that the bandwidth ceiling is below the true limit. A bottleneck
name is an inference from timings and measured ceilings: no hardware
counter took part, so no stall, cache hit, bank conflict or occupancy was
observed.

Every source passes its own verifier first, so a tampered source sets no
number. All three must name the same interpreter, libraries and platform.
The checklist and timing reports must also come from one config, with the
same float32 matmul precision, MPS switches and admitted kernels. The report
is a measurement only when all three sources are, and it keeps the load
averages of each run, because the three runs are separate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import TypeVar

from .gpu_ceiling_io import load_report as load_ceiling_report
from .gpu_ceiling_roofline import MeasuredRoofline, _from_summary
from .gpu_ceiling_source import GpuCeilingError
from .gpu_ceiling_verify import GpuCeilingVerificationError
from .gpu_ceiling_verify import verify_report as verify_ceiling_report
from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .matmul_bottleneck_config import BottleneckConfig
from .matmul_bottleneck_report import reasons_mapping, top_bottleneck
from .matmul_bottleneck_rule import PointInputs, classify
from .matmul_checklist_bound_timing_io import load_timing_report
from .matmul_checklist_bound_timing_verify import verify_timing_report
from .matmul_checklist_bound_traffic import (
    MODEL_TEXT,
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
    device_bytes,
    intensity,
    kernel_traffic,
)
from .matmul_checklist_io import load_report as load_checklist_report
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_steps import CHECKLIST, checklist_sha256
from .matmul_checklist_verify import MatmulChecklistVerificationError
from .matmul_checklist_verify import verify_report as verify_checklist_report
from .matmul_metal_report import MATMUL_PRECISION_FIELD
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_model_traffic import ratio_mapping
from .matmul_roofline_gap import flops_per_byte
from .matmul_tiled_comparison import RATIO_RULE
from .matmul_tiled_comparison import step as ratio_step
from .matmul_tiled_gap import _ideal, _identity, _one_call
from .memory_access_report import seal_report
from .memory_access_tiling import MatmulShape

Checked = TypeVar("Checked")
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_checklist_bound"
CHECKLIST_SOURCE = "matmul_checklist"
CEILING_SOURCE = "gpu_ceilings"
TIMING_SOURCE = "one_call_timing"
SOURCE_NAMES = (CHECKLIST_SOURCE, CEILING_SOURCE, TIMING_SOURCE)
RULE_MODEL = "ideal_reuse"
BOUND_ENVIRONMENT_FIELDS = (*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD)
DEFINITIONS = {
    "steady_ns_per_call": (
        "the median sample time of the checklist report divided by its calls "
        "per sample; with one call per sample it also holds that call's "
        "synchronization"
    ),
    "median_gflops_per_second": (
        "flops_per_call times calls_per_sample divided by median_sample_ns, as "
        "the checklist report states it"
    ),
    "fma_ceiling_share": "median GFLOP/s of a side divided by the FMA ceiling",
    "traffic": (
        "the counts of matmul_checklist_bound_traffic for the side's kernel: "
        "no_reuse for a Day 74 kernel, blocked for a tiled kernel; null for "
        "torch.matmul, whose kernel PyTorch does not name"
    ),
    "ideal_reuse": (
        "4 * (M * K + K * N + M * N) bytes: each element moves once, and no "
        "kernel moves fewer"
    ),
    "bandwidth_roof_gflops_per_second": (
        "the measured bandwidth ceiling in GB/s times an intensity in FLOPs per " "byte"
    ),
    "attainable_gflops_per_second": (
        "MeasuredRoofline: the lower of the FMA ceiling and the bandwidth roof"
    ),
    "attainable_share": (
        "median GFLOP/s of a side divided by the attainable rate at its modeled "
        "intensity; above 1, the kernel moved fewer device bytes than the model "
        "counts, or the bandwidth ceiling is below the true limit"
    ),
    "modeled_device_gb_per_second": (
        "the modeled device bytes times the median GFLOP/s divided by the "
        "FLOPs: the device bandwidth that the rate needs if the model holds"
    ),
    "modeled_threadgroup_reads_per_ns": (
        "the modeled threadgroup reads times the median GFLOP/s divided by the "
        "FLOPs; null for a kernel with no threadgroup memory"
    ),
    "ideal_reuse_share": (
        "median GFLOP/s of a side divided by the attainable rate at the "
        "ideal-reuse intensity"
    ),
    "one_call": (
        "the mps_event, host_synchronized and host_dispatch medians of one call "
        "of the side at the same shape, from the one-call timing report"
    ),
    "fastest_calls": (
        "the smallest host_synchronized median of each side over the floor and "
        "every point of the one-call timing report"
    ),
    "decision": (
        "matmul_bottleneck_rule.classify with the ideal-reuse bandwidth roof, "
        "the FMA ceiling and the thresholds of the rule config"
    ),
    "steps": (
        "for each kernel that a checklist step adds: the ratios of its modeled "
        "device bytes and load operations to those of the step's base, the "
        "threadgroup reads for each fma of both, and the step ratio and verdict "
        "of the Day 76 rule from the checklist samples; a kernel that did not "
        "run has no measured values"
    ),
    "top_bottleneck": (
        "the name that most points of a side get; a tie between names gives "
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
    "occupancy_measured": False,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These numbers join three verified reports from the Apple GPU of one "
        "host: the checklist rates, the ceilings that the ceiling lab's own "
        "kernels reached, and one-call timings. Each is a host-specific Apple "
        "GPU (Metal) measurement from a separate run. A ceiling is a measured "
        "rate, not a vendor peak or a hardware limit, and the traffic comes "
        "from a model, not from measured bytes. Each bottleneck name is an "
        "inference from timings and measured ceilings. No hardware counters, "
        "Nsight or Instruments were used, so no stall, cache hit, bank conflict "
        "or occupancy was observed. No CUDA kernel ran, no NVIDIA GPU was used, "
        "and cuBLAS was not measured. These numbers do not measure or predict "
        "CUDA, NVIDIA, cuBLAS, distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "inference_from_timings": True,
    "hardware_counters_used": False,
    "nsight_used": False,
    "instruments_used": False,
    "occupancy_measured": False,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "cublas_measured": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "At least one source report is synthetic, so each rate, share and name "
        "in this report comes from synthetic numbers. It describes no host. No "
        "hardware counters were used, no CUDA kernel ran, no NVIDIA GPU was "
        "used, and cuBLAS was not measured."
    ),
}
_REJECTED = (
    MatmulChecklistBoundVerificationError,
    MatmulChecklistVerificationError,
    GpuCeilingVerificationError,
)
_INVALID = (MatmulChecklistError, GpuCeilingError)


def _checked(name: str, verify: Callable[..., Checked], *arguments: object) -> Checked:
    """Run a source's own loader or verifier; name the source in any failure.

    A source that was read but failed its checks is a rejected report, and
    anything else is bad input, so callers keep the two exit codes apart.
    """

    try:
        return verify(*arguments)
    except _REJECTED as error:
        raise MatmulChecklistBoundVerificationError(
            f"{name} report: {error}"
        ) from error
    except _INVALID as error:
        raise MatmulChecklistBoundError(f"{name} report: {error}") from error


def _pair(
    checklist: Mapping[str, object],
    ceilings: Mapping[str, object],
    timing: Mapping[str, object],
    timing_summary: Mapping[str, object],
) -> None:
    """Refuse sources from other software, another config or other kernels."""

    environments = (
        checklist["environment"],
        ceilings["environment"],
        timing["environment"],
    )
    differing = [
        field
        for field in ENVIRONMENT_FIELDS
        if len({environment[field] for environment in environments}) != 1
    ]
    if differing:
        raise MatmulChecklistBoundError(
            "the three reports name different software or platforms: "
            + ", ".join(differing)
        )
    precision = MATMUL_PRECISION_FIELD
    if checklist["environment"][precision] != timing["environment"][precision]:
        raise MatmulChecklistBoundError(
            "the checklist and timing reports ran at other float32 matmul precisions"
        )
    if checklist["config_sha256"] != timing["config_sha256"]:
        raise MatmulChecklistBoundError(
            "the timing report was not made from the checklist report's config"
        )
    if (
        checklist["torch_mps_switches"]["values"]
        != timing["torch_mps_switches"]["values"]
    ):
        raise MatmulChecklistBoundError(
            "the checklist and timing reports set other MPS switches"
        )
    refused = [entry["name"] for entry in checklist["refusals"]]
    if refused != timing_summary["refused_variants"]:
        raise MatmulChecklistBoundError(
            "the checklist and timing runs admitted other kernels: "
            f"{refused} against {timing_summary['refused_variants']}"
        )


def _traffic(
    shape: MatmulShape, side: str, rate: float, roofline: MeasuredRoofline
) -> dict[str, object] | None:
    """Model one kernel side and set its roof; ``torch.matmul`` has no model."""

    if side == TORCH_MATMUL:
        return None
    model = kernel_traffic(side, shape)
    mapping = model.to_mapping()
    value = flops_per_byte(intensity(model))
    below = roofline.below_ceiling(rate, value)
    shared = mapping["threadgroup"]
    flops = shape.flops
    return {
        **mapping,
        "roof": {
            "bandwidth_roof_gflops_per_second": roofline.bandwidth_gb_per_second
            * value,
            "attainable_gflops_per_second": below["attainable_gflops_per_second"],
            "attainable_share": below["fraction_of_ceiling"],
            "bound": below["bound"],
            "modeled_device_gb_per_second": device_bytes(model) * rate / flops,
            "modeled_threadgroup_reads_per_ns": None
            if shared is None
            else shared["read_elements"] * rate / flops,
        },
    }


def _point(
    checklist_point: Mapping[str, object],
    timing_point: Mapping[str, object],
    sides: list[str],
    roofline: MeasuredRoofline,
    scales: bool,
    fastest: Mapping[str, Mapping[str, int]],
    config: BottleneckConfig,
) -> dict[str, object]:
    shape = MatmulShape(
        checklist_point["m"], checklist_point["n"], checklist_point["k"]
    )
    index = checklist_point["index"]
    if [timing_point[axis] for axis in "mnk"] != [shape.m, shape.n, shape.k]:
        raise MatmulChecklistBoundError(
            f"point {index}: the timing report timed another shape"
        )
    if checklist_point["side_order"] != sides:
        raise MatmulChecklistBoundError(
            f"point {index}: the checklist and timing runs timed other sides"
        )
    recorded = checklist_point["sides"]
    calls = {recorded[side]["calls_per_sample"] for side in sides}
    if len(calls) != 1:
        raise MatmulChecklistBoundError(
            f"point {index}: the sides ran different calls per sample"
        )
    (calls_per_sample,) = calls
    ideal = _ideal(shape, roofline)
    entries = {}
    for side in sides:
        timing = recorded[side]["timing"]
        median = timing["median_ns"]
        rate = recorded[side]["median_gflops_per_second"]
        one_call = _one_call(timing_point, side)
        inputs = PointInputs(
            steady_ns_per_call=median / calls_per_sample,
            rate_gflops_per_second=rate,
            gpu_ns=one_call["median_gpu_ns"],
            host_ns=one_call["median_host_ns"],
            dispatch_ns=one_call["median_dispatch_ns"],
            fma_ceiling_gflops_per_second=roofline.compute_gflops_per_second,
            fma_scales_with_work=scales,
            bandwidth_roof_gflops_per_second=ideal["bandwidth_roof_gflops_per_second"],
            fixed_cost_bound_ns=fastest[side]["median_host_ns"],
        )
        entries[side] = {
            "median_sample_ns": median,
            "p25_sample_ns": timing["p25_ns"],
            "p75_sample_ns": timing["p75_ns"],
            "steady_ns_per_call": inputs.steady_ns_per_call,
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / roofline.compute_gflops_per_second,
            "ideal_reuse_share": rate / ideal["attainable_gflops_per_second"],
            "traffic": _traffic(shape, side, rate, roofline),
            "one_call": one_call,
            "decision": classify(inputs, config.rule).to_mapping(),
        }
    return {
        "index": index,
        **shape.to_mapping(),
        "flops_per_call": shape.flops,
        "calls_per_sample": calls_per_sample,
        "ideal_reuse": ideal,
        "sides": entries,
    }


def _reads(name: str, shape: MatmulShape) -> dict[str, object] | None:
    shared = kernel_traffic(name, shape).to_mapping()["threadgroup"]
    return None if shared is None else shared["reads_per_fma"]


def _pick(entry: Mapping[str, object] | None, *path: str) -> object:
    """Follow ``path`` into a side entry; a missing side or model gives None."""

    for key in path:
        if entry is None:
            return None
        entry = entry[key]
    return entry


def _step_point(
    point: dict[str, object],
    checklist_sides: Mapping[str, object],
    base: str,
    variant: str,
    predicted: tuple[str, ...],
    minimum: float,
) -> dict[str, object]:
    shape = MatmulShape(point["m"], point["n"], point["k"])
    first, second = kernel_traffic(variant, shape), kernel_traffic(base, shape)
    entries = point["sides"]
    ran = variant in entries and base in entries
    measured = None
    if ran:
        ratio = ratio_step(checklist_sides[variant], checklist_sides[base], minimum)
        measured = {
            "ratio": ratio["ratio"],
            "iqr_separated": ratio["iqr_separated"],
            "verdict": ratio["verdict"],
            "held": ratio["verdict"] in predicted,
        }
    sides = {"base": entries.get(base), "variant": entries.get(variant)}
    return {
        "index": point["index"],
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "model": {
            "device_bytes_ratio": ratio_mapping(
                Fraction(device_bytes(first), device_bytes(second))
            ),
            "load_operations_ratio": ratio_mapping(
                Fraction(first.load_operations, second.load_operations)
            ),
            "threadgroup_reads_per_fma": {
                "base": _reads(base, shape),
                "variant": _reads(variant, shape),
            },
        },
        "measured": measured,
        "fma_ceiling_share": {
            role: _pick(entry, "fma_ceiling_share") for role, entry in sides.items()
        },
        "attainable_share": {
            role: _pick(entry, "traffic", "roof", "attainable_share")
            for role, entry in sides.items()
        },
        "bottleneck": {
            role: _pick(entry, "decision", "bottleneck")
            for role, entry in sides.items()
        },
    }


def _steps(
    points: list[dict[str, object]],
    checklist_points: list[Mapping[str, object]],
    minimum: float,
) -> list[dict[str, object]]:
    """Set the modeled change of each step kernel beside its measured change."""

    return [
        {
            "step": step.step,
            "item": step.item,
            "variant": variant,
            "base": step.base,
            "predicted": list(step.predicted(variant)),
            "points": [
                _step_point(
                    point,
                    source["sides"],
                    step.base,
                    variant,
                    step.predicted(variant),
                    minimum,
                )
                for point, source in zip(points, checklist_points)
            ],
        }
        for step in CHECKLIST
        for variant in step.variants
    ]


def _span(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def _summary(
    label: str,
    sides: list[str],
    points: list[dict[str, object]],
    steps: list[dict[str, object]],
) -> dict[str, object]:
    """Name the top bottleneck of each side and the range of each share.

    A modeled roof exists for the kernel sides only, so its fields leave
    ``torch.matmul`` out.
    """

    kernels = [side for side in sides if side != TORCH_MATMUL]

    def values(side: str, *path: str) -> list[object]:
        return [_pick(point["sides"][side], *path) for point in points]

    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            side: top_bottleneck(values(side, "decision", "bottleneck"))
            for side in sides
        },
        "fma_ceiling_share_range": {
            side: _span(values(side, "fma_ceiling_share")) for side in sides
        },
        "ideal_reuse_share_range": {
            side: _span(values(side, "ideal_reuse_share")) for side in sides
        },
        "modeled_roof_share_range": {
            side: _span(values(side, "traffic", "roof", "attainable_share"))
            for side in kernels
        },
        "points_above_modeled_roof": {
            side: sum(
                share > 1
                for share in values(side, "traffic", "roof", "attainable_share")
            )
            for side in kernels
        },
        "modeled_bandwidth_bound_points": {
            side: values(side, "traffic", "roof", "bound").count("bandwidth")
            for side in kernels
        },
        "step_verdicts": {
            row["variant"]: [
                _pick(entry["measured"], "verdict") for entry in row["points"]
            ]
            for row in steps
        },
    }


def _bound(
    checklist_report: Mapping[str, object],
    checklist_summary: Mapping[str, object],
    ceiling_report: Mapping[str, object],
    ceiling_summary: Mapping[str, object],
    timing_report: Mapping[str, object],
    timing_summary: Mapping[str, object],
    config: BottleneckConfig,
) -> dict[str, object]:
    """Assemble the unsealed bound report from three verified sources."""

    if type(config) is not BottleneckConfig:
        raise TypeError("config must be a BottleneckConfig")
    if config.rule.intensity_model != RULE_MODEL:
        raise MatmulChecklistBoundError(
            "the rule config must name the ideal_reuse roof, the only roof that "
            "bounds every side"
        )
    if checklist_summary["checklist_sha256"] != checklist_sha256():
        raise MatmulChecklistBoundError(
            "the checklist report seals another plan than the declared checklist"
        )
    _pair(checklist_report, ceiling_report, timing_report, timing_summary)
    roofline = _from_summary(dict(ceiling_summary))
    scales = ceiling_summary["fma_scales_with_work"]
    sides = list(timing_summary["sides"])
    fastest = timing_report["summary"]["fastest_host"]
    checklist_points = checklist_report["benchmark"]["points"]
    points = [
        _point(source, timing, sides, roofline, scales, fastest, config)
        for source, timing in zip(checklist_points, timing_report["points"])
    ]
    minimum = checklist_report["config"]["comparison"]["min_faster_ratio"]
    steps = _steps(points, checklist_points, minimum)
    measured = all(
        summary["measured"] is True
        for summary in (checklist_summary, ceiling_summary, timing_summary)
    )
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
        "sources": {
            CHECKLIST_SOURCE: _identity(checklist_summary),
            CEILING_SOURCE: _identity(ceiling_summary),
            TIMING_SOURCE: _identity(timing_summary),
        },
        "checklist_sha256": checklist_sha256(),
        "rule": {"config": config.to_mapping(), "config_sha256": config.fingerprint()},
        "comparison": {"min_faster_ratio": minimum, "ratio_rule": RATIO_RULE},
        "environment": {
            field: checklist_report["environment"][field]
            for field in BOUND_ENVIRONMENT_FIELDS
        },
        "torch_mps_switches": dict(checklist_report["torch_mps_switches"]["values"]),
        "ceilings": {
            "fma_gflops_per_second": roofline.compute_gflops_per_second,
            "bandwidth_gb_per_second": roofline.bandwidth_gb_per_second,
            "ridge_flops_per_byte": roofline.ridge_flops_per_byte,
            "fma_scales_with_work": scales,
        },
        "sides": sides,
        "fastest_calls": {side: dict(fastest[side]) for side in sides},
        "traffic_model": MODEL_TEXT,
        "definitions": dict(DEFINITIONS),
        "reasons": reasons_mapping(),
        "points": points,
        "steps": steps,
        "summary": _summary(label, sides, points, steps),
    }


def build_bound_report(
    checklist_report: object,
    ceiling_report: object,
    timing_report: object,
    config: BottleneckConfig,
) -> dict[str, object]:
    """Verify the three source reports, then return the sealed bound report."""

    checklist_summary = _checked("checklist", verify_checklist_report, checklist_report)
    ceiling_summary = _checked("gpu-ceilings", verify_ceiling_report, ceiling_report)
    timing_summary = _checked("one-call timing", verify_timing_report, timing_report)
    return seal_report(
        _bound(
            checklist_report,
            checklist_summary,
            ceiling_report,
            ceiling_summary,
            timing_report,
            timing_summary,
            config,
        )
    )


def bound_from_files(
    checklist_path: Path,
    ceiling_path: Path,
    timing_path: Path,
    config: BottleneckConfig,
) -> dict[str, object]:
    """Load the sources through their verified, canonical loaders; seal the bound."""

    checklist_report, checklist_summary = _checked(
        "checklist", load_checklist_report, checklist_path
    )
    ceiling_report, ceiling_summary = _checked(
        "gpu-ceilings", load_ceiling_report, ceiling_path
    )
    timing_report, timing_summary = _checked(
        "one-call timing", load_timing_report, timing_path
    )
    return seal_report(
        _bound(
            checklist_report,
            checklist_summary,
            ceiling_report,
            ceiling_summary,
            timing_report,
            timing_summary,
            config,
        )
    )
