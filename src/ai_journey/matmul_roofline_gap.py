"""How far each naive Metal matmul runs below this host's measured ceilings.

Day 74 asks how far below peak a naive matmul runs. A matmul-metal report
holds the median GFLOP/s of each kernel at each sweep shape, and a
gpu-ceilings report holds the FMA and bandwidth ceilings that its own kernels
reached on the same host. This module joins them with the no-reuse and
ideal-reuse byte counts of the model lab, and never with a vendor number. For
each shape and side the gap report gives:

* the share of the measured FMA ceiling: the median rate over that ceiling;
* the share of the bandwidth roof at each intensity: the median rate over the
  bandwidth ceiling times the intensity. A share above 1 at no reuse shows
  that the kernel probably moved fewer bytes than the no-reuse model counts,
  or that the measured bandwidth ceiling is below the true limit of the host;
* the ratio to ``torch.matmul``, as the matmul-metal report states it.

Both sources must pass their own verifiers first, so a tampered source sets
no number. They must also name the same interpreter, libraries and platform,
because a ceiling from another software stack says nothing about these
kernels. The gap is a measurement only when both sources are, and it keeps
their labels. The two reports come from separate runs, so the load of the host
can differ between them; only the ceiling report records its load averages,
and the gap report keeps them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import TypeVar

from .gpu_ceiling_io import load_report as load_ceiling_report
from .gpu_ceiling_roofline import MeasuredRoofline
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
from .matmul_metal_io import load_report as load_matmul_report
from .matmul_metal_mapping import MatmulMetalError
from .matmul_metal_report import MATMUL_PRECISION_FIELD
from .matmul_metal_runtime import SIDES, TORCH_MATMUL
from .matmul_metal_source import KERNELS
from .matmul_metal_verify import MatmulMetalVerificationError
from .matmul_metal_verify import verify_report as verify_matmul_report
from .matmul_model_traffic import TRAFFIC_MODELS, account_matmul, ratio_mapping
from .memory_access_report import seal_report
from .memory_access_tiling import MatmulShape

Checked = TypeVar("Checked")
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_roofline_gap"
MATMUL_SOURCE = "matmul_metal"
CEILING_SOURCE = "gpu_ceilings"
NAIVE_SIDES = tuple(side for side in SIDES if side != TORCH_MATMUL)
GAP_ENVIRONMENT_FIELDS = (*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD)
CEILING_DEFINITION = (
    "The FMA ceiling is the highest median GFLOP/s of the ceiling lab's FMA "
    "points. The bandwidth ceiling is the higher median GB/s of its two stream "
    "kernels at the largest buffer. Both come from one run on this host, and "
    "neither is a vendor peak or a hardware limit."
)
DEFINITIONS = {
    "median_gflops_per_second": "flops_per_sample divided by median_sample_ns",
    "fma_ceiling_share": (
        "median GFLOP/s of a side divided by the measured FMA ceiling"
    ),
    "bandwidth_roof": (
        "the measured bandwidth ceiling in GB/s times the intensity in FLOPs "
        "per byte, in GFLOP/s"
    ),
    "roof_share": (
        "median GFLOP/s of a side divided by the bandwidth roof at one intensity"
    ),
    "bound": (
        "bandwidth where the bandwidth roof is below the FMA ceiling, else compute"
    ),
    "torch_ratio": (
        "median sample time of torch.matmul divided by that of the naive kernel; "
        "all sides of a point ran the same calls per sample"
    ),
    "no_reuse": (
        "the model lab's no_reuse traffic: 4 * (2 * M * N * K + M * N) bytes, "
        "K / (4 * K + 2) FLOPs per byte"
    ),
    "ideal_reuse": (
        "the model lab's ideal_reuse traffic: 4 * (M * K + K * N + M * N) bytes, "
        "each element moved once"
    ),
}
MEASUREMENT_BOUNDARY = {
    "apple_gpu_metal_this_host": True,
    "host_specific": True,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "These ratios compare two verified reports from the Apple GPU of one "
        "host. The matmul rates and the ceilings are host-specific Apple GPU "
        "(Metal) measurements from two separate runs. Each ceiling is a rate "
        "that the ceiling lab's own kernels reached, not a vendor peak and not "
        "a hardware limit. The intensities come from the byte counts of the "
        "model lab, not from measured traffic. No CUDA kernel ran, and no NVIDIA "
        "GPU was used. These ratios do not measure or predict CUDA, NVIDIA, "
        "distributed or production performance."
    ),
}
SYNTHETIC_BOUNDARY = {
    "apple_gpu_metal_this_host": False,
    "synthetic_timing": True,
    "vendor_peak_used": False,
    "traffic_measured": False,
    "cuda_kernel_executed": False,
    "nvidia_gpu_used": False,
    "distributed_claim": False,
    "production_claim": False,
    "portable_performance_claim": False,
    "statement": (
        "At least one source report is synthetic, so each rate, ceiling and "
        "ratio in this report is synthetic. They are not measurements of this "
        "host or of any other host. No CUDA kernel ran, and no NVIDIA GPU was "
        "used."
    ),
}


class MatmulRooflineError(ValueError):
    """Raised when the inputs of a gap report are unreadable or do not pair."""


class MatmulRooflineVerificationError(MatmulRooflineError):
    """Raised when a gap report, or one of its sources, fails verification."""


def flops_per_byte(intensity: Fraction) -> float:
    """Return an exact intensity as the nearest float.

    Integer true division rounds once, so the verifier gets the same bits.
    """

    return intensity.numerator / intensity.denominator


def _sources(
    matmul: Mapping[str, object], ceilings: Mapping[str, object]
) -> dict[str, object]:
    def identity(summary: Mapping[str, object]) -> dict[str, object]:
        return {
            field: summary[field]
            for field in ("kind", "report_sha256", "config_sha256", "measured", "label")
        }

    return {
        MATMUL_SOURCE: identity(matmul),
        CEILING_SOURCE: {
            **identity(ceilings),
            "load_average_before": list(ceilings["load_average_before"]),
            "load_average_after": list(ceilings["load_average_after"]),
        },
    }


def _environment(
    matmul_report: Mapping[str, object], ceiling_report: Mapping[str, object]
) -> dict[str, str]:
    """Return the shared environment; refuse reports from other software."""

    matmul, ceilings = matmul_report["environment"], ceiling_report["environment"]
    differing = [
        field for field in ENVIRONMENT_FIELDS if matmul[field] != ceilings[field]
    ]
    if differing:
        raise MatmulRooflineError(
            "the two reports name different software or platforms: "
            + ", ".join(differing)
        )
    return {field: matmul[field] for field in GAP_ENVIRONMENT_FIELDS}


def _roofline(summary: Mapping[str, object]) -> MeasuredRoofline:
    return MeasuredRoofline(
        compute_gflops_per_second=summary["fma_ceiling_gflops_per_second"],
        bandwidth_gb_per_second=summary["bandwidth_ceiling_gb_per_second"],
        report_sha256=summary["report_sha256"],
        measured=summary["measured"],
        label=summary["label"],
    )


def _point(
    point: Mapping[str, object], roofline: MeasuredRoofline
) -> dict[str, object]:
    shape = MatmulShape(point["m"], point["n"], point["k"])
    accounting = account_matmul(shape)
    sides = point["sides"]
    calls = {sides[side]["calls_per_sample"] for side in SIDES}
    if len(calls) != 1:
        raise MatmulRooflineError(
            f"point {point['index']}: the sides ran different calls per sample"
        )
    roofs = {}
    intensities = {}
    for model, traffic in zip(
        TRAFFIC_MODELS, (accounting.no_reuse, accounting.ideal_reuse)
    ):
        intensity = flops_per_byte(traffic.intensity)
        roofs[model] = roofline.bandwidth_gb_per_second * intensity
        intensities[model] = {
            "bytes": traffic.bytes,
            "flops_per_byte": ratio_mapping(traffic.intensity),
            "bandwidth_roof_gflops_per_second": roofs[model],
            "bound": roofline.bound(intensity),
        }
    entries = {}
    for side in SIDES:
        rate = sides[side]["median_gflops_per_second"]
        entry = {
            "median_sample_ns": sides[side]["timing"]["median_ns"],
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / roofline.compute_gflops_per_second,
            **{f"{model}_roof_share": rate / roofs[model] for model in TRAFFIC_MODELS},
        }
        if side != TORCH_MATMUL:
            entry["mapping"] = KERNELS[side].mapping
            entry["torch_ratio"] = point["naive_to_torch_gflops_ratio"][side]
        entries[side] = entry
    (calls_per_sample,) = calls
    return {
        "index": point["index"],
        **shape.to_mapping(),
        "flops_per_call": shape.flops,
        "calls_per_sample": calls_per_sample,
        "flops_per_sample": shape.flops * calls_per_sample,
        "intensities": intensities,
        "sides": entries,
    }


def _span(points: list[dict[str, object]], side: str, field: str) -> list[float]:
    values = [point["sides"][side][field] for point in points]
    return [min(values), max(values)]


def _summary(label: str, points: list[dict[str, object]]) -> dict[str, object]:
    """Name the best FMA share of each side and the range of each other ratio.

    ``max`` keeps the first point of a tie, which the verifier repeats.
    """

    def best(side: str) -> dict[str, object]:
        top = max(points, key=lambda point: point["sides"][side]["fma_ceiling_share"])
        return {
            "index": top["index"],
            "fma_ceiling_share": top["sides"][side]["fma_ceiling_share"],
        }

    return {
        "label": label,
        "points": len(points),
        "best_fma_ceiling_share": {side: best(side) for side in SIDES},
        **{
            f"{model}_roof_share_range": {
                side: _span(points, side, f"{model}_roof_share") for side in NAIVE_SIDES
            }
            for model in TRAFFIC_MODELS
        },
        "points_above_no_reuse_roof": {
            side: sum(
                point["sides"][side]["no_reuse_roof_share"] > 1 for point in points
            )
            for side in NAIVE_SIDES
        },
        "torch_ratio_range": {
            side: _span(points, side, "torch_ratio") for side in NAIVE_SIDES
        },
    }


def _gap(
    matmul_report: Mapping[str, object],
    matmul_summary: Mapping[str, object],
    ceiling_report: Mapping[str, object],
    ceiling_summary: Mapping[str, object],
) -> dict[str, object]:
    """Assemble the unsealed gap report from two verified sources."""

    environment = _environment(matmul_report, ceiling_report)
    roofline = _roofline(ceiling_summary)
    points = [_point(point, roofline) for point in matmul_report["benchmark"]["points"]]
    measured = (
        matmul_summary["measured"] is True and ceiling_summary["measured"] is True
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
        "sources": _sources(matmul_summary, ceiling_summary),
        "environment": environment,
        "ceilings": {
            "definition": CEILING_DEFINITION,
            "fma_gflops_per_second": roofline.compute_gflops_per_second,
            "bandwidth_gb_per_second": roofline.bandwidth_gb_per_second,
            "ridge_flops_per_byte": roofline.ridge_flops_per_byte,
            "fma_scales_with_work": ceiling_summary["fma_scales_with_work"],
        },
        "definitions": dict(DEFINITIONS),
        "points": points,
        "summary": _summary(label, points),
    }


def _checked(name: str, verify: Callable[..., Checked], *arguments: object) -> Checked:
    """Run a source's own loader or verifier; name the source in any failure.

    A source that was read but failed its checks is a rejected report, and
    anything else is bad input, so callers keep the two exit codes apart.
    """

    try:
        return verify(*arguments)
    except (MatmulMetalVerificationError, GpuCeilingVerificationError) as error:
        raise MatmulRooflineVerificationError(f"{name} report: {error}") from error
    except (MatmulMetalError, GpuCeilingError) as error:
        raise MatmulRooflineError(f"{name} report: {error}") from error


def build_gap_report(
    matmul_report: object, ceiling_report: object
) -> dict[str, object]:
    """Verify both source reports, then return the sealed gap report."""

    matmul_summary = _checked("matmul-metal", verify_matmul_report, matmul_report)
    ceiling_summary = _checked("gpu-ceilings", verify_ceiling_report, ceiling_report)
    return seal_report(
        _gap(matmul_report, matmul_summary, ceiling_report, ceiling_summary)
    )


def gap_from_files(matmul_path: Path, ceiling_path: Path) -> dict[str, object]:
    """Load both sources through their verified, canonical loaders; seal the gap."""

    matmul_report, matmul_summary = _checked(
        "matmul-metal", load_matmul_report, matmul_path
    )
    ceiling_report, ceiling_summary = _checked(
        "gpu-ceilings", load_ceiling_report, ceiling_path
    )
    return seal_report(
        _gap(matmul_report, matmul_summary, ceiling_report, ceiling_summary)
    )
