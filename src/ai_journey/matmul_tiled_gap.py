"""How far each tiled-lab side runs below this host's measured ceilings, and why.

Day 76 asks why the tiles were not faster than the naive kernel. This
module joins three verified reports of one host and never uses a vendor
number:

* the tiled report: the median sample time and rate of each side at each
  sweep shape;
* the gpu-ceilings report: the FMA and bandwidth ceilings that the ceiling
  lab's own kernels reached;
* the one-call timing report: the event, host and dispatch medians of one
  call of each side, and the fastest host call of each side.

For each side and shape the gap report gives the share of the measured FMA
ceiling and the modeled device traffic of the side's kernel from
``memory_access_tiling``: no reuse for the naive kernel, and
``TiledMatmulTraffic`` for each tile. PyTorch does not name the kernel that
``torch.matmul`` runs, so no traffic model applies to that side. It then
gives the attainable rate of ``MeasuredRoofline`` at the modeled intensity
and at the ideal-reuse intensity, and the Day 75 bottleneck name from
``matmul_bottleneck_rule.classify`` with the ideal-reuse roof. No kernel
moves fewer bytes than ideal reuse, so that roof bounds every side.

A share of a modeled roof above 1 shows that the kernel moved fewer device
bytes than the model counts, probably because a cache served repeated
loads, or that the bandwidth ceiling is below the true limit. A bottleneck
name is an inference from timings and measured ceilings: no hardware
counter took part, so no stall, cache hit or occupancy was observed.

Every source passes its own verifier first, so a tampered source sets no
number. All three must name the same interpreter, libraries and platform.
The tiled and timing reports must also come from one config, with the same
float32 matmul precision, MPS switches and admitted tiles. The gap is a
measurement only when all three sources are, and it keeps the load
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
from .matmul_metal_report import MATMUL_PRECISION_FIELD
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_model_traffic import ideal_reuse_traffic, ratio_mapping
from .matmul_roofline_gap import flops_per_byte
from .matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
)
from .matmul_tiled_gap_timing_io import load_timing_report
from .matmul_tiled_gap_timing_verify import verify_timing_report
from .matmul_tiled_io import load_report as load_tiled_report
from .matmul_tiled_runtime import NAIVE_KERNEL
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import VARIANTS
from .matmul_tiled_verify import MatmulTiledVerificationError
from .matmul_tiled_verify import verify_report as verify_tiled_report
from .memory_access_report import seal_report
from .memory_access_tiling import MatmulShape, NaiveMatmulTraffic, TiledMatmulTraffic

Checked = TypeVar("Checked")
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "matmul_tiled_gap"
TILED_SOURCE = "matmul_tiled"
CEILING_SOURCE = "gpu_ceilings"
TIMING_SOURCE = "one_call_timing"
SOURCE_NAMES = (TILED_SOURCE, CEILING_SOURCE, TIMING_SOURCE)
RULE_MODEL = "ideal_reuse"
NO_REUSE = "no_reuse"
TILED_MODEL = "tiled"
GAP_ENVIRONMENT_FIELDS = (*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD)
ONE_CALL_CLOCKS = (
    ("median_gpu_ns", "mps_event"),
    ("median_host_ns", "host_synchronized"),
    ("median_dispatch_ns", "host_dispatch"),
)
DEFINITIONS = {
    "steady_ns_per_call": (
        "the median sample time of the tiled report divided by its calls per "
        "sample; with one call per sample it also holds that call's "
        "synchronization"
    ),
    "median_gflops_per_second": (
        "flops_per_call times calls_per_sample divided by median_sample_ns, as "
        "the tiled report states it"
    ),
    "fma_ceiling_share": "median GFLOP/s of a side divided by the FMA ceiling",
    "traffic": (
        "modeled device traffic of the side's kernel, with no cache: no_reuse "
        "for the naive kernel, 4 * (2 * M * N * K + M * N) bytes; tiled for a "
        "T x T tile, 4 * (ceil(N / T) * M * K + ceil(M / T) * K * N + M * N) "
        "bytes; null for torch.matmul, whose kernel PyTorch does not name"
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
        "host: the tiled rates, the ceilings that the ceiling lab's own kernels "
        "reached, and one-call timings. Each is a host-specific Apple GPU "
        "(Metal) measurement from a separate run. A ceiling is a measured rate, "
        "not a vendor peak or a hardware limit, and the traffic comes from "
        "models, not from measured bytes. Each bottleneck name is an inference "
        "from timings and measured ceilings. No hardware counters, Nsight or "
        "Instruments were used, so no stall, cache hit or occupancy was "
        "observed. No CUDA kernel ran, no NVIDIA GPU was used, and cuBLAS was "
        "not measured. These numbers do not measure or predict CUDA, NVIDIA, "
        "cuBLAS, distributed or production performance."
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
    MatmulTiledVerificationError,
    GpuCeilingVerificationError,
    MatmulTiledGapVerificationError,
)
_INVALID = (MatmulTiledError, GpuCeilingError)


def _checked(name: str, verify: Callable[..., Checked], *arguments: object) -> Checked:
    """Run a source's own loader or verifier; name the source in any failure.

    A source that was read but failed its checks is a rejected report, and
    anything else is bad input, so callers keep the two exit codes apart.
    """

    try:
        return verify(*arguments)
    except _REJECTED as error:
        raise MatmulTiledGapVerificationError(f"{name} report: {error}") from error
    except _INVALID as error:
        raise MatmulTiledGapError(f"{name} report: {error}") from error


def _identity(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        field: summary[field]
        for field in (
            "kind",
            "report_sha256",
            "config_sha256",
            "measured",
            "label",
            "load_average_before",
            "load_average_after",
        )
    }


def _refused(admissions: list[Mapping[str, object]]) -> list[str]:
    return [entry["name"] for entry in admissions if not entry["admitted"]]


def _pair(
    tiled: Mapping[str, object],
    ceilings: Mapping[str, object],
    timing: Mapping[str, object],
) -> None:
    """Refuse sources from other software, another config or other kernels."""

    environments = (
        tiled["environment"],
        ceilings["environment"],
        timing["environment"],
    )
    differing = [
        field
        for field in ENVIRONMENT_FIELDS
        if len({environment[field] for environment in environments}) != 1
    ]
    if differing:
        raise MatmulTiledGapError(
            "the three reports name different software or platforms: "
            + ", ".join(differing)
        )
    precision = MATMUL_PRECISION_FIELD
    if tiled["environment"][precision] != timing["environment"][precision]:
        raise MatmulTiledGapError(
            "the tiled and timing reports ran at other float32 matmul precisions"
        )
    if tiled["config_sha256"] != timing["config_sha256"]:
        raise MatmulTiledGapError(
            "the timing report was not made from the tiled report's config"
        )
    if tiled["torch_mps_switches"]["values"] != timing["torch_mps_switches"]["values"]:
        raise MatmulTiledGapError("the tiled and timing reports set other MPS switches")
    tiled_refused = [entry["name"] for entry in tiled["refusals"]]
    if tiled_refused != _refused(timing["kernels"]["admissions"]):
        raise MatmulTiledGapError(
            "the tiled and timing runs admitted other tiles: "
            f"{tiled_refused} against {_refused(timing['kernels']['admissions'])}"
        )


def _ideal(shape: MatmulShape, roofline: MeasuredRoofline) -> dict[str, object]:
    traffic = ideal_reuse_traffic(shape)
    intensity = flops_per_byte(traffic.intensity)
    return {
        "bytes": traffic.bytes,
        "flops_per_byte": ratio_mapping(traffic.intensity),
        "bandwidth_roof_gflops_per_second": roofline.bandwidth_gb_per_second
        * intensity,
        "attainable_gflops_per_second": roofline.attainable_gflops_per_second(
            intensity
        ),
        "bound": roofline.bound(intensity),
    }


def _traffic(
    shape: MatmulShape, side: str, rate: float, roofline: MeasuredRoofline
) -> dict[str, object] | None:
    """Model the device traffic of one kernel side; ``torch.matmul`` has none."""

    if side == TORCH_MATMUL:
        return None
    tile = None if side == NAIVE_KERNEL else VARIANTS[side].tile
    model = (
        NaiveMatmulTraffic(shape) if tile is None else TiledMatmulTraffic(shape, tile)
    )
    intensity = model.arithmetic_intensity
    below = roofline.below_ceiling(rate, intensity)
    return {
        "model": NO_REUSE if tile is None else TILED_MODEL,
        "tile": tile,
        "load_bytes": model.load_bytes,
        "store_bytes": model.store_bytes,
        "bytes": model.total_bytes,
        "flops_per_byte": ratio_mapping(Fraction(model.flops, model.total_bytes)),
        "bandwidth_roof_gflops_per_second": roofline.bandwidth_gb_per_second
        * intensity,
        "attainable_gflops_per_second": below["attainable_gflops_per_second"],
        "attainable_share": below["fraction_of_ceiling"],
        "bound": below["bound"],
    }


def _one_call(point: Mapping[str, object], side: str) -> dict[str, int]:
    clocks = point["sides"][side]["clocks"]
    return {
        name: clocks[clock]["timing"]["median_ns"] for name, clock in ONE_CALL_CLOCKS
    }


def _point(
    tiled_point: Mapping[str, object],
    timing_point: Mapping[str, object],
    sides: list[str],
    roofline: MeasuredRoofline,
    scales: bool,
    fastest: Mapping[str, Mapping[str, int]],
    config: BottleneckConfig,
) -> dict[str, object]:
    shape = MatmulShape(tiled_point["m"], tiled_point["n"], tiled_point["k"])
    if [timing_point[axis] for axis in "mnk"] != [shape.m, shape.n, shape.k]:
        raise MatmulTiledGapError(
            f"point {tiled_point['index']}: the timing report timed another shape"
        )
    if tiled_point["side_order"] != sides:
        raise MatmulTiledGapError(
            f"point {tiled_point['index']}: the tiled and timing runs timed other "
            "sides"
        )
    recorded = tiled_point["sides"]
    calls = {recorded[side]["calls_per_sample"] for side in sides}
    if len(calls) != 1:
        raise MatmulTiledGapError(
            f"point {tiled_point['index']}: the sides ran different calls per sample"
        )
    (calls_per_sample,) = calls
    ideal = _ideal(shape, roofline)
    entries = {}
    for side in sides:
        median = recorded[side]["timing"]["median_ns"]
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
            "steady_ns_per_call": inputs.steady_ns_per_call,
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / roofline.compute_gflops_per_second,
            "ideal_reuse_share": rate / ideal["attainable_gflops_per_second"],
            "traffic": _traffic(shape, side, rate, roofline),
            "one_call": one_call,
            "decision": classify(inputs, config.rule).to_mapping(),
        }
    return {
        "index": tiled_point["index"],
        **shape.to_mapping(),
        "flops_per_call": shape.flops,
        "calls_per_sample": calls_per_sample,
        "ideal_reuse": ideal,
        "sides": entries,
    }


def _span(points: list[dict[str, object]], side: str, field: str) -> list[float]:
    values = [point["sides"][side][field] for point in points]
    return [min(values), max(values)]


def _summary(
    label: str, sides: list[str], points: list[dict[str, object]]
) -> dict[str, object]:
    """Name the top bottleneck of each side and the range of each share.

    A modeled roof exists for the kernel sides only, so its fields leave
    ``torch.matmul`` out.
    """

    kernels = [side for side in sides if side != TORCH_MATMUL]

    def modeled(point: dict[str, object], side: str) -> float:
        return point["sides"][side]["traffic"]["attainable_share"]

    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            side: top_bottleneck(
                [point["sides"][side]["decision"]["bottleneck"] for point in points]
            )
            for side in sides
        },
        "fma_ceiling_share_range": {
            side: _span(points, side, "fma_ceiling_share") for side in sides
        },
        "ideal_reuse_share_range": {
            side: _span(points, side, "ideal_reuse_share") for side in sides
        },
        "modeled_roof_share_range": {
            side: [
                min(modeled(point, side) for point in points),
                max(modeled(point, side) for point in points),
            ]
            for side in kernels
        },
        "points_above_modeled_roof": {
            side: sum(modeled(point, side) > 1 for point in points) for side in kernels
        },
    }


def _gap(
    tiled_report: Mapping[str, object],
    tiled_summary: Mapping[str, object],
    ceiling_report: Mapping[str, object],
    ceiling_summary: Mapping[str, object],
    timing_report: Mapping[str, object],
    timing_summary: Mapping[str, object],
    config: BottleneckConfig,
) -> dict[str, object]:
    """Assemble the unsealed gap report from three verified sources."""

    if type(config) is not BottleneckConfig:
        raise TypeError("config must be a BottleneckConfig")
    if config.rule.intensity_model != RULE_MODEL:
        raise MatmulTiledGapError(
            "the rule config must name the ideal_reuse roof, the only roof that "
            "bounds every side"
        )
    _pair(tiled_report, ceiling_report, timing_report)
    roofline = _from_summary(dict(ceiling_summary))
    scales = ceiling_summary["fma_scales_with_work"]
    sides = list(timing_summary["sides"])
    fastest = timing_report["summary"]["fastest_host"]
    points = [
        _point(tiled_point, timing_point, sides, roofline, scales, fastest, config)
        for tiled_point, timing_point in zip(
            tiled_report["benchmark"]["points"], timing_report["points"]
        )
    ]
    measured = all(
        summary["measured"] is True
        for summary in (tiled_summary, ceiling_summary, timing_summary)
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
            TILED_SOURCE: _identity(tiled_summary),
            CEILING_SOURCE: _identity(ceiling_summary),
            TIMING_SOURCE: _identity(timing_summary),
        },
        "rule": {"config": config.to_mapping(), "config_sha256": config.fingerprint()},
        "environment": {
            field: tiled_report["environment"][field]
            for field in GAP_ENVIRONMENT_FIELDS
        },
        "torch_mps_switches": dict(tiled_report["torch_mps_switches"]["values"]),
        "ceilings": {
            "fma_gflops_per_second": roofline.compute_gflops_per_second,
            "bandwidth_gb_per_second": roofline.bandwidth_gb_per_second,
            "ridge_flops_per_byte": roofline.ridge_flops_per_byte,
            "fma_scales_with_work": scales,
        },
        "sides": sides,
        "fastest_calls": {side: dict(fastest[side]) for side in sides},
        "definitions": dict(DEFINITIONS),
        "reasons": reasons_mapping(),
        "points": points,
        "summary": _summary(label, sides, points),
    }


def build_gap_report(
    tiled_report: object,
    ceiling_report: object,
    timing_report: object,
    config: BottleneckConfig,
) -> dict[str, object]:
    """Verify the three source reports, then return the sealed gap report."""

    tiled_summary = _checked("tiled", verify_tiled_report, tiled_report)
    ceiling_summary = _checked("gpu-ceilings", verify_ceiling_report, ceiling_report)
    timing_summary = _checked("one-call timing", verify_timing_report, timing_report)
    return seal_report(
        _gap(
            tiled_report,
            tiled_summary,
            ceiling_report,
            ceiling_summary,
            timing_report,
            timing_summary,
            config,
        )
    )


def gap_from_files(
    tiled_path: Path,
    ceiling_path: Path,
    timing_path: Path,
    config: BottleneckConfig,
) -> dict[str, object]:
    """Load the sources through their verified, canonical loaders; seal the gap."""

    tiled_report, tiled_summary = _checked("tiled", load_tiled_report, tiled_path)
    ceiling_report, ceiling_summary = _checked(
        "gpu-ceilings", load_ceiling_report, ceiling_path
    )
    timing_report, timing_summary = _checked(
        "one-call timing", load_timing_report, timing_path
    )
    return seal_report(
        _gap(
            tiled_report,
            tiled_summary,
            ceiling_report,
            ceiling_summary,
            timing_report,
            timing_summary,
            config,
        )
    )
