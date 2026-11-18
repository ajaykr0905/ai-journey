"""Verified Day 74 sources of the bottleneck rule, joined with one timing input.

The rule reads three sealed Day 74 reports, and each one passes its own
verifier before the rule uses a number from it:

* the matmul-metal report, for the launch geometry and SIMD width;
* the gpu-ceilings report, for the measured FMA and bandwidth ceilings;
* the roofline gap report, for the steady-state rates, the intensities and
  the roofs. It must equal the gap that its two sources rebuild.

The timing input then joins the gap point by point. It must hold every naive
kernel at every gap shape and name the same interpreter, libraries and
platform as the reports. It may hold more entries, such as ``torch_matmul``
or other shapes; the rule ignores them, and the report counts them.

A source that was read but failed its checks raises
``MatmulBottleneckVerificationError``; bad input and sources that do not pair
raise ``MatmulBottleneckError``. Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .gpu_ceiling_source import GpuCeilingError
from .gpu_ceiling_verify import GpuCeilingVerificationError
from .gpu_ceiling_verify import verify_report as verify_ceiling_report
from .kernel_metal_report import ENVIRONMENT_FIELDS
from .matmul_bottleneck_config import (
    BottleneckRule,
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
)
from .matmul_bottleneck_rule import PointInputs
from .matmul_bottleneck_timing import TimingEntry, TimingInput
from .matmul_metal_io import load_report as load_matmul_report
from .matmul_metal_mapping import MatmulMetalError, matmul_shape
from .matmul_metal_verify import MatmulMetalVerificationError
from .matmul_metal_verify import verify_report as verify_matmul_report
from .matmul_roofline_gap import (
    MATMUL_SOURCE,
    NAIVE_SIDES,
    MatmulRooflineError,
    MatmulRooflineVerificationError,
    build_gap_report,
)
from .matmul_roofline_io import load_report as load_gap_report
from .matmul_roofline_verify import verify_gap_report
from .memory_access_report import DIGEST_FIELD
from .memory_access_tiling import MatmulShape

Checked = TypeVar("Checked")
_REJECTED = (
    MatmulMetalVerificationError,
    GpuCeilingVerificationError,
    MatmulRooflineVerificationError,
)
_INVALID = (MatmulMetalError, GpuCeilingError, MatmulRooflineError)


def _checked(name: str, verify: Callable[..., Checked], *args, **kwargs) -> Checked:
    """Run a Day 74 loader or verifier; name the source in any failure."""

    try:
        return verify(*args, **kwargs)
    except _REJECTED as error:
        raise MatmulBottleneckVerificationError(f"{name} report: {error}") from error
    except _INVALID as error:
        raise MatmulBottleneckError(f"{name} report: {error}") from error


@dataclass(frozen=True)
class Day74Sources:
    """The verified matmul-metal report and the gap bound to both its sources."""

    matmul_report: Mapping[str, object]
    gap_report: Mapping[str, object]
    gap_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            self.gap_report["sources"][MATMUL_SOURCE]["report_sha256"]
            != self.matmul_report[DIGEST_FIELD]
        ):
            raise MatmulBottleneckError(
                "the gap report was not made from this matmul-metal report"
            )
        if self.gap_summary.get("sources_checked") is not True:
            raise MatmulBottleneckError("the gap report was not bound to its sources")

    @property
    def group_size(self) -> tuple[int, int]:
        return tuple(self.matmul_report["config"]["launch"]["group_size"])

    @property
    def simd_widths(self) -> dict[str, int]:
        return {
            kernel["name"]: kernel["limits"]["thread_execution_width"]
            for kernel in self.matmul_report["shaders"]["kernels"]
        }


def sources_from_reports(
    matmul_report: object, ceiling_report: object, gap_report: object
) -> Day74Sources:
    """Verify three in-memory reports, each with its own verifier, and bind them."""

    _checked("matmul-metal", verify_matmul_report, matmul_report)
    _checked("gpu-ceilings", verify_ceiling_report, ceiling_report)
    rebuilt = _checked("roofline gap", build_gap_report, matmul_report, ceiling_report)
    summary = _checked("roofline gap", verify_gap_report, gap_report, rebuilt=rebuilt)
    return Day74Sources(matmul_report, gap_report, summary)


def sources_from_files(
    matmul_path: Path, ceiling_path: Path, gap_path: Path
) -> Day74Sources:
    """Load the three reports through their own verified, canonical loaders.

    The gap loader rebuilds the gap from both source files, which verifies
    the ceiling report too.
    """

    matmul_report, _ = _checked("matmul-metal", load_matmul_report, matmul_path)
    gap_report, summary = _checked(
        "roofline gap",
        load_gap_report,
        gap_path,
        matmul_report=matmul_path,
        ceiling_report=ceiling_path,
    )
    return Day74Sources(matmul_report, gap_report, summary)


@dataclass(frozen=True)
class JoinedPoint:
    """One naive kernel at one gap point, with its rule inputs and timing."""

    index: int
    shape: MatmulShape
    kernel: str
    inputs: PointInputs
    timing: TimingEntry


def _environment(sources: Day74Sources, timing: TimingInput) -> None:
    gap = sources.gap_report["environment"]
    timed = dict(timing.environment)
    differing = [field for field in ENVIRONMENT_FIELDS if gap[field] != timed[field]]
    if differing:
        raise MatmulBottleneckError(
            "the timing input names other software or platforms than the Day 74 "
            "reports: " + ", ".join(differing)
        )


def join_points(
    sources: Day74Sources, timing: TimingInput, rule: BottleneckRule
) -> tuple[JoinedPoint, ...]:
    """Pair each naive kernel at each gap point with its timing entry."""

    if type(sources) is not Day74Sources or type(timing) is not TimingInput:
        raise TypeError("join_points takes Day74Sources and a TimingInput")
    _environment(sources, timing)
    gap = sources.gap_report
    ceilings = gap["ceilings"]
    joined = []
    for point in gap["points"]:
        shape = matmul_shape(point["m"], point["n"], point["k"])
        roof = point["intensities"][rule.intensity_model][
            "bandwidth_roof_gflops_per_second"
        ]
        for kernel in NAIVE_SIDES:
            entry = timing.entry(shape, kernel)
            if entry is None:
                raise MatmulBottleneckError(
                    f"the timing input has no {kernel} entry at "
                    f"{shape.m} x {shape.n} x {shape.k}"
                )
            side = point["sides"][kernel]
            inputs = PointInputs(
                steady_ns_per_call=side["median_sample_ns"] / point["calls_per_sample"],
                rate_gflops_per_second=side["median_gflops_per_second"],
                gpu_ns=entry.gpu_ns,
                host_ns=entry.host_ns,
                dispatch_ns=entry.dispatch_ns,
                fma_ceiling_gflops_per_second=ceilings["fma_gflops_per_second"],
                fma_scales_with_work=ceilings["fma_scales_with_work"],
                bandwidth_roof_gflops_per_second=roof,
            )
            joined.append(JoinedPoint(point["index"], shape, kernel, inputs, entry))
    return tuple(joined)
