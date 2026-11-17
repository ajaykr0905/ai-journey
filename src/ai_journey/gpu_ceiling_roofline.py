"""How far below this host's measured ceilings a kernel runs.

Day 74 asks how far below peak a kernel runs. This module answers with the
ceilings of a verified ceiling report and never with a vendor number. At an
arithmetic intensity of ``I`` FLOPs per byte, the attainable rate is the
lower of the measured FMA ceiling and the measured bandwidth ceiling times
``I``; the fraction of the ceiling is the achieved rate over that.

A roofline is built only from a report that passes ``verify_report``, so a
tampered report cannot set the ceilings, and it keeps the report's digest
and label. A synthetic report gives a roofline whose ``measured`` is false,
and every comparison made with it says so. The ceilings describe one host;
nothing here predicts another host, CUDA or NVIDIA performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .gpu_ceiling_config import GpuCeilingConfig
from .gpu_ceiling_io import load_report
from .gpu_ceiling_report import MEASUREMENT_LABEL, SYNTHETIC_LABEL
from .gpu_ceiling_source import GpuCeilingError
from .gpu_ceiling_verify import verify_report

COMPUTE_BOUND = "compute"
BANDWIDTH_BOUND = "bandwidth"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _positive(value: object, name: str) -> float:
    """Return a positive finite float; an integer no float holds is refused too."""

    if type(value) not in (int, float):
        raise GpuCeilingError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise GpuCeilingError(f"{name} must be a positive finite number")
    return number


@dataclass(frozen=True)
class MeasuredRoofline:
    """The two ceilings of one verified report, with the report's identity."""

    compute_gflops_per_second: float
    bandwidth_gb_per_second: float
    report_sha256: str
    measured: bool
    label: str

    def __post_init__(self) -> None:
        _positive(self.compute_gflops_per_second, "compute_gflops_per_second")
        _positive(self.bandwidth_gb_per_second, "bandwidth_gb_per_second")
        if not isinstance(self.report_sha256, str) or not _SHA256.fullmatch(
            self.report_sha256
        ):
            raise GpuCeilingError("report_sha256 must be a SHA-256 hex digest")
        if type(self.measured) is not bool or self.label != (
            MEASUREMENT_LABEL if self.measured else SYNTHETIC_LABEL
        ):
            raise GpuCeilingError("the label must match whether the report measured")

    @property
    def ridge_flops_per_byte(self) -> float:
        """The intensity where the two ceilings meet."""

        return self.compute_gflops_per_second / self.bandwidth_gb_per_second

    def attainable_gflops_per_second(self, flops_per_byte: float) -> float:
        """The lower ceiling at one arithmetic intensity, in GFLOP/s."""

        intensity = _positive(flops_per_byte, "flops_per_byte")
        return min(
            self.compute_gflops_per_second, self.bandwidth_gb_per_second * intensity
        )

    def bound(self, flops_per_byte: float) -> str:
        """Name the ceiling that limits a kernel at this intensity."""

        intensity = _positive(flops_per_byte, "flops_per_byte")
        if self.bandwidth_gb_per_second * intensity < self.compute_gflops_per_second:
            return BANDWIDTH_BOUND
        return COMPUTE_BOUND

    def below_ceiling(
        self, achieved_gflops_per_second: float, flops_per_byte: float
    ) -> dict[str, object]:
        """Compare one achieved rate with the attainable rate of this host.

        The fraction can exceed 1 when a kernel beats these ceilings, which
        are the best that this lab's kernels reached, not hardware limits.
        """

        achieved = _positive(achieved_gflops_per_second, "achieved_gflops_per_second")
        attainable = self.attainable_gflops_per_second(flops_per_byte)
        return {
            "achieved_gflops_per_second": achieved,
            "flops_per_byte": float(flops_per_byte),
            "attainable_gflops_per_second": attainable,
            "fraction_of_ceiling": achieved / attainable,
            "bound": self.bound(flops_per_byte),
            "ceiling_report_sha256": self.report_sha256,
            "measured": self.measured,
            "label": self.label,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the ceilings and the identity of the report behind them."""

        return {
            "compute_gflops_per_second": self.compute_gflops_per_second,
            "bandwidth_gb_per_second": self.bandwidth_gb_per_second,
            "ridge_flops_per_byte": self.ridge_flops_per_byte,
            "report_sha256": self.report_sha256,
            "measured": self.measured,
            "label": self.label,
        }


def _from_summary(summary: dict[str, object]) -> MeasuredRoofline:
    return MeasuredRoofline(
        compute_gflops_per_second=summary["fma_ceiling_gflops_per_second"],
        bandwidth_gb_per_second=summary["bandwidth_ceiling_gb_per_second"],
        report_sha256=summary["report_sha256"],
        measured=summary["measured"],
        label=summary["label"],
    )


def roofline_from_report(
    report: object, *, config: GpuCeilingConfig | None = None
) -> MeasuredRoofline:
    """Verify a ceiling report, then take its two ceilings as a roofline."""

    return _from_summary(verify_report(report, config=config))


def load_roofline(
    path: Path, *, config: GpuCeilingConfig | None = None
) -> MeasuredRoofline:
    """Read a canonical ceiling report through the verified loader."""

    _, summary = load_report(path, config=config)
    return _from_summary(summary)
