"""Join the bandwidth and FMA ceilings of one verified Day 74 report to a run.

Attention moves bytes and does arithmetic, so this lab reads two ceilings
from one Day 74 gpu-ceilings report: the Day 79 ``BandwidthCeiling`` (the
better median GB/s of the two stream kernels at the largest buffer) and the
best median GFLOP/s of the Day 74 FMA chains. The report is read only
through the Day 74 loader, so its verifier must accept it first, and a
tampered report sets no ceiling.

A fraction of a ceiling is a measurement only when the attention run and
the ceiling report are both measurements from the same interpreter,
libraries and platform. Neither ceiling is a vendor peak or a hardware
limit, so a fraction above 1 is possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .gpu_ceiling_io import load_report
from .metal_flash_source import MetalFlashError
from .metal_fused_ceiling import SOURCE, BandwidthCeiling, ceiling_from_report
from .metal_fused_source import MetalFusedError

COMPUTE_DEFINITION = (
    "the FMA ceiling of the same verified Day 74 gpu-ceilings report: the best "
    "median GFLOP/s of its FMA chains, 2 operations for each fma, on the host "
    "that made it; not a vendor peak and not a hardware limit"
)


@dataclass(frozen=True)
class FlashCeiling:
    """The bandwidth and FMA ceilings of one verified report."""

    bandwidth: BandwidthCeiling
    fma_gflops_per_second: float

    def __post_init__(self) -> None:
        if not isinstance(self.bandwidth, BandwidthCeiling):
            raise TypeError("bandwidth must be a BandwidthCeiling")
        rate = self.fma_gflops_per_second
        if type(rate) is not float or not math.isfinite(rate) or rate <= 0:
            raise MetalFlashError("the FMA ceiling must be a positive float")

    @property
    def measured(self) -> bool:
        """Whether the ceiling report was a measurement."""

        return self.bandwidth.measured

    def same_environment(self, environment: dict[str, str]) -> bool:
        """Whether a run used the interpreter, libraries and platform of the report."""

        return self.bandwidth.same_environment(environment)

    def compute_mapping(self) -> dict[str, object]:
        """Return the FMA ceiling and the report it came from."""

        return {
            "source": SOURCE,
            "definition": COMPUTE_DEFINITION,
            "report_sha256": self.bandwidth.report_sha256,
            "fma_gflops_per_second": self.fma_gflops_per_second,
        }


def flash_ceiling(
    report: dict[str, object], summary: dict[str, object]
) -> FlashCeiling:
    """Build both ceilings from a report and the summary its verifier returned."""

    return FlashCeiling(
        bandwidth=ceiling_from_report(report, summary),
        fma_gflops_per_second=summary["fma_ceiling_gflops_per_second"],
    )


def load_flash_ceiling(path: Path) -> FlashCeiling:
    """Read one gpu-ceilings report through the verified Day 74 loader."""

    try:
        report, summary = load_report(path)
        return flash_ceiling(report, summary)
    except MetalFlashError:
        raise
    except (ValueError, MetalFusedError) as error:
        raise MetalFlashError(f"the ceiling report was refused: {error}") from error
