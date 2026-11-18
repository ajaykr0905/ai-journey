"""Split a launch time into a fixed overhead and a size-dependent part.

The model is a straight line over the sweep: ``time = fixed + slope * flops``.
The intercept ``fixed`` estimates the cost that every launch pays whatever
its size, and ``slope * flops`` the part that grows with the work. The sweep
runs from a 1 x 1 x 1 matmul to millions of times more work, so an ordinary
least-squares line would follow the largest points and leave the intercept
to their noise. The fit therefore weights each point by ``1 / median**2``,
so it minimizes relative, not absolute, residuals, and each point counts
about the same.

The closed form uses ``math.fsum`` for every sum. Each sum is then correctly
rounded and independent of the order of the points, so a verifier with its
own code gets the same bits. One line per side and clock is fitted, and a
fit never mixes clocks.

A straight line is a model, not a law. The naive kernels change how they use
caches as the matrices grow, so the report gives the largest relative
residual, and a reader can see where the line does not hold. A fitted
intercept can come out zero or negative when the line fails at small sizes;
it is reported as fitted, with ``fixed_share`` showing the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gpu_timing_clocks import GpuTimingError

FIT_METHOD = (
    "weighted least squares of median_ns = intercept_ns + slope_ns_per_flop * "
    "flops over the sweep points of one side and one clock, with weight "
    "1 / median_ns**2 (relative residuals); every sum is math.fsum"
)
SPLIT_RULE = (
    "fixed_ns = the fitted intercept_ns; size_ns = median_ns - fixed_ns; "
    "fixed_share = fixed_ns / median_ns"
)
FIXED_DOMINANT = "fixed_overhead_dominant"
SIZE_DOMINANT = "size_dependent_dominant"
REGIME_RULE = (
    f"{FIXED_DOMINANT} when fixed_share >= 0.5, otherwise {SIZE_DOMINANT}; an "
    "inference from the fitted line, not a hardware counter"
)
MIN_FIT_POINTS = 3


@dataclass(frozen=True)
class LineFit:
    """One fitted line with the residual that shows how well it holds."""

    intercept_ns: float
    slope_ns_per_flop: float
    points: int
    max_relative_residual: float

    def to_mapping(self) -> dict[str, object]:
        """Return the fit as a report records it."""

        return {
            "intercept_ns": self.intercept_ns,
            "slope_ns_per_flop": self.slope_ns_per_flop,
            "points": self.points,
            "max_relative_residual": self.max_relative_residual,
        }

    def predict_ns(self, flops: int) -> float:
        """Return the line's time at ``flops``."""

        return self.intercept_ns + self.slope_ns_per_flop * flops


def fit_line(flops: tuple[int, ...], medians_ns: tuple[int, ...]) -> LineFit:
    """Fit ``median = intercept + slope * flops`` by relative least squares.

    It needs at least three points with at least two distinct work counts,
    so the line is not forced through two points and the slope exists.
    """

    if (
        type(flops) is not tuple
        or type(medians_ns) is not tuple
        or len(flops) != len(medians_ns)
        or len(flops) < MIN_FIT_POINTS
    ):
        raise GpuTimingError(
            f"a fit needs two equal tuples of at least {MIN_FIT_POINTS} values"
        )
    for name, values in (("flops", flops), ("medians_ns", medians_ns)):
        if any(type(value) is not int or value <= 0 for value in values):
            raise GpuTimingError(f"{name} must hold positive integers")
    if len(set(flops)) < 2:
        raise GpuTimingError("a fit needs at least two distinct work counts")
    weights = [1.0 / (median * median) for median in medians_ns]
    total = math.fsum(weights)
    sum_x = math.fsum(w * x for w, x in zip(weights, flops))
    sum_y = math.fsum(w * y for w, y in zip(weights, medians_ns))
    sum_xx = math.fsum(w * x * x for w, x in zip(weights, flops))
    sum_xy = math.fsum(w * x * y for w, x, y in zip(weights, flops, medians_ns))
    determinant = total * sum_xx - sum_x * sum_x
    if not determinant > 0.0:
        raise GpuTimingError("the fit is singular at float64 precision")
    slope = (total * sum_xy - sum_x * sum_y) / determinant
    intercept = (sum_y - slope * sum_x) / total
    residual = max(
        abs(y - (intercept + slope * x)) / y for x, y in zip(flops, medians_ns)
    )
    return LineFit(intercept, slope, len(flops), residual)


@dataclass(frozen=True)
class OverheadSplit:
    """One median split by a fitted line into fixed and size-dependent parts."""

    median_ns: int
    fixed_ns: float

    def __post_init__(self) -> None:
        if type(self.median_ns) is not int or self.median_ns <= 0:
            raise GpuTimingError("median_ns must be a positive integer")
        if type(self.fixed_ns) is not float or not math.isfinite(self.fixed_ns):
            raise GpuTimingError("fixed_ns must be a finite float")

    @property
    def size_ns(self) -> float:
        """The part of the median that the line puts on the work."""

        return self.median_ns - self.fixed_ns

    @property
    def fixed_share(self) -> float:
        """The fraction of the median that the line calls fixed overhead."""

        return self.fixed_ns / self.median_ns

    @property
    def regime(self) -> str:
        """Name which part is larger, as an inference from the line."""

        return FIXED_DOMINANT if self.fixed_share >= 0.5 else SIZE_DOMINANT

    def to_mapping(self) -> dict[str, object]:
        """Return the split as a report records it."""

        return {
            "median_ns": self.median_ns,
            "fixed_ns": self.fixed_ns,
            "size_ns": self.size_ns,
            "fixed_share": self.fixed_share,
            "regime": self.regime,
        }


def split(median_ns: int, fit: LineFit) -> OverheadSplit:
    """Split one median with the intercept of its own side and clock."""

    if not isinstance(fit, LineFit):
        raise TypeError("fit must be a LineFit")
    return OverheadSplit(median_ns, fit.intercept_ns)
