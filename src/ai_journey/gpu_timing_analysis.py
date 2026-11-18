"""Derive the overhead estimate, the fits, the splits and the inferences.

Every number here comes from the medians of the raw samples, and each one
stays inside one clock domain unless its record says otherwise:

- The overhead estimate quotes the empty kernel's median on each clock and
  each side's median at the smallest shape. Both are direct measurements of
  a launch that does almost no work.
- One weighted line per side and clock splits each median into the fitted
  intercept and the size-dependent rest. The ratio of each intercept to the
  empty kernel's median on the same clock shows how well the line's fixed
  part agrees with the direct floor.
- ``dispatch_to_synchronized_ratio`` divides two host medians, one domain.
- ``host_minus_event`` is the one cross-domain subtraction. Its record names
  both clocks and carries ``cross_domain: true``.
- ``event_gflops_per_second`` divides the work by the event median, and its
  fraction of the Day 74 FMA ceiling says how far the side is from the
  compute ceiling. That is an inference from timings and an earlier
  measurement on this host, not a hardware counter reading.
"""

from __future__ import annotations

from .gpu_timing_clocks import (
    CLOCKS,
    CROSS_DOMAIN_MEANING,
    DISPATCH,
    EVENT,
    SYNCHRONIZED,
    GpuTimingError,
    cross_domain_difference,
    require_same_domain,
)
from .gpu_timing_fit import (
    FIT_METHOD,
    REGIME_RULE,
    SPLIT_RULE,
    LineFit,
    fit_line,
    split,
)
from .gpu_timing_sweep import EmptyPoint, SweepPoint
from .matmul_metal_runtime import SIDES

INFERENCE_BASIS = (
    "an inference from measured timings on this host and the Day 74 measured "
    "FMA ceiling; no hardware counter, profiler or vendor figure is used"
)
FRACTION_RULE = (
    "event_gflops_per_second = flops_per_call / event median_ns; "
    "event_fraction_of_fma_ceiling = that / the Day 74 FMA ceiling"
)


def _dispatch_ratio(timing: object) -> float:
    require_same_domain(DISPATCH, SYNCHRONIZED)
    return timing.median_ns(DISPATCH) / timing.median_ns(SYNCHRONIZED)


def _host_minus_event(timing: object) -> dict[str, object]:
    return cross_domain_difference(
        timing.median_ns(SYNCHRONIZED), timing.median_ns(EVENT)
    )


def fit_sides(points: tuple[SweepPoint, ...]) -> dict[str, dict[str, LineFit]]:
    """Fit one line per side and clock over the whole sweep."""

    flops = tuple(point.shape.flops for point in points)
    return {
        side: {
            clock: fit_line(
                flops, tuple(point.timing(side).median_ns(clock) for point in points)
            )
            for clock in CLOCKS
        }
        for side in SIDES
    }


def _point(
    point: SweepPoint, fits: dict[str, dict[str, LineFit]], ceiling: float
) -> dict[str, object]:
    sides = {}
    for side in SIDES:
        timing = point.timing(side)
        clocks = {}
        for clock in CLOCKS:
            summary = timing.samples(clock).summary
            clocks[clock] = {
                **split(summary.median_ns, fits[side][clock]).to_mapping(),
                "iqr_ns": summary.iqr_ns,
            }
        rate = point.shape.flops / timing.median_ns(EVENT)
        sides[side] = {
            "clocks": clocks,
            "dispatch_to_synchronized_ratio": _dispatch_ratio(timing),
            "host_minus_event": _host_minus_event(timing),
            "event_gflops_per_second": rate,
            "event_fraction_of_fma_ceiling": rate / ceiling,
        }
    return {
        "index": point.index,
        **point.shape.to_mapping(),
        "flops_per_call": point.shape.flops,
        "sides": sides,
    }


def analyze(
    empty: EmptyPoint, points: tuple[SweepPoint, ...], fma_ceiling: float
) -> dict[str, object]:
    """Return the overhead estimate, fits, per-point splits and inferences."""

    if not isinstance(empty, EmptyPoint):
        raise TypeError("empty must be an EmptyPoint")
    if type(points) is not tuple or not all(
        isinstance(point, SweepPoint) for point in points
    ):
        raise TypeError("points must be a tuple of SweepPoint")
    if type(fma_ceiling) is not float or not fma_ceiling > 0.0:
        raise GpuTimingError("fma_ceiling must be a positive float")
    floor = {clock: empty.timing.median_ns(clock) for clock in CLOCKS}
    fits = fit_sides(points)
    smallest = points[0]
    return {
        "fit_method": FIT_METHOD,
        "split_rule": SPLIT_RULE,
        "regime_rule": REGIME_RULE,
        "cross_domain_meaning": CROSS_DOMAIN_MEANING,
        "inference_basis": INFERENCE_BASIS,
        "fraction_rule": FRACTION_RULE,
        "fma_ceiling_gflops_per_second": fma_ceiling,
        "overhead_estimate": {
            "empty_kernel_median_ns": floor,
            "empty_kernel_dispatch_to_synchronized_ratio": _dispatch_ratio(
                empty.timing
            ),
            "empty_kernel_host_minus_event": _host_minus_event(empty.timing),
            "smallest_shape": {
                "index": smallest.index,
                **smallest.shape.to_mapping(),
                "median_ns": {
                    side: {
                        clock: smallest.timing(side).median_ns(clock)
                        for clock in CLOCKS
                    }
                    for side in SIDES
                },
            },
        },
        "fits": {
            side: {
                clock: {
                    **fit.to_mapping(),
                    "intercept_to_empty_kernel_ratio": fit.intercept_ns / floor[clock],
                }
                for clock, fit in by_clock.items()
            }
            for side, by_clock in fits.items()
        },
        "points": [_point(point, fits, fma_ceiling) for point in points],
    }
