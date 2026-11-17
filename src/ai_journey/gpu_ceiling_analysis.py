"""Derive the measured ceilings, the work scaling and the bank finding.

Every input is a point mapping as the report stores it, and every number is
recomputed from the recorded integers, so a verifier can repeat each step:

- The FMA ceiling is the highest median GFLOP/s of the sweep. It means
  something only if the fixed cost of a call is small next to that call,
  so a least-squares line ``time = fixed + flops / rate`` is fitted to the
  points of each chain count in exact rational arithmetic. The ceiling
  "scales with work" when its chain count's fixed cost is at most 5% of the
  ceiling call's median time.
- The bandwidth ceiling is the better median GB/s of the copy and the
  read-reduce at the largest buffer, the size least helped by caches.
- The bank finding compares each layout's time per access with the
  unpadded stride-1 time. A ratio of 1.25 or more counts as a slowdown;
  this lab treats a smaller one as noise. The CUDA bank model of the
  Day 74 memory-access lab (32 banks of 4-byte words) gives each pattern a
  conflict degree to compare with; it is a model, not a fact about Apple
  GPUs, and the finding says only what the timings show.

The ridge point divides the compute ceiling by the bandwidth ceiling: the
FLOPs per byte at which this host's measured roofline turns flat. No vendor
peak enters any of these numbers.
"""

from __future__ import annotations

from fractions import Fraction

from .gpu_ceiling_reference import bank_lane_addresses
from .gpu_ceiling_source import (
    BANK_ROW_WORDS,
    STREAM_COPY,
    STREAM_READ_REDUCE,
    GpuCeilingError,
)
from .memory_access_banks import analyze_banks
from .memory_access_pattern import WarpAccess

FIXED_SHARE_LIMIT = 0.05
SLOWDOWN_LIMIT = 1.25
UNPADDED, PADDED = BANK_ROW_WORDS
BANK_MODEL = (
    "CUDA shared memory as the Day 74 memory-access lab models it: 32 banks of "
    "4-byte words, conflict degree = most distinct words one bank serves"
)
NO_CLEAR_PATTERN = "no_clear_conflict_pattern"
REMOVED_BY_PADDING = "stride_slowdown_removed_by_padding"
REDUCED_BY_PADDING = "stride_slowdown_reduced_by_padding"
NO_SINGLE_PATTERN = "no_single_pattern"
FINDINGS = (NO_CLEAR_PATTERN, REMOVED_BY_PADDING, REDUCED_BY_PADDING, NO_SINGLE_PATTERN)


def _call_ns(point: dict[str, object]) -> Fraction:
    """The median time of one call, exactly: median sample over its calls."""

    return Fraction(point["timing"]["median_ns"], point["calls_per_sample"])


def _best(points: list[dict[str, object]], key: str) -> dict[str, object]:
    """Return the first point with the highest value of ``key``."""

    best = points[0]
    for point in points[1:]:
        if point[key] > best[key]:
            best = point
    return best


def fit_fixed_cost(
    work: list[int], times: list[Fraction]
) -> tuple[Fraction | None, Fraction]:
    """Fit ``time = fixed + slope * work`` by least squares; return slope, fixed.

    Exact rationals make the fit the same on every host. A slope of zero or
    less has no rate and returns ``None``.
    """

    count = len(work)
    if count < 2 or len(times) != count or len(set(work)) < 2:
        raise GpuCeilingError("a fit needs two or more distinct work sizes")
    sum_x = sum(Fraction(value) for value in work)
    sum_y = sum(times)
    sum_xx = sum(Fraction(value) ** 2 for value in work)
    sum_xy = sum(Fraction(value) * time for value, time in zip(work, times))
    slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x**2)
    fixed = (sum_y - slope * sum_x) / count
    return (slope if slope > 0 else None), fixed


def _series(points: list[dict[str, object]], chains: int) -> list[dict[str, object]]:
    """The points of one chain count, in order of work."""

    series = [point for point in points if point["chains"] == chains]
    return sorted(series, key=lambda point: point["fma_steps"])


def _fit(series: list[dict[str, object]]) -> tuple[Fraction | None, Fraction]:
    return fit_fixed_cost(
        [point["flops_per_call"] for point in series],
        [_call_ns(point) for point in series],
    )


def fma_scaling(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fit the fixed cost of a call for each chain count of the sweep."""

    rows = []
    for chains in sorted({point["chains"] for point in points}):
        series = _series(points, chains)
        slope, fixed = _fit(series)
        steps = []
        for index, point in enumerate(series):
            previous = series[index - 1] if index else None
            steps.append(
                {
                    "fma_steps": point["fma_steps"],
                    "flops_per_call": point["flops_per_call"],
                    "median_ns_per_call": point["median_ns_per_call"],
                    "work_ratio_to_previous": None
                    if previous is None
                    else point["flops_per_call"] / previous["flops_per_call"],
                    "time_ratio_to_previous": None
                    if previous is None
                    else float(_call_ns(point) / _call_ns(previous)),
                }
            )
        rows.append(
            {
                "chains": chains,
                "points": steps,
                "marginal_gflops_per_second": None
                if slope is None
                else float(1 / slope),
                "fixed_ns_per_call": float(fixed),
                "fixed_share_at_largest": float(fixed / _call_ns(series[-1])),
            }
        )
    return rows


def fma_ceiling(points: list[dict[str, object]]) -> dict[str, object]:
    """The highest median GFLOP/s of the sweep, with its fixed-cost share."""

    best = _best(points, "median_gflops_per_second")
    slope, fixed = _fit(_series(points, best["chains"]))
    share = fixed / _call_ns(best)
    return {
        "gflops_per_second": best["median_gflops_per_second"],
        "best_sample_gflops_per_second": best["best_gflops_per_second"],
        "kernel": best["kernel"],
        "chains": best["chains"],
        "fma_steps": best["fma_steps"],
        "flops_per_call": best["flops_per_call"],
        "fixed_share_of_ceiling_call": float(share),
        "fixed_share_limit": FIXED_SHARE_LIMIT,
        "scales_with_work": slope is not None and share <= Fraction(FIXED_SHARE_LIMIT),
    }


def bandwidth_ceiling(points: list[dict[str, object]]) -> dict[str, object]:
    """The better median GB/s of the two stream kernels at the largest buffer."""

    largest = max(point["buffer_bytes"] for point in points)
    candidates = [point for point in points if point["buffer_bytes"] == largest]
    if sorted(point["kernel"] for point in candidates) != [
        STREAM_COPY,
        STREAM_READ_REDUCE,
    ]:
        raise GpuCeilingError("the largest buffer needs one copy and one read-reduce")
    best = _best(candidates, "median_gb_per_second")
    rates = {point["kernel"]: point["median_gb_per_second"] for point in candidates}
    return {
        "gb_per_second": best["median_gb_per_second"],
        "best_sample_gb_per_second": best["best_gb_per_second"],
        "kernel": best["kernel"],
        "buffer_bytes": largest,
        "copy_gb_per_second": rates[STREAM_COPY],
        "read_reduce_gb_per_second": rates[STREAM_READ_REDUCE],
    }


def model_conflict_degree(stride: int, row_words: int) -> int:
    """The CUDA 32-bank model's conflict degree for one load of the pattern."""

    access = WarpAccess(4, bank_lane_addresses(stride, row_words))
    return analyze_banks(access).conflict_degree


def bank_analysis(points: list[dict[str, object]]) -> dict[str, object]:
    """Compare each layout and stride with the unpadded stride-1 time."""

    layouts = {
        words: sorted(
            (point for point in points if point["row_words"] == words),
            key=lambda point: point["stride"],
        )
        for words in BANK_ROW_WORDS
    }
    strides = [point["stride"] for point in layouts[UNPADDED]]
    if strides != [point["stride"] for point in layouts[PADDED]] or strides[0] != 1:
        raise GpuCeilingError("both layouts need the same strides, starting at 1")
    unpadded = [_call_ns(point) for point in layouts[UNPADDED]]
    padded = [_call_ns(point) for point in layouts[PADDED]]
    base = unpadded[0]
    rows = []
    for index, stride in enumerate(strides):
        rows.append(
            {
                "stride": stride,
                "unpadded_ps_per_access": layouts[UNPADDED][index][
                    "median_ps_per_access"
                ],
                "padded_ps_per_access": layouts[PADDED][index]["median_ps_per_access"],
                "unpadded_relative_to_stride_1": float(unpadded[index] / base),
                "padded_relative_to_own_stride_1": float(padded[index] / padded[0]),
                "padded_relative_to_unpadded_stride_1": float(padded[index] / base),
                "padding_speedup": float(unpadded[index] / padded[index]),
                "model_unpadded_conflict_degree": model_conflict_degree(
                    stride, UNPADDED
                ),
                "model_padded_conflict_degree": model_conflict_degree(stride, PADDED),
            }
        )
    limit = Fraction(SLOWDOWN_LIMIT)
    unpadded_ratios = [time / base for time in unpadded]
    padded_ratios = [time / base for time in padded]
    rises = all(
        later > earlier for earlier, later in zip(unpadded_ratios, unpadded_ratios[1:])
    )
    slowdown = max(unpadded_ratios + padded_ratios) >= limit
    padded_within = max(padded_ratios) < limit
    speedup = unpadded[-1] / padded[-1]
    if not slowdown:
        finding = NO_CLEAR_PATTERN
    elif rises and unpadded_ratios[-1] >= limit and padded_within:
        finding = REMOVED_BY_PADDING
    elif rises and unpadded_ratios[-1] >= limit and speedup >= limit:
        finding = REDUCED_BY_PADDING
    else:
        finding = NO_SINGLE_PATTERN
    return {
        "model": BANK_MODEL,
        "slowdown_limit": SLOWDOWN_LIMIT,
        "rows": rows,
        "unpadded_rises_at_every_stride": rises,
        "largest_unpadded_ratio": float(max(unpadded_ratios)),
        "largest_padded_ratio": float(max(padded_ratios)),
        "padded_within_limit": padded_within,
        "padding_speedup_at_largest_stride": float(speedup),
        "finding": finding,
    }


def ridge_point(fma: dict[str, object], bandwidth: dict[str, object]) -> float:
    """FLOPs per byte where the measured compute and bandwidth ceilings meet."""

    return fma["gflops_per_second"] / bandwidth["gb_per_second"]
