"""Exact, deterministic analysis for bounded serving observations."""

from __future__ import annotations


class ServingAnalysisError(ValueError):
    """Raised when serving metrics would be undefined or misleading."""


def nearest_rank_percentile(values: tuple[int, ...], percentile: int) -> int:
    """Return an integer nearest-rank percentile without interpolation."""

    if not isinstance(values, tuple) or not values:
        raise TypeError("values must be a nonempty integer tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ServingAnalysisError("percentile values must be nonnegative integers")
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, int)
        or not 1 <= percentile <= 100
    ):
        raise ServingAnalysisError("percentile must be an integer in [1, 100]")
    ordered = sorted(values)
    rank = (percentile * len(ordered) + 99) // 100
    return ordered[rank - 1]
