"""The three numbers of each sweep point and the step ratios between them.

Each point of one run gives three numbers, each a median GFLOP/s with the
range from the IQR of its sample times:

- naive: ``matmul_x_columns``, the faster Day 74 naive kernel.
- tiled: the admitted variant that ``TILED_RULE`` picks at that point.
- vendor: ``torch.matmul`` on the MPS device. PyTorch does not report which
  kernel it runs, so ``vendor_label`` names the call and the PyTorch
  version, never a kernel or a library.

No NVIDIA GPU and no cuBLAS exist on this host, so the cuBLAS field of every
point is ``CUBLAS`` and holds no number.

Each step ratio divides two median rates of the same point in the same run:
tiled over naive and vendor over tiled. ``RATIO_RULE`` calls a side faster
only when its ratio reaches the config's ``min_faster_ratio`` and the IQR
ranges of the two sides' sample times do not overlap; the inverse ratio
calls it slower by the same test, and anything else is
``no_clear_difference``. With no admitted variant, the tiled number and both
steps are ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping

from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_tiled_runtime import NAIVE_KERNEL
from .matmul_tiled_source import MatmulTiledError

FASTER = "faster"
SLOWER = "slower"
NO_CLEAR_DIFFERENCE = "no_clear_difference"
VERDICTS = (FASTER, SLOWER, NO_CLEAR_DIFFERENCE)
NAIVE_LABEL = (
    "naive GPU matmul: matmul_x_columns, the Day 74 kernel with x over the "
    "columns of C"
)
TILED_RULE = (
    "the admitted tiled variant with the highest median GFLOP/s at this point "
    "in this run; a tie goes to the variant listed first in the config"
)
CUBLAS = "not measured, no NVIDIA GPU or cuBLAS on this host"
RATIO_RULE = (
    "ratio = median GFLOP/s of the first side / median GFLOP/s of the second, "
    "from the same point of the same run; faster when ratio >= min_faster_ratio "
    "and the first side's p75 sample time is below the second side's p25; "
    "slower when 1 / ratio >= min_faster_ratio and the second side's p75 is "
    "below the first side's p25; otherwise no_clear_difference"
)
STEPS = (
    ("tiled_over_naive", "tiled", "naive"),
    ("vendor_over_tiled", "vendor", "tiled"),
)


def vendor_label(torch_version: str) -> str:
    """Name the vendor side by its call and PyTorch version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise MatmulTiledError("the vendor label needs the PyTorch version")
    return f"vendor GPU matmul: torch.matmul on MPS, PyTorch {torch_version}"


def _number(point: Mapping[str, object], side: str) -> dict[str, object]:
    entry = point["sides"][side]
    return {
        "side": side,
        "median_gflops_per_second": entry["median_gflops_per_second"],
        "iqr_gflops_per_second": list(entry["iqr_gflops_per_second"]),
    }


def step(
    first: Mapping[str, object],
    second: Mapping[str, object],
    min_faster_ratio: float,
) -> dict[str, object]:
    """Compare two sides of one point by ``RATIO_RULE``.

    ``first`` and ``second`` are side entries of one benchmark point, with
    the same calls per sample, so the rate ratio is the inverse ratio of
    median sample times.
    """

    one, two = first["timing"], second["timing"]
    ratio = first["median_gflops_per_second"] / second["median_gflops_per_second"]
    if ratio >= min_faster_ratio and one["p75_ns"] < two["p25_ns"]:
        verdict = FASTER
    elif 1 / ratio >= min_faster_ratio and two["p75_ns"] < one["p25_ns"]:
        verdict = SLOWER
    else:
        verdict = NO_CLEAR_DIFFERENCE
    return {
        "first": first["implementation"],
        "second": second["implementation"],
        "ratio": ratio,
        "iqr_separated": one["p75_ns"] < two["p25_ns"] or two["p75_ns"] < one["p25_ns"],
        "verdict": verdict,
    }


def best_tile(point: Mapping[str, object], variants: tuple[str, ...]) -> str | None:
    """Pick the tiled side by ``TILED_RULE``; ``None`` when none was timed."""

    best = None
    for name in variants:
        if name not in point["sides"]:
            continue
        rate = point["sides"][name]["median_gflops_per_second"]
        if best is None or rate > point["sides"][best]["median_gflops_per_second"]:
            best = name
    return best


def three_numbers(
    point: Mapping[str, object],
    variants: tuple[str, ...],
    tiles: Mapping[str, int],
    min_faster_ratio: float,
) -> dict[str, object]:
    """Return the naive, tiled and vendor numbers of one point and its steps."""

    sides = point["sides"]
    if NAIVE_KERNEL not in sides or TORCH_MATMUL not in sides:
        raise MatmulTiledError("a point needs the naive and the vendor side")
    chosen = best_tile(point, variants)
    numbers = {
        "naive": _number(point, NAIVE_KERNEL),
        "tiled": None
        if chosen is None
        else {**_number(point, chosen), "tile": tiles[chosen]},
        "vendor": _number(point, TORCH_MATMUL),
    }
    entries = {
        "naive": sides[NAIVE_KERNEL],
        "tiled": None if chosen is None else sides[chosen],
        "vendor": sides[TORCH_MATMUL],
    }
    steps = {
        name: None
        if entries[first] is None or entries[second] is None
        else step(entries[first], entries[second], min_faster_ratio)
        for name, first, second in STEPS
    }
    return {
        "index": point["index"],
        "m": point["m"],
        "n": point["n"],
        "k": point["k"],
        **numbers,
        "cublas": CUBLAS,
        **steps,
    }


def comparison(
    points: list[Mapping[str, object]],
    variants: tuple[str, ...],
    tiles: Mapping[str, int],
    min_faster_ratio: float,
    torch_version: str,
) -> dict[str, object]:
    """Return the labels, the rules and the three numbers of every point."""

    return {
        "naive": NAIVE_LABEL,
        "tiled_rule": TILED_RULE,
        "vendor": vendor_label(torch_version),
        "cublas": CUBLAS,
        "ratio_rule": RATIO_RULE,
        "min_faster_ratio": min_faster_ratio,
        "points": [
            three_numbers(point, variants, tiles, min_faster_ratio) for point in points
        ],
    }
