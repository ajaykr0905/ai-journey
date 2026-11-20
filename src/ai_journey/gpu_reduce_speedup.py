"""The speedup of each reduction step, from the samples of one run only.

For each timing size and each step, the section gives:

- the median GB/s of the step's kernel and the range from the IQR of its
  samples
- the fraction of the measured bandwidth ceiling and of ``torch.sum``
- the step ratio to the step's base: the step before it, or ``torch.sum``
  for R0, so the ratio is the speedup of the step's own change
- the step ratio to R0, the speedup of every change so far
- whether the verdict over the base is one that the plan predicted for the
  size class of n

Each ratio divides two median rates of one size of one run, with the Day 76
rule: ``faster`` when the ratio reaches ``min_faster_ratio`` and the IQR
ranges of the two sides' sample times do not overlap, ``slower`` by the same
test in the other direction, and ``no_clear_difference`` otherwise. Every
side of a size reads the same bytes in the same calls per sample, so the
rate ratio is the inverse ratio of the median times. The vendor side names
the call and the PyTorch version, never a kernel, because PyTorch does not
report which kernel ``torch.sum`` runs.
"""

from __future__ import annotations

from collections.abc import Mapping

from .gpu_reduce_plan import PLAN, TORCH_SUM, size_class
from .gpu_reduce_source import R0, STEP_KERNELS, GpuReduceError
from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER

RATIO_RULE = (
    "ratio = median GB/s of the first side / median GB/s of the second, from "
    "the same size of the same run; faster when ratio >= min_faster_ratio and "
    "the first side's p75 sample time is below the second side's p25; slower "
    "when 1 / ratio >= min_faster_ratio and the second side's p75 is below the "
    "first side's p25; otherwise no_clear_difference"
)
HYPOTHESIS_RULE = (
    "a prediction holds at a size when the verdict of the step's kernel over "
    "its base is one of the verdicts that the plan states for the size class "
    "of n"
)
FRACTION_RULE = (
    "median GB/s of a side / the measured bandwidth ceiling, and median GB/s of "
    "a kernel / median GB/s of torch.sum at the same size of the same run"
)
BEST_RULE = (
    "the step kernel with the highest median GB/s at this size; a tie goes to "
    "the earlier step"
)


def vendor_label(torch_version: str) -> str:
    """Name the vendor side by its call and PyTorch version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise GpuReduceError("the vendor label needs the PyTorch version")
    return f"vendor GPU sum: torch.sum on MPS, PyTorch {torch_version}"


def step_ratio(
    first: Mapping[str, object], second: Mapping[str, object], minimum: float
) -> dict[str, object]:
    """Compare two sides of one size by ``RATIO_RULE``."""

    one, two = first["timing"], second["timing"]
    ratio = first["median_gb_per_second"] / second["median_gb_per_second"]
    ahead, behind = one["p75_ns"] < two["p25_ns"], two["p75_ns"] < one["p25_ns"]
    verdict = NO_CLEAR_DIFFERENCE
    if ratio >= minimum and ahead:
        verdict = FASTER
    elif 1 / ratio >= minimum and behind:
        verdict = SLOWER
    return {
        "first": first["implementation"],
        "second": second["implementation"],
        "ratio": ratio,
        "iqr_separated": ahead or behind,
        "verdict": verdict,
    }


def _rate(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "median_gb_per_second": entry["median_gb_per_second"],
        "iqr_gb_per_second": list(entry["iqr_gb_per_second"]),
    }


def best_kernel(point: Mapping[str, object]) -> str:
    """Pick the fastest step kernel of one size by ``BEST_RULE``."""

    sides = point["sides"]
    best = STEP_KERNELS[0]
    for name in STEP_KERNELS[1:]:
        if sides[name]["median_gb_per_second"] > sides[best]["median_gb_per_second"]:
            best = name
    return best


def _point(
    point: Mapping[str, object], minimum: float, ceiling: float
) -> dict[str, object]:
    sides = point["sides"]
    if set(sides) != {*STEP_KERNELS, TORCH_SUM}:
        raise GpuReduceError("a point needs every step kernel and torch.sum")
    vendor = sides[TORCH_SUM]["median_gb_per_second"]
    rows = []
    for step in PLAN:
        entry = sides[step.kernel]
        to_base = step_ratio(entry, sides[step.base], minimum)
        predicted = step.predicted(point["n"])
        rows.append(
            {
                "step": step.step,
                "kernel": step.kernel,
                "base": step.base,
                "rate": _rate(entry),
                "fraction_of_ceiling": entry["median_gb_per_second"] / ceiling,
                "fraction_of_torch_sum": entry["median_gb_per_second"] / vendor,
                "to_base": to_base,
                "to_r0": None
                if step.kernel == R0
                else step_ratio(entry, sides[R0], minimum),
                "predicted": list(predicted),
                "held": to_base["verdict"] in predicted,
            }
        )
    best = best_kernel(point)
    return {
        "index": point["index"],
        "n": point["n"],
        "size_class": size_class(point["n"]),
        "torch_sum": {
            **_rate(sides[TORCH_SUM]),
            "fraction_of_ceiling": vendor / ceiling,
        },
        "rows": rows,
        "best": {
            "kernel": best,
            **_rate(sides[best]),
            "fraction_of_ceiling": sides[best]["median_gb_per_second"] / ceiling,
            "fraction_of_torch_sum": sides[best]["median_gb_per_second"] / vendor,
            "to_r0": None
            if best == R0
            else step_ratio(sides[best], sides[R0], minimum),
        },
    }


def _hypotheses(points: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for position, step in enumerate(PLAN):
        rows = [(point["index"], point["rows"][position]) for point in points]
        records.append(
            {
                "step": step.step,
                "kernel": step.kernel,
                "base": step.base,
                "held_at": [index for index, row in rows if row["held"]],
                "missed_at": [index for index, row in rows if not row["held"]],
            }
        )
    return records


def speedup(
    points: list[Mapping[str, object]],
    min_faster_ratio: float,
    torch_version: str,
    ceiling_gb_per_second: float,
) -> dict[str, object]:
    """Return the rules, the rows of every size and the record of each hypothesis."""

    entries = [
        _point(point, min_faster_ratio, ceiling_gb_per_second) for point in points
    ]
    return {
        "reference_kernel": R0,
        "vendor": vendor_label(torch_version),
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": min_faster_ratio,
        "ceiling_gb_per_second": ceiling_gb_per_second,
        "points": entries,
        "hypotheses": _hypotheses(entries),
    }
