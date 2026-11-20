"""The speedup after each checklist change, from the samples of one run only.

For each sweep point and each kernel that a step adds, the section gives:

- the median GFLOP/s of the kernel and the range from the IQR of its samples
- the step ratio to the step's base, the kernel that the change starts from
- the step ratio to S0, ``matmul_x_columns``, the speedup of every change
  so far
- the fraction of the ``torch.matmul`` median at the same point
- whether the verdict over the base is one that the hypothesis predicted

Each ratio divides two median rates of one point of one run, and the Day 76
``RATIO_RULE`` names it ``faster``, ``slower`` or ``no_clear_difference``
with the config's ``min_faster_ratio``. A kernel that its pipeline refused
has no numbers, and a kernel whose base was refused has no verdict over it.
The vendor side names the call and the PyTorch version, never a kernel, and
no point holds a cuBLAS number.
"""

from __future__ import annotations

from collections.abc import Mapping

from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_steps import CHECKLIST, REFERENCE, ChecklistStep
from .matmul_checklist_variants import VARIANTS
from .matmul_metal_runtime import TORCH_MATMUL
from .matmul_tiled_comparison import CUBLAS, RATIO_RULE, vendor_label
from .matmul_tiled_comparison import step as ratio_step

HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the kernel over its base "
    "is one of the verdicts that its step predicted; a kernel or base that did "
    "not run gives no verdict"
)
FRACTION_RULE = (
    "median GFLOP/s of the kernel / median GFLOP/s of torch.matmul at the same "
    "point of the same run"
)
BEST_RULE = (
    "the timed kernel with the highest median GFLOP/s at this point; a tie goes "
    "to the kernel that the registry lists first"
)


def _rate(entry: Mapping[str, object] | None) -> dict[str, object] | None:
    if entry is None:
        return None
    return {
        "median_gflops_per_second": entry["median_gflops_per_second"],
        "iqr_gflops_per_second": list(entry["iqr_gflops_per_second"]),
    }


def _row(
    point: Mapping[str, object],
    step: ChecklistStep,
    variant: str,
    minimum: float,
) -> dict[str, object]:
    sides = point["sides"]
    entry, base = sides.get(variant), sides.get(step.base)
    to_base = (
        None if entry is None or base is None else ratio_step(entry, base, minimum)
    )
    held = None if to_base is None else to_base["verdict"] in step.predicted(variant)
    return {
        "step": step.step,
        "item": step.item,
        "variant": variant,
        "base": step.base,
        "ran": entry is not None,
        "rate": _rate(entry),
        "base_rate": _rate(base),
        "to_base": to_base,
        "to_s0": None
        if entry is None
        else ratio_step(entry, sides[REFERENCE], minimum),
        "fraction_of_torch": None
        if entry is None
        else entry["median_gflops_per_second"]
        / sides[TORCH_MATMUL]["median_gflops_per_second"],
        "predicted": list(step.predicted(variant)),
        "held": held,
    }


def best_kernel(point: Mapping[str, object]) -> str:
    """Pick the fastest timed kernel of one point by ``BEST_RULE``."""

    best = None
    for name in VARIANTS:
        if name not in point["sides"]:
            continue
        rate = point["sides"][name]["median_gflops_per_second"]
        if best is None or rate > point["sides"][best]["median_gflops_per_second"]:
            best = name
    return best


def _point(point: Mapping[str, object], minimum: float) -> dict[str, object]:
    sides = point["sides"]
    if REFERENCE not in sides or TORCH_MATMUL not in sides:
        raise MatmulChecklistError("a point needs the S0 kernel and torch.matmul")
    best = best_kernel(point)
    torch_rate = sides[TORCH_MATMUL]["median_gflops_per_second"]
    return {
        "index": point["index"],
        "m": point["m"],
        "n": point["n"],
        "k": point["k"],
        "torch": _rate(sides[TORCH_MATMUL]),
        "best": {
            "variant": best,
            "to_s0": ratio_step(sides[best], sides[REFERENCE], minimum),
            "fraction_of_torch": sides[best]["median_gflops_per_second"] / torch_rate,
        },
        "rows": [
            _row(point, step, variant, minimum)
            for step in CHECKLIST
            for variant in step.variants
        ],
    }


def _hypotheses(points: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for step in CHECKLIST:
        for variant in step.variants:
            rows = [
                (point["index"], row)
                for point in points
                for row in point["rows"]
                if row["variant"] == variant
            ]
            records.append(
                {
                    "step": step.step,
                    "variant": variant,
                    "predicted": list(step.predicted(variant)),
                    "held_at": [index for index, row in rows if row["held"] is True],
                    "missed_at": [index for index, row in rows if row["held"] is False],
                    "no_verdict_at": [
                        index for index, row in rows if row["held"] is None
                    ],
                }
            )
    return records


def speedup(
    points: list[Mapping[str, object]],
    min_faster_ratio: float,
    torch_version: str,
) -> dict[str, object]:
    """Return the rules, every point's rows and the record of each hypothesis."""

    rows = [_point(point, min_faster_ratio) for point in points]
    return {
        "reference": REFERENCE,
        "vendor": vendor_label(torch_version),
        "cublas": CUBLAS,
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": min_faster_ratio,
        "points": rows,
        "hypotheses": _hypotheses(rows),
    }
