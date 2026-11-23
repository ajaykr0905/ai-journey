"""The speedup of each fusion step and its share of the bandwidth ceiling.

For each timing point and each step of the plan, the section gives:

- the effective GB/s of the step (compulsory bytes over the median sample)
  with the range from the IQR of its samples, and its modeled GB/s;
- the ratio to the step's base, the step before it, and the ratio to U0,
  the unfused baseline, each with the verdict of ``RATIO_RULE``;
- the ratio to ``torch.softmax`` at the same point, with its verdict;
- the fraction of the bandwidth ceiling, for the effective and the modeled
  bytes;
- whether the verdict over the base is one that the hypothesis predicted.

Each ratio compares two sides of one point of one run with the same calls
per sample, so it is the inverse ratio of their median sample times. The
config's ``min_faster_ratio`` is the declared 1.1 rule. A step that its
pipeline refused has no numbers, and a step whose base did not run has no
verdict over it. The vendor side names the call and the PyTorch version,
never a kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER
from .metal_fused_plan import BASELINE, PLAN, TORCH_SOFTMAX
from .metal_fused_source import STEP_VARIANTS, MetalFusedError

RATIO_RULE = (
    "ratio = median sample time of the second side / median sample time of the "
    "first side, from the same point of the same run with the same calls per "
    "sample; faster when ratio >= min_faster_ratio and the first side's p75 "
    "sample time is below the second side's p25; slower when 1 / ratio >= "
    "min_faster_ratio and the second side's p75 is below the first side's p25; "
    "otherwise no_clear_difference"
)
HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the step over its base is "
    "one of the verdicts that the step predicted; a step or base that did not "
    "run gives no verdict"
)
FRACTION_RULE = (
    "effective GB/s of a side / the bandwidth ceiling GB/s, and for a step also "
    "its modeled GB/s / the ceiling; the ceiling comes from a separate run"
)
BEST_RULE = (
    "the step with the highest effective GB/s at this point; a tie goes to the "
    "earlier step of the plan"
)


def vendor_label(torch_version: str) -> str:
    """Name the vendor side by its call and PyTorch version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise MetalFusedError("the vendor label needs the PyTorch version")
    return f"vendor GPU softmax: torch.softmax(dim=-1) on MPS, PyTorch {torch_version}"


def ratio_step(
    first: Mapping[str, object], second: Mapping[str, object], minimum: float
) -> dict[str, object]:
    """Compare two sides of one point by ``RATIO_RULE``."""

    one, two = first["timing"], second["timing"]
    ratio = two["median_ns"] / one["median_ns"]
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


def _rate(entry: Mapping[str, object], ceiling: float) -> dict[str, object]:
    modeled = entry["modeled_gb_per_second"]
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": list(entry["iqr_effective_gb_per_second"]),
        "modeled_gb_per_second": modeled,
        "effective_fraction_of_ceiling": entry["effective_gb_per_second"] / ceiling,
        "modeled_fraction_of_ceiling": None if modeled is None else modeled / ceiling,
    }


def _compare(sides, name, other, minimum):
    if name == other or name not in sides or other not in sides:
        return None
    return ratio_step(sides[name], sides[other], minimum)


def _point(
    point: Mapping[str, object], ceiling: float, minimum: float
) -> dict[str, object]:
    sides = point["sides"]
    if TORCH_SOFTMAX not in sides:
        raise MetalFusedError("a point needs the torch.softmax side")
    rows = []
    for step in PLAN:
        name = step.variant
        to_base = _compare(sides, name, step.base, minimum)
        rows.append(
            {
                "step": step.step,
                "variant": name,
                "base": step.base,
                "ran": name in sides,
                "rate": _rate(sides[name], ceiling) if name in sides else None,
                "to_base": to_base,
                "to_u0": _compare(sides, name, BASELINE, minimum),
                "to_torch": _compare(sides, name, TORCH_SOFTMAX, minimum),
                "predicted": list(step.predicted),
                "held": None
                if to_base is None
                else to_base["verdict"] in step.predicted,
            }
        )
    best = None
    for name in STEP_VARIANTS:
        if name in sides and (
            best is None
            or sides[name]["effective_gb_per_second"]
            > sides[best]["effective_gb_per_second"]
        ):
            best = name
    return {
        "index": point["index"],
        "shape": dict(point["shape"]),
        "torch": {
            "rate": _rate(sides[TORCH_SOFTMAX], ceiling),
            "to_u0": _compare(sides, TORCH_SOFTMAX, BASELINE, minimum),
        },
        "best": None
        if best is None
        else {
            "variant": best,
            "to_u0": _compare(sides, best, BASELINE, minimum),
            "to_torch": ratio_step(sides[best], sides[TORCH_SOFTMAX], minimum),
        },
        "rows": rows,
    }


def _hypotheses(points: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for step in PLAN:
        held = [
            (point["index"], row["held"])
            for point in points
            for row in point["rows"]
            if row["variant"] == step.variant
        ]
        records.append(
            {
                "step": step.step,
                "variant": step.variant,
                "predicted": list(step.predicted),
                "held_at": [index for index, value in held if value is True],
                "missed_at": [index for index, value in held if value is False],
                "no_verdict_at": [index for index, value in held if value is None],
            }
        )
    return records


def speedup(
    points: list[Mapping[str, object]],
    ceiling_gb_per_second: float,
    min_faster_ratio: float,
    torch_version: str,
) -> dict[str, object]:
    """Return the rules, every point's rows and the record of each hypothesis."""

    rows = [_point(point, ceiling_gb_per_second, min_faster_ratio) for point in points]
    return {
        "baseline": BASELINE,
        "vendor": vendor_label(torch_version),
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": min_faster_ratio,
        "ceiling_gb_per_second": ceiling_gb_per_second,
        "points": rows,
        "hypotheses": _hypotheses(rows),
    }
