"""The verdict of each plan step at each point, and the shares of both ceilings.

For each timing point the section gives each side's effective GB/s
(compulsory bytes over the median sample) and GFLOP/s, each with the range
from the IQR of its samples, its modeled GB/s, and three fractions: the
effective and the modeled GB/s of the bandwidth ceiling, and the GFLOP/s of
the FMA ceiling. It then gives, for each step of the plan, the ratio of the
first side to its base with the verdict of the Day 79 ``RATIO_RULE``, and
whether the verdict is one that the step predicted.

Each ratio compares two sides of one point of one run with the same calls
per sample, so it is the inverse ratio of their median sample times. The
config's ``min_faster_ratio`` is the declared 1.1 rule. The PyTorch side
names the call and the PyTorch version, never a kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

from .metal_flash_plan import PLAN, SIDES, TORCH_CALL
from .metal_flash_source import MetalFlashError
from .metal_fused_speedup import RATIO_RULE, ratio_step

HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the first side of a step "
    "over its base is one of the verdicts that the step predicted"
)
FRACTION_RULE = (
    "effective GB/s of a side / the bandwidth ceiling, modeled GB/s / the "
    "bandwidth ceiling, and GFLOP/s / the FMA ceiling; both ceilings come from "
    "one separate Day 74 run on this host"
)


def vendor_label(torch_version: str) -> str:
    """Name the PyTorch side by its call and version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise MetalFlashError("the vendor label needs the PyTorch version")
    return f"{TORCH_CALL}, PyTorch {torch_version}"


def _rate(entry: Mapping[str, object], bandwidth: float, fma: float) -> dict:
    modeled = entry["modeled_gb_per_second"]
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": list(entry["iqr_effective_gb_per_second"]),
        "modeled_gb_per_second": modeled,
        "gflops_per_second": entry["gflops_per_second"],
        "iqr_gflops_per_second": list(entry["iqr_gflops_per_second"]),
        "median_ns_per_call": entry["median_ns_per_call"],
        "effective_fraction_of_bandwidth": entry["effective_gb_per_second"] / bandwidth,
        "modeled_fraction_of_bandwidth": None
        if modeled is None
        else modeled / bandwidth,
        "fraction_of_fma": entry["gflops_per_second"] / fma,
    }


def _point(point: Mapping[str, object], bandwidth: float, fma: float, minimum: float):
    sides = point["sides"]
    if set(sides) != set(SIDES):
        raise MetalFlashError("a point needs the three sides of the plan")
    steps = []
    for step in PLAN:
        verdict = ratio_step(sides[step.first], sides[step.base], minimum)
        steps.append(
            {
                "step": step.step,
                **verdict,
                "predicted": list(step.predicted),
                "held": verdict["verdict"] in step.predicted,
            }
        )
    return {
        "index": point["index"],
        "sequences": point["sequences"],
        "length": point["length"],
        "mode": point["mode"],
        "rates": {side: _rate(sides[side], bandwidth, fma) for side in SIDES},
        "steps": steps,
    }


def speedup(
    points: list[Mapping[str, object]],
    bandwidth_gb_per_second: float,
    fma_gflops_per_second: float,
    min_faster_ratio: float,
    torch_version: str,
) -> dict[str, object]:
    """Return the rules, every point's rows and the record of each hypothesis."""

    rows = [
        _point(point, bandwidth_gb_per_second, fma_gflops_per_second, min_faster_ratio)
        for point in points
    ]
    return {
        "vendor": vendor_label(torch_version),
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "min_faster_ratio": min_faster_ratio,
        "bandwidth_gb_per_second": bandwidth_gb_per_second,
        "fma_gflops_per_second": fma_gflops_per_second,
        "points": rows,
        "hypotheses": [
            {
                "step": step.step,
                "predicted": list(step.predicted),
                "held_at": [
                    row["index"] for row in rows if row["steps"][number]["held"]
                ],
                "missed_at": [
                    row["index"] for row in rows if not row["steps"][number]["held"]
                ],
            }
            for number, step in enumerate(PLAN)
        ],
    }
