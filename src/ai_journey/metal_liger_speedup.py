"""The speedup of each fused kernel over its PyTorch sequence, and ceiling shares.

For each timing point, the section gives each side's effective GB/s
(compulsory bytes over the median sample) with the range from the IQR of
its samples, its modeled GB/s, and both as a fraction of the bandwidth
ceiling. It then gives the ratio of the fused kernel to the PyTorch
sequence with the verdict of the Day 79 ``RATIO_RULE``, and whether that
verdict is one that the declared hypothesis predicted.

Each ratio compares two sides of one point of one run with the same calls
per sample, so it is the inverse ratio of their median sample times. The
config's ``min_faster_ratio`` is the declared 1.1 rule. The PyTorch side
names the sequence and the PyTorch version, never a kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

from .metal_fused_speedup import RATIO_RULE, ratio_step
from .metal_liger_plan import PLAN, TORCH_SEQUENCES, step_of
from .metal_liger_source import MetalLigerError

HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the fused kernel over its "
    "PyTorch sequence is one of the verdicts that the plan predicted"
)
FRACTION_RULE = (
    "effective GB/s of a side / the bandwidth ceiling GB/s, and its modeled GB/s "
    "/ the ceiling; the ceiling comes from a separate Day 74 run on this host"
)


def vendor_label(family: str, torch_version: str) -> str:
    """Name a PyTorch side by its sequence and PyTorch version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise MetalLigerError("the vendor label needs the PyTorch version")
    return f"{TORCH_SEQUENCES[family]}, PyTorch {torch_version}"


def _rate(entry: Mapping[str, object], ceiling: float) -> dict[str, object]:
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": list(entry["iqr_effective_gb_per_second"]),
        "modeled_gb_per_second": entry["modeled_gb_per_second"],
        "effective_fraction_of_ceiling": entry["effective_gb_per_second"] / ceiling,
        "modeled_fraction_of_ceiling": entry["modeled_gb_per_second"] / ceiling,
    }


def _point(point: Mapping[str, object], ceiling: float, minimum: float) -> dict:
    step = step_of(point["family"])
    sides = point["sides"]
    if set(sides) != {step.kernel, step.base}:
        raise MetalLigerError("a point needs the kernel and its PyTorch side")
    to_torch = ratio_step(sides[step.kernel], sides[step.base], minimum)
    return {
        "family": step.family,
        "index": point["index"],
        "shape": dict(point["shape"]),
        "kernel": _rate(sides[step.kernel], ceiling),
        "torch": _rate(sides[step.base], ceiling),
        "to_torch": to_torch,
        "predicted": list(step.predicted),
        "held": to_torch["verdict"] in step.predicted,
    }


def speedup(
    points: list[Mapping[str, object]],
    ceiling_gb_per_second: float,
    min_faster_ratio: float,
    torch_version: str,
) -> dict[str, object]:
    """Return the rules, every point's row and the record of each hypothesis."""

    rows = [_point(point, ceiling_gb_per_second, min_faster_ratio) for point in points]
    return {
        "vendors": {
            step.family: vendor_label(step.family, torch_version) for step in PLAN
        },
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "min_faster_ratio": min_faster_ratio,
        "ceiling_gb_per_second": ceiling_gb_per_second,
        "points": rows,
        "hypotheses": [
            {
                "family": step.family,
                "predicted": list(step.predicted),
                "held_at": [
                    row["index"]
                    for row in rows
                    if row["family"] == step.family and row["held"]
                ],
                "missed_at": [
                    row["index"]
                    for row in rows
                    if row["family"] == step.family and not row["held"]
                ],
            }
            for step in PLAN
        ],
    }
