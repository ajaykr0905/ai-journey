"""The verdict of each plan step at each size, and each side's share of the ceilings.

For each timing point the section gives each side's median TFLOP/s with
the range from the IQR of its samples, its compulsory intensity, its
roofline from the two Day 74 ceilings and two fractions: of the FMA
ceiling and of the roofline. For each step of the plan it then gives the
ratio of the first side to its base with the verdict of the Day 79
``RATIO_RULE``, and whether the verdict is one that the step predicted.

Each ratio compares two sides of one point of one run with the same calls
per sample, so it is the inverse ratio of their median sample times. The
config's ``min_faster_ratio`` is the declared 1.1 rule. A step whose first
or base side is unsupported on the host has no verdict. A PyTorch side
names the call and the PyTorch version, never a kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

from .metal_fused_speedup import RATIO_RULE, ratio_step
from .metal_tc_ceiling import ROOFLINE_RULE, roofline
from .metal_tc_plan import PLAN, SIDES, TORCH_CALL
from .metal_tc_source import MetalTcError

HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the first side of a step "
    "over its base is one of the verdicts that the step predicted; a step with "
    "an unsupported side gives no verdict"
)
FRACTION_RULE = (
    "TFLOP/s of a side * 1000 / the FMA ceiling in GFLOP/s, and the same over "
    "the roofline of the side; both ceilings come from one separate Day 74 run "
    "on this host"
)


def vendor_label(torch_version: str) -> str:
    """Name the PyTorch sides by their call and version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise MetalTcError("the vendor label needs the PyTorch version")
    return f"{TORCH_CALL}, PyTorch {torch_version}"


def _rate(entry: Mapping[str, object], bandwidth: float, fma: float) -> dict:
    gflops = entry["tflops_per_second"] * 1000
    bound = roofline(entry["intensity_flops_per_byte"], bandwidth, fma)
    return {
        "tflops_per_second": entry["tflops_per_second"],
        "iqr_tflops_per_second": list(entry["iqr_tflops_per_second"]),
        "median_ns_per_call": entry["median_ns_per_call"],
        "intensity_flops_per_byte": entry["intensity_flops_per_byte"],
        "roofline_gflops_per_second": bound,
        "fraction_of_fma": gflops / fma,
        "fraction_of_roofline": gflops / bound,
        "compute_bound": entry["intensity_flops_per_byte"] * bandwidth >= fma,
    }


def _point(point: Mapping[str, object], bandwidth: float, fma: float, minimum: float):
    sides = point["sides"]
    if not set(sides) <= set(SIDES):
        raise MetalTcError("a point holds only sides of the plan")
    steps = []
    for step in PLAN:
        if step.first not in sides or step.base not in sides:
            steps.append(
                {
                    "step": step.step,
                    "ran": False,
                    "predicted": list(step.predicted),
                    "held": None,
                }
            )
            continue
        verdict = ratio_step(sides[step.first], sides[step.base], minimum)
        steps.append(
            {
                "step": step.step,
                "ran": True,
                **verdict,
                "predicted": list(step.predicted),
                "held": verdict["verdict"] in step.predicted,
            }
        )
    return {
        "index": point["index"],
        "size": point["size"],
        "rates": {
            side: _rate(sides[side], bandwidth, fma) for side in SIDES if side in sides
        },
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
        "roofline_rule": ROOFLINE_RULE,
        "min_faster_ratio": min_faster_ratio,
        "bandwidth_gb_per_second": bandwidth_gb_per_second,
        "fma_gflops_per_second": fma_gflops_per_second,
        "ridge_flops_per_byte": fma_gflops_per_second / bandwidth_gb_per_second,
        "points": rows,
        "hypotheses": [
            {
                "step": step.step,
                "predicted": list(step.predicted),
                "held_at": [
                    row["index"] for row in rows if row["steps"][number]["held"] is True
                ],
                "missed_at": [
                    row["index"]
                    for row in rows
                    if row["steps"][number]["held"] is False
                ],
                "not_run_at": [
                    row["index"] for row in rows if row["steps"][number]["held"] is None
                ],
            }
            for number, step in enumerate(PLAN)
        ],
    }
