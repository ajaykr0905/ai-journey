"""The rate of each side and the ratios between sides, from one run's samples.

For each sweep point and each kernel of the plan, the section gives:

- the median GB/s of the kernel and the range from the IQR of its samples;
- the ratio to its base in the plan, with its verdict;
- the ratios to S0 and to ``torch.cumsum``, with their verdicts;
- the fraction of the copy rate at the same point, and the modeled fraction:
  the modeled bytes of the scan over its median time, over the copy rate;
- whether the verdict over the base is one that the plan predicted at this
  length. A length that the plan does not pin has no prediction.

Each ratio divides two median rates of one point of one run. The scan sides
and ``torch.cumsum`` move the same 8 N bytes in each call, so the ratio of
rates is the inverse ratio of median times. ``RATIO_RULE`` names it with the
config's ``min_faster_ratio``, as the Day 76 rule names a GFLOP/s ratio. A
kernel that its pipeline refused has no numbers, and a kernel whose base did
not run has no verdict over it. The copy is a same-run reference for the
device traffic, not a scan, so it takes part in no verdict. PyTorch does not
report which kernel ``torch.cumsum`` runs, so the vendor side names the call
and the PyTorch version, never a kernel.
"""

from __future__ import annotations

from collections.abc import Mapping

from .gpu_scan_plan import COPY, PLAN, VENDOR, ScanHypothesis
from .gpu_scan_source import KERNELS, S0_HILLIS_STEELE, GpuScanError
from .matmul_tiled_comparison import FASTER, NO_CLEAR_DIFFERENCE, SLOWER

RATIO_RULE = (
    "ratio = median GB/s of the first side / median GB/s of the second, from the "
    "same point of the same run; faster when ratio >= min_faster_ratio and the "
    "first side's p75 sample time is below the second side's p25; slower when "
    "1 / ratio >= min_faster_ratio and the second side's p75 is below the first "
    "side's p25; otherwise no_clear_difference"
)
HYPOTHESIS_RULE = (
    "a prediction holds at a point when the verdict of the kernel over its base "
    "is one that the plan predicted at that length; a kernel or base that did "
    "not run gives no verdict, and a length that the plan does not pin gives no "
    "prediction"
)
COPY_RULE = (
    "fraction_of_copy = median GB/s of the side / median GB/s of stream_copy at "
    "the same point; modeled_fraction_of_copy uses the modeled bytes of one scan "
    "instead of its 8 N bytes"
)
BEST_RULE = (
    "the timed kernel with the highest median GB/s at this point; a tie goes to "
    "the kernel that the registry lists first"
)


def vendor_label(torch_version: str) -> str:
    """Name the vendor side by its call and PyTorch version, not by a kernel."""

    if not isinstance(torch_version, str) or not torch_version:
        raise GpuScanError("the vendor label needs the PyTorch version")
    return f"vendor GPU scan: torch.cumsum on MPS, PyTorch {torch_version}"


def ratio_step(
    first: Mapping[str, object], second: Mapping[str, object], minimum: float
) -> dict[str, object]:
    """Compare two sides of one point by ``RATIO_RULE``."""

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


def _rate(entry: Mapping[str, object] | None) -> dict[str, object] | None:
    if entry is None:
        return None
    return {
        "median_gb_per_second": entry["median_gb_per_second"],
        "iqr_gb_per_second": list(entry["iqr_gb_per_second"]),
    }


def _step(
    entry: Mapping[str, object] | None,
    other: Mapping[str, object] | None,
    minimum: float,
) -> dict[str, object] | None:
    if entry is None or other is None:
        return None
    return ratio_step(entry, other, minimum)


def _row(
    point: Mapping[str, object], step: ScanHypothesis, minimum: float
) -> dict[str, object]:
    sides = point["sides"]
    entry, base = sides.get(step.kernel), sides.get(step.base)
    copy = sides[COPY]["median_gb_per_second"]
    to_base = _step(entry, base, minimum)
    predicted = step.predicted(point["length"])
    held = (
        None
        if to_base is None or predicted is None
        else to_base["verdict"] in predicted
    )
    modeled = (
        None
        if entry is None
        else entry["calls_per_sample"]
        * point["modeled_bytes_per_call"]
        / entry["timing"]["median_ns"]
    )
    return {
        "step": step.step,
        "kernel": step.kernel,
        "base": step.base,
        "ran": entry is not None,
        "rate": _rate(entry),
        "to_base": to_base,
        "to_s0": None
        if step.kernel == S0_HILLIS_STEELE
        else _step(entry, sides.get(S0_HILLIS_STEELE), minimum),
        "to_vendor": _step(entry, sides[VENDOR], minimum),
        "fraction_of_copy": None
        if entry is None
        else entry["median_gb_per_second"] / copy,
        "modeled_gb_per_second": modeled,
        "modeled_fraction_of_copy": None if modeled is None else modeled / copy,
        "predicted": None if predicted is None else list(predicted),
        "held": held,
    }


def best_kernel(point: Mapping[str, object]) -> str | None:
    """Pick the fastest timed kernel of one point by ``BEST_RULE``."""

    best = None
    for name in KERNELS:
        if name not in point["sides"]:
            continue
        rate = point["sides"][name]["median_gb_per_second"]
        if best is None or rate > point["sides"][best]["median_gb_per_second"]:
            best = name
    return best


def _point(point: Mapping[str, object], minimum: float) -> dict[str, object]:
    sides = point["sides"]
    if VENDOR not in sides or COPY not in sides:
        raise GpuScanError("a point needs torch.cumsum and the copy")
    best = best_kernel(point)
    vendor, copy = sides[VENDOR], sides[COPY]
    return {
        "index": point["index"],
        "length": point["length"],
        "vendor": {
            **_rate(vendor),
            "fraction_of_copy": vendor["median_gb_per_second"]
            / copy["median_gb_per_second"],
        },
        "copy": _rate(copy),
        "best": None
        if best is None
        else {
            "kernel": best,
            "to_vendor": ratio_step(sides[best], vendor, minimum),
            "fraction_of_copy": sides[best]["median_gb_per_second"]
            / copy["median_gb_per_second"],
        },
        "rows": [_row(point, step, minimum) for step in PLAN],
    }


def _hypotheses(points: list[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for step in PLAN:
        rows = [
            (point["index"], row)
            for point in points
            for row in point["rows"]
            if row["kernel"] == step.kernel
        ]
        records.append(
            {
                "step": step.step,
                "kernel": step.kernel,
                "base": step.base,
                "held_at": [index for index, row in rows if row["held"] is True],
                "missed_at": [index for index, row in rows if row["held"] is False],
                "no_verdict_at": [
                    index for index, row in rows if row["to_base"] is None
                ],
                "no_prediction_at": [
                    index
                    for index, row in rows
                    if row["to_base"] is not None and row["predicted"] is None
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
        "vendor": vendor_label(torch_version),
        "copy": COPY,
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "copy_rule": COPY_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": min_faster_ratio,
        "points": rows,
        "hypotheses": _hypotheses(rows),
    }
