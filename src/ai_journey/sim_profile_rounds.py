"""Seal the profiling loop: one snapshot per stage, compared round by round.

Round N compares the snapshot of stage N - 1, before its change, with the
snapshot of stage N, after it. For each round the report states these
values:

* the measured top hotspot before the change, and the round's declared
  target with its rank, calls and profiler times before and after;
* the profiled function calls of all three workloads before and after;
* each workload's median and IQR before and after, and the speedup
  ``before median / after median`` rounded to 3 decimals;
* the sum of the three medians before and after, and its speedup.

Every snapshot must pass its own verifier, bind to the config's workloads,
name its stage, and use the same clock and host environment as the others.
A round must target one of the hotspots named before its change, so no
round can claim a function that the profile did not rank. The snapshots go
into the report whole, so a verifier can redo every number from them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .kernel_launch_report import DIGEST_FIELD, seal_report
from .sim_profile_config import SimProfileConfig, SimProfileError
from .sim_profile_snapshot import (
    MEASUREMENT_BOUNDARY,
    SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
)
from .sim_profile_verify import verify_snapshot

ROUNDS_KIND = "sim_profile_rounds"
SPEEDUP_DECIMALS = 3
SPEEDUP_RULE = (
    "speedup = round(before median_ns / after median_ns, 3); above 1 is faster. "
    "The total sums the three workload medians of one snapshot."
)


def speedup(before_ns: int, after_ns: int) -> float:
    """Return how many times faster the after median is, to 3 decimals."""

    if type(before_ns) is not int or type(after_ns) is not int:
        raise TypeError("medians must be integers")
    if before_ns <= 0 or after_ns <= 0:
        raise SimProfileError("medians must be positive")
    return round(before_ns / after_ns, SPEEDUP_DECIMALS)


def _function(snapshot: Mapping, name: str) -> dict[str, int | None]:
    """Return one function's rank and merged profile entry, or zero calls."""

    ranks = {item["function"]: item["rank"] for item in snapshot["profile"]["hotspots"]}
    for item in snapshot["profile"]["functions"]:
        if item["function"] == name:
            return {
                "rank": ranks.get(name),
                "calls": item["calls"],
                "total_ns": item["total_ns"],
                "cumulative_ns": item["cumulative_ns"],
            }
    return {"rank": None, "calls": 0, "total_ns": 0, "cumulative_ns": 0}


def _medians(snapshot: Mapping) -> list[tuple[str, int, int]]:
    return [
        (
            entry["name"],
            entry["timing"]["timing"]["median_ns"],
            entry["timing"]["timing"]["iqr_ns"],
        )
        for entry in snapshot["workloads"]
    ]


def _round(
    number: int, spec: object, before: Mapping, after: Mapping
) -> dict[str, object]:
    target = spec.target
    entry_before = _function(before, target)
    if entry_before["rank"] is None:
        raise SimProfileError(
            f"round {number} targets {target}, which is not a hotspot of "
            f"stage {before['stage']}"
        )
    workloads = [
        {
            "name": name,
            "before_median_ns": median_before,
            "before_iqr_ns": iqr_before,
            "after_median_ns": median_after,
            "after_iqr_ns": iqr_after,
            "speedup": speedup(median_before, median_after),
        }
        for (name, median_before, iqr_before), (_, median_after, iqr_after) in zip(
            _medians(before), _medians(after)
        )
    ]
    total_before = before["summary"]["total_median_ns"]
    total_after = after["summary"]["total_median_ns"]
    return {
        "round": number,
        "before_stage": before["stage"],
        "after_stage": after["stage"],
        "target": target,
        "change": spec.change,
        "top_hotspot_before": before["summary"]["top_hotspot"],
        "top_hotspot_after": after["summary"]["top_hotspot"],
        "target_before": entry_before,
        "target_after": _function(after, target),
        "function_calls_before": before["profile"]["calls"],
        "function_calls_after": after["profile"]["calls"],
        "workloads": workloads,
        "total": {
            "before_median_ns": total_before,
            "after_median_ns": total_after,
            "speedup": speedup(total_before, total_after),
        },
    }


def _stage_entry(snapshot: Mapping) -> dict[str, object]:
    return {
        "stage": snapshot["stage"],
        "report_sha256": snapshot[DIGEST_FIELD],
        "load_average_before": snapshot["load_average_before"],
        "load_average_after": snapshot["load_average_after"],
        "total_median_ns": snapshot["summary"]["total_median_ns"],
        "function_calls": snapshot["profile"]["calls"],
        "top_hotspot": snapshot["summary"]["top_hotspot"],
    }


def build_rounds(
    config: SimProfileConfig, snapshots: Sequence[Mapping]
) -> dict[str, object]:
    """Verify one snapshot per config stage, in order; return the unsealed report."""

    if type(config) is not SimProfileConfig:
        raise TypeError("config must be SimProfileConfig")
    if not config.rounds:
        raise SimProfileError("the config names no round to report")
    snapshots = list(snapshots)
    if len(snapshots) != len(config.stages):
        raise SimProfileError(
            f"the config has {len(config.stages)} stages; give one snapshot each"
        )
    summaries = [verify_snapshot(item, config=config) for item in snapshots]
    stages = [summary["stage"] for summary in summaries]
    if stages != list(config.stages):
        raise SimProfileError(f"snapshots must follow the stages {config.stages}")
    first = snapshots[0]
    for item in snapshots[1:]:
        if item["timer"] != first["timer"]:
            raise SimProfileError("every snapshot must use the same clock")
        if item["environment"] != first["environment"]:
            raise SimProfileError("every snapshot must come from the same environment")
    rounds = [
        _round(number, spec, snapshots[number - 1], snapshots[number])
        for number, spec in enumerate(config.rounds, start=1)
    ]
    baseline = first["summary"]["total_median_ns"]
    final = snapshots[-1]["summary"]["total_median_ns"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ROUNDS_KIND,
        "config": config.to_mapping(),
        "workload_sha256": config.workload_sha256,
        "timer": first["timer"],
        "measured": first["measured"],
        "scope": first["scope"],
        "label": first["label"],
        "environment": first["environment"],
        "speedup_rule": SPEEDUP_RULE,
        "stages": [_stage_entry(item) for item in snapshots],
        "rounds": rounds,
        "overall": {
            "baseline_median_ns": baseline,
            "final_median_ns": final,
            "speedup": speedup(baseline, final),
            "function_calls_baseline": first["profile"]["calls"],
            "function_calls_final": snapshots[-1]["profile"]["calls"],
        },
        "snapshots": snapshots,
        "evidence_boundary": dict(
            MEASUREMENT_BOUNDARY if first["measured"] else SYNTHETIC_BOUNDARY
        ),
    }


def rounds_report(
    config: SimProfileConfig, snapshots: Sequence[Mapping]
) -> dict[str, object]:
    """Build and seal the rounds report."""

    return seal_report(build_rounds(config, snapshots))
