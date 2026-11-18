"""Independent checks of a sealed profiling-rounds report.

The verifier first checks each embedded snapshot with ``verify_snapshot``
against the embedded config, so no snapshot number is trusted. It then
redoes the rounds with this module's own code: for each round it looks up
the target and the top hotspot in the two snapshots, recomputes the calls,
the medians, the IQRs and every speedup, and requires the same values. The
labels must agree with every snapshot, and the stage list and the overall
section must follow from the snapshots. The first field that differs is
named.

As for a snapshot, timings are checked for structure and arithmetic only.
"""

from __future__ import annotations

from collections.abc import Mapping

from .kernel_launch_report import DIGEST_FIELD
from .sim_profile_config import (
    SimProfileConfig,
    SimProfileError,
    SimProfileVerificationError,
)
from .sim_profile_rounds import ROUNDS_KIND, SPEEDUP_RULE
from .sim_profile_snapshot import (
    MEASUREMENT_BOUNDARY,
    SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
)
from .sim_profile_verify import (
    require_equal,
    require_fields,
    require_seal,
    verify_snapshot,
)

_ROUNDS_FIELDS = {
    "schema_version",
    "kind",
    "config",
    "workload_sha256",
    "timer",
    "measured",
    "scope",
    "label",
    "environment",
    "speedup_rule",
    "stages",
    "rounds",
    "overall",
    "snapshots",
    "evidence_boundary",
    DIGEST_FIELD,
}
_SHARED = ("timer", "measured", "scope", "label", "environment")


def _ratio(before: int, after: int) -> float:
    # Both medians passed the snapshot checks, so each is a positive integer.
    return round(before / after, 3)


def _entry(snapshot: Mapping, name: str) -> dict[str, object]:
    found = [
        item for item in snapshot["profile"]["functions"] if item["function"] == name
    ]
    rank = next(
        (
            item["rank"]
            for item in snapshot["profile"]["hotspots"]
            if item["function"] == name
        ),
        None,
    )
    if not found:
        return {"rank": rank, "calls": 0, "total_ns": 0, "cumulative_ns": 0}
    (item,) = found
    return {
        "rank": rank,
        "calls": item["calls"],
        "total_ns": item["total_ns"],
        "cumulative_ns": item["cumulative_ns"],
    }


def _expected_round(
    number: int, target: str, change: str, before: Mapping, after: Mapping
) -> dict[str, object]:
    workloads = []
    for old, new in zip(before["workloads"], after["workloads"]):
        old_timing, new_timing = old["timing"]["timing"], new["timing"]["timing"]
        workloads.append(
            {
                "name": old["name"],
                "before_median_ns": old_timing["median_ns"],
                "before_iqr_ns": old_timing["iqr_ns"],
                "after_median_ns": new_timing["median_ns"],
                "after_iqr_ns": new_timing["iqr_ns"],
                "speedup": _ratio(old_timing["median_ns"], new_timing["median_ns"]),
            }
        )
    old_total = sum(item["before_median_ns"] for item in workloads)
    new_total = sum(item["after_median_ns"] for item in workloads)
    return {
        "round": number,
        "before_stage": before["stage"],
        "after_stage": after["stage"],
        "target": target,
        "change": change,
        "top_hotspot_before": before["profile"]["hotspots"][0]["function"],
        "top_hotspot_after": after["profile"]["hotspots"][0]["function"],
        "target_before": _entry(before, target),
        "target_after": _entry(after, target),
        "function_calls_before": before["profile"]["calls"],
        "function_calls_after": after["profile"]["calls"],
        "workloads": workloads,
        "total": {
            "before_median_ns": old_total,
            "after_median_ns": new_total,
            "speedup": _ratio(old_total, new_total),
        },
    }


def verify_rounds(
    report: object, *, config: SimProfileConfig | None = None
) -> dict[str, object]:
    """Check a sealed rounds report; return its digest and headline speedups.

    With ``config``, the report must embed exactly that config, rounds
    included.
    """

    rounds = require_fields(report, _ROUNDS_FIELDS, "rounds report")
    require_equal(SCHEMA_VERSION, rounds["schema_version"], "schema_version")
    require_equal(ROUNDS_KIND, rounds["kind"], "kind")
    require_seal(rounds, "rounds report")
    try:
        embedded = SimProfileConfig.from_mapping(rounds["config"])
    except (TypeError, ValueError) as error:
        raise SimProfileVerificationError(
            f"config is not a valid config: {error}"
        ) from error
    require_equal(embedded.to_mapping(), rounds["config"], "config")
    if not embedded.rounds:
        raise SimProfileVerificationError("config names no round")
    if config is not None:
        if type(config) is not SimProfileConfig:
            raise TypeError("config must be SimProfileConfig")
        if config.to_mapping() != embedded.to_mapping():
            raise SimProfileError("the rounds report was made with another config")
    require_equal(
        embedded.workload_sha256, rounds["workload_sha256"], "workload_sha256"
    )
    snapshots = rounds["snapshots"]
    if type(snapshots) is not list or len(snapshots) != len(embedded.stages):
        raise SimProfileVerificationError("snapshots must hold one report per stage")
    for index, (snapshot, stage) in enumerate(zip(snapshots, embedded.stages)):
        try:
            verify_snapshot(snapshot, config=embedded)
        except SimProfileError as error:
            # A snapshot that does not bind to the embedded config is tampering.
            raise SimProfileVerificationError(f"snapshots[{index}]: {error}") from error
        require_equal(stage, snapshot["stage"], f"snapshots[{index}].stage")
        for key in _SHARED:
            require_equal(snapshot[key], rounds[key], key)
    require_equal(SPEEDUP_RULE, rounds["speedup_rule"], "speedup_rule")
    require_equal(
        MEASUREMENT_BOUNDARY if rounds["measured"] else SYNTHETIC_BOUNDARY,
        rounds["evidence_boundary"],
        "evidence_boundary",
    )
    stages = [
        {
            "stage": item["stage"],
            "report_sha256": item[DIGEST_FIELD],
            "load_average_before": item["load_average_before"],
            "load_average_after": item["load_average_after"],
            "total_median_ns": item["summary"]["total_median_ns"],
            "function_calls": item["profile"]["calls"],
            "top_hotspot": item["profile"]["hotspots"][0]["function"],
        }
        for item in snapshots
    ]
    require_equal(stages, rounds["stages"], "stages")
    expected = []
    for number, spec in enumerate(embedded.rounds, start=1):
        entry = _expected_round(
            number, spec.target, spec.change, snapshots[number - 1], snapshots[number]
        )
        if entry["target_before"]["rank"] is None:
            raise SimProfileVerificationError(
                f"rounds[{number - 1}].target is not a hotspot before its change"
            )
        expected.append(entry)
    if type(rounds["rounds"]) is not list or len(rounds["rounds"]) != len(expected):
        raise SimProfileVerificationError("rounds must hold one entry per round")
    for index, (entry, actual) in enumerate(zip(expected, rounds["rounds"])):
        require_equal(entry, actual, f"rounds[{index}]")
    first, last = stages[0], stages[-1]
    overall = {
        "baseline_median_ns": first["total_median_ns"],
        "final_median_ns": last["total_median_ns"],
        "speedup": _ratio(first["total_median_ns"], last["total_median_ns"]),
        "function_calls_baseline": first["function_calls"],
        "function_calls_final": last["function_calls"],
    }
    require_equal(overall, rounds["overall"], "overall")
    return {
        "kind": ROUNDS_KIND,
        "report_sha256": rounds[DIGEST_FIELD],
        "measured": rounds["measured"],
        "scope": rounds["scope"],
        "rounds": len(expected),
        "round_speedups": [entry["total"]["speedup"] for entry in expected],
        "speedup": overall["speedup"],
    }
