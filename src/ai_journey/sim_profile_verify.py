"""Independent checks of sealed simulator-profiling snapshots.

The verifier trusts no derived number in a snapshot. It checks the field
sets, the fixed labels and evidence boundary, and the SHA-256 seal. It
parses the embedded config and recomputes each workload's launch geometry
and reference digest from it, with the pure-Python references and without
the simulator. It then redoes the arithmetic with this module's own code:
the per-workload profile totals, the merged profile, the hotspot ranking
and its shares, the nearest-rank statistics of every timed sample list,
and the summary. The first field that differs is named.

Timings are checked for structure only: positive integer samples, as many
as the config repeats. No check can prove that a timing was observed, and
the profiler times only rank functions within one profile.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from hashlib import sha256

from .kernel_launch_io import canonical_json_bytes
from .kernel_launch_report import DIGEST_FIELD
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .sim_profile_config import (
    MAX_ROUNDS,
    WORKLOADS,
    SimProfileConfig,
    SimProfileError,
    SimProfileVerificationError,
)
from .sim_profile_harness import RANKING
from .sim_profile_semantics import SEMANTICS_SHA256, SEMANTICS_TOTALS
from .sim_profile_snapshot import (
    GETITEM,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SCHEMA_VERSION,
    SETITEM,
    SNAPSHOT_KIND,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .sim_profile_timing import LOAD_AVERAGE_SOURCE
from .sim_profile_workloads import build_workloads

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STAGE = re.compile(r"baseline|round-[1-9][0-9]?")
_SNAPSHOT_FIELDS = {
    "schema_version",
    "kind",
    "stage",
    "config",
    "workload_sha256",
    "environment",
    "timer",
    "measured",
    "scope",
    "label",
    "percentile_method",
    "ranking",
    "load_average_source",
    "load_average_before",
    "load_average_after",
    "semantics",
    "workloads",
    "profile",
    "summary",
    "evidence_boundary",
    DIGEST_FIELD,
}
_WORKLOAD_FIELDS = {
    "name",
    "launch",
    "output_dtype",
    "reference_sha256",
    "output_sha256",
    "reads",
    "writes",
    "profile",
    "timing",
}
_FUNCTION_FIELDS = {
    "function",
    "calls",
    "primitive_calls",
    "total_ns",
    "cumulative_ns",
}
_SEMANTICS_FIELDS = {
    "sha256",
    "runs",
    "out_of_bounds_reads",
    "out_of_bounds_writes",
    "write_conflicts",
    "read_write_conflicts",
    "canonical_records_order_free",
    "guarded_output_order_free",
}
_ENVIRONMENT_FIELDS = {
    "python_version",
    "python_implementation",
    "platform_system",
    "platform_machine",
}
_PPM = 1_000_000


def _fail(path: str, problem: str) -> SimProfileVerificationError:
    return SimProfileVerificationError(f"{path} {problem}")


def require_fields(value: object, expected: set[str], path: str) -> Mapping:
    """Require a JSON object with exactly the expected field names."""

    if type(value) is not dict:
        raise _fail(path, "must be a JSON object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise _fail(path, f"fields mismatch: missing={missing}, extra={extra}")
    return value


def require_equal(expected: object, actual: object, path: str) -> None:
    """Require exact JSON equality; ``True`` never equals ``1`` here."""

    if canonical_json_bytes(expected) != canonical_json_bytes(actual):
        raise _fail(path, "does not match its recomputed value")


def require_count(value: object, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(path, f"must be an integer of at least {minimum}")
    return value


def require_seal(report: Mapping, path: str) -> None:
    """Recompute the SHA-256 of the canonical report without its digest."""

    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    digest = report.get(DIGEST_FIELD)
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise _fail(f"{path}.{DIGEST_FIELD}", "must be a SHA-256 hex digest")
    if sha256(canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise _fail(f"{path}.{DIGEST_FIELD}", "does not match the report content")


def nearest_rank(samples: list[int], percent: int) -> int:
    """Return sorted samples[ceil(percent * n / 100) - 1], with no interpolation."""

    ordered = sorted(samples)
    return ordered[(percent * len(ordered) + 99) // 100 - 1]


def timing_statistics(samples: list[int]) -> dict[str, int]:
    """Recompute min, quartiles, median, max and IQR of positive samples."""

    p25, p50, p75 = (nearest_rank(samples, percent) for percent in (25, 50, 75))
    return {
        "count": len(samples),
        "min_ns": min(samples),
        "p25_ns": p25,
        "median_ns": p50,
        "p75_ns": p75,
        "max_ns": max(samples),
        "iqr_ns": p75 - p25,
    }


def _load_average(value: object, path: str) -> list[float]:
    if type(value) is not list or len(value) != 3:
        raise _fail(path, "must hold the 1, 5 and 15 minute load averages")
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(item) or item < 0:
            raise _fail(path, "must hold three finite nonnegative numbers")
    return value


def _functions(value: object, path: str) -> list[Mapping]:
    if type(value) is not list or not value:
        raise _fail(path, "must be a nonempty list")
    names = []
    for index, item in enumerate(value):
        entry = require_fields(item, _FUNCTION_FIELDS, f"{path}[{index}]")
        if type(entry["function"]) is not str or not entry["function"]:
            raise _fail(f"{path}[{index}].function", "must be a nonempty string")
        calls = require_count(entry["calls"], f"{path}[{index}].calls", 1)
        primitive = require_count(
            entry["primitive_calls"], f"{path}[{index}].primitive_calls", 1
        )
        if primitive > calls:
            raise _fail(f"{path}[{index}]", "has more primitive calls than calls")
        require_count(entry["total_ns"], f"{path}[{index}].total_ns")
        require_count(entry["cumulative_ns"], f"{path}[{index}].cumulative_ns")
        names.append(entry["function"])
    if names != sorted(set(names)):
        raise _fail(path, "must list unique functions sorted by name")
    return value


def _totals(functions: list[Mapping]) -> dict[str, int]:
    return {
        key: sum(entry[source] for entry in functions)
        for key, source in (
            ("calls", "calls"),
            ("primitive_calls", "primitive_calls"),
            ("total_ns", "total_ns"),
        )
    }


def merged_functions(lists: list[list[Mapping]]) -> list[dict[str, object]]:
    """Sum the entries of several profiles by function name, sorted by name."""

    sums: dict[str, list[int]] = {}
    for functions in lists:
        for entry in functions:
            total = sums.setdefault(entry["function"], [0, 0, 0, 0])
            for position, key in enumerate(sorted(_FUNCTION_FIELDS - {"function"})):
                total[position] += entry[key]
    keys = sorted(_FUNCTION_FIELDS - {"function"})
    return [{"function": name, **dict(zip(keys, sums[name]))} for name in sorted(sums)]


def ranked_hotspots(functions: list[Mapping], top: int) -> list[dict[str, object]]:
    """Rank by self time, largest first, ties by name; floor shares in ppm."""

    grand = sum(entry["total_ns"] for entry in functions)
    if grand <= 0:
        raise _fail("profile", "recorded no time")
    ordered = sorted(
        functions, key=lambda entry: (-entry["total_ns"], entry["function"])
    )
    return [
        {
            "rank": rank,
            "function": entry["function"],
            "calls": entry["calls"],
            "total_ns": entry["total_ns"],
            "cumulative_ns": entry["cumulative_ns"],
            "share_ppm": entry["total_ns"] * _PPM // grand,
        }
        for rank, entry in enumerate(ordered[:top], start=1)
    ]


def _parse_config(mapping: object) -> SimProfileConfig:
    if type(mapping) is not dict:
        raise _fail("config", "must be a JSON object")
    try:
        return SimProfileConfig.from_mapping({**mapping, "rounds": []})
    except (TypeError, ValueError) as error:
        raise _fail("config", f"is not a valid config: {error}") from error


def _check_labels(report: Mapping) -> bool:
    timer = report["timer"]
    if timer not in (TIMER, INJECTED_TIMER):
        raise _fail("timer", f"must be {TIMER!r} or {INJECTED_TIMER!r}")
    measured = timer == TIMER
    require_equal(measured, report["measured"], "measured")
    require_equal(
        MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE, report["scope"], "scope"
    )
    require_equal(
        MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL, report["label"], "label"
    )
    require_equal(
        MEASUREMENT_BOUNDARY if measured else SYNTHETIC_BOUNDARY,
        report["evidence_boundary"],
        "evidence_boundary",
    )
    require_equal(PERCENTILE_METHOD, report["percentile_method"], "percentile_method")
    require_equal(RANKING, report["ranking"], "ranking")
    require_equal(
        LOAD_AVERAGE_SOURCE, report["load_average_source"], "load_average_source"
    )
    return measured


def _check_semantics(value: object) -> None:
    semantics = require_fields(value, _SEMANTICS_FIELDS, "semantics")
    require_equal(SEMANTICS_SHA256, semantics["sha256"], "semantics.sha256")
    for key in sorted(_SEMANTICS_FIELDS - {"sha256"}):
        expected = True if key.endswith("order_free") else SEMANTICS_TOTALS[key]
        require_equal(expected, semantics[key], f"semantics.{key}")


def _check_workload(
    entry: object, index: int, expected: object, config: SimProfileConfig
) -> tuple[list[Mapping], int]:
    path = f"workloads[{index}]"
    workload = require_fields(entry, _WORKLOAD_FIELDS, path)
    require_equal(expected.name, workload["name"], f"{path}.name")
    require_equal(expected.config.geometry(), workload["launch"], f"{path}.launch")
    require_equal(
        expected.output_dtype, workload["output_dtype"], f"{path}.output_dtype"
    )
    for key in ("reference_sha256", "output_sha256"):
        require_equal(expected.reference_sha256, workload[key], f"{path}.{key}")
    profile = require_fields(
        workload["profile"],
        {"calls", "primitive_calls", "total_ns", "functions"},
        f"{path}.profile",
    )
    functions = _functions(profile["functions"], f"{path}.profile.functions")
    totals = _totals(functions)
    for key, value in totals.items():
        require_equal(value, profile[key], f"{path}.profile.{key}")
    calls = {item["function"]: item["calls"] for item in functions}
    # The pinned kernels are guarded, so every buffer call is in bounds.
    require_equal(calls.get(GETITEM, 0), workload["reads"], f"{path}.reads")
    require_equal(calls.get(SETITEM, 0), workload["writes"], f"{path}.writes")
    timing = require_fields(
        workload["timing"],
        {"warmup", "repeats", "samples_ns", "timing"},
        f"{path}.timing",
    )
    require_equal(config.warmup, timing["warmup"], f"{path}.timing.warmup")
    require_equal(config.repeats, timing["repeats"], f"{path}.timing.repeats")
    samples = timing["samples_ns"]
    if type(samples) is not list or len(samples) != config.repeats:
        raise _fail(f"{path}.timing.samples_ns", "must hold one sample per repeat")
    for sample in samples:
        require_count(sample, f"{path}.timing.samples_ns", 1)
    statistics = timing_statistics(samples)
    require_equal(statistics, timing["timing"], f"{path}.timing.timing")
    return functions, statistics["median_ns"]


def verify_snapshot(
    report: object, *, config: SimProfileConfig | None = None
) -> dict[str, object]:
    """Check one sealed snapshot; return its stage, digest and headline numbers.

    With ``config``, the snapshot must also bind to that config's workloads
    and name one of its stages.
    """

    snapshot = require_fields(report, _SNAPSHOT_FIELDS, "snapshot")
    require_equal(SCHEMA_VERSION, snapshot["schema_version"], "schema_version")
    require_equal(SNAPSHOT_KIND, snapshot["kind"], "kind")
    require_seal(snapshot, "snapshot")
    stage = snapshot["stage"]
    if type(stage) is not str or _STAGE.fullmatch(stage) is None:
        raise _fail("stage", "must be baseline or round-1 to round-8")
    if stage != "baseline" and int(stage.split("-")[1]) > MAX_ROUNDS:
        raise _fail("stage", "must be baseline or round-1 to round-8")
    embedded = _parse_config(snapshot["config"])
    require_equal(embedded.snapshot_mapping(), snapshot["config"], "config")
    require_equal(
        embedded.workload_sha256, snapshot["workload_sha256"], "workload_sha256"
    )
    if config is not None:
        if type(config) is not SimProfileConfig:
            raise TypeError("config must be SimProfileConfig")
        if config.workload_sha256 != embedded.workload_sha256:
            raise SimProfileError("the snapshot was made with other workloads")
        if stage not in config.stages:
            raise SimProfileError(f"stage {stage} is not a stage of the config")
    measured = _check_labels(snapshot)
    environment = require_fields(
        snapshot["environment"], _ENVIRONMENT_FIELDS, "environment"
    )
    for key, value in environment.items():
        if type(value) is not str or not value:
            raise _fail(f"environment.{key}", "must be a nonempty string")
    for key in ("load_average_before", "load_average_after"):
        _load_average(snapshot[key], key)
    _check_semantics(snapshot["semantics"])
    workloads = snapshot["workloads"]
    if type(workloads) is not list or len(workloads) != len(WORKLOADS):
        raise _fail("workloads", f"must list the workloads {WORKLOADS}")
    checked = [
        _check_workload(entry, index, expected, embedded)
        for index, (entry, expected) in enumerate(
            zip(workloads, build_workloads(embedded))
        )
    ]
    merged = merged_functions([functions for functions, _ in checked])
    profile = require_fields(
        snapshot["profile"],
        {"calls", "primitive_calls", "total_ns", "functions", "hotspots"},
        "profile",
    )
    require_equal(merged, profile["functions"], "profile.functions")
    for key, value in _totals(merged).items():
        require_equal(value, profile[key], f"profile.{key}")
    hotspots = ranked_hotspots(merged, embedded.top_functions)
    require_equal(hotspots, profile["hotspots"], "profile.hotspots")
    summary = {
        "total_median_ns": sum(median for _, median in checked),
        "top_hotspot": hotspots[0]["function"],
        "top_hotspot_share_ppm": hotspots[0]["share_ppm"],
    }
    require_equal(summary, snapshot["summary"], "summary")
    return {
        "kind": SNAPSHOT_KIND,
        "stage": stage,
        "report_sha256": snapshot[DIGEST_FIELD],
        "measured": measured,
        "scope": snapshot["scope"],
        "function_calls": profile["calls"],
        **summary,
    }
