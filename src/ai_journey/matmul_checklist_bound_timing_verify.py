"""Independent verification of sealed checklist one-call timings, on the CPU alone.

Timings cannot be observed again, so they are checked for structure, and
every number that derives from them is rebuilt: each event sample from its
raw milliseconds, each nearest-rank statistic, the rule that a dispatch
median stays within its synchronized median, the agreement bound of each
output check, the fastest host call of each side, the run totals and the
summary. The point, clock and fastest-call checks are those of the Day 76
timing verifier, and the shared helpers come from the Day 76 and Day 77
verifiers, not from a module that measures. torch is never imported.

The kernel section must name the Day 74 source and the checklist library by
the digests that the registry rebuilds, and each verdict must follow from
the recorded pipeline limits. Each naive pipeline must admit the naive
group, because the run checks every launch against its pipeline. Only a
``time.perf_counter_ns`` report from ``torch.mps.compile_shader`` verifies
as a measurement. A matching digest shows internal consistency; it cannot
prove that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import math
import re

from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_checklist_bound_timing import (
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
    method_mapping,
)
from .matmul_checklist_bound_traffic import (
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
)
from .matmul_checklist_config import MatmulChecklistConfig, config_from_mapping
from .matmul_checklist_variants import (
    LIBRARY_VARIANTS,
    NAIVE_VARIANTS,
    VARIANTS,
    checklist_library,
)
from .matmul_checklist_verify import _admission
from .matmul_metal_source import SHADER_SOURCE
from .matmul_tiled_gap_timing import FLOOR_INDEX, FLOOR_SHAPE
from .matmul_tiled_gap_timing import MatmulTiledGapVerificationError
from .matmul_tiled_gap_timing_verify import _fastest, _point
from .matmul_tiled_verify import (
    MatmulTiledVerificationError,
    _digest,
    _environment,
    _exact,
    _limits,
    _load,
    _switches,
    _text_digest,
    _warnings,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD

_TORCH = "torch_matmul"
_HOST_CLOCKS = ("host_synchronized", "host_dispatch")
_NS = 1_000_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "backend",
    "timer",
    "percentile_method",
    "config",
    "config_sha256",
    "environment",
    "torch_mps_switches",
    "host_load",
    "warnings",
    "kernels",
    "method",
    "floor",
    "points",
    "run",
    "summary",
    DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulChecklistBoundVerificationError(message)


def _labels(report: dict[str, object]) -> tuple[bool, str, str]:
    _require(
        report["timer"] in (TIMER, INJECTED_TIMER)
        and report["backend"] in (BACKEND, INJECTED_BACKEND)
        and report["percentile_method"] == PERCENTILE_METHOD,
        "measurement method fields were changed",
    )
    measured = report["timer"] == TIMER and report["backend"] == BACKEND
    _require(
        report["measured"] is measured,
        "only the host timer and torch.mps.compile_shader make a measurement",
    )
    scope, label, boundary = (
        (MEASUREMENT_SCOPE, MEASUREMENT_LABEL, MEASUREMENT_BOUNDARY)
        if measured
        else (SYNTHETIC_SCOPE, SYNTHETIC_LABEL, SYNTHETIC_BOUNDARY)
    )
    _require(
        report["measurement_scope"] == scope
        and report["label"] == label
        and _exact(report["evidence_boundary"], boundary),
        "evidence labels or boundary were weakened",
    )
    return measured, scope, label


def _kernels(section: object, config: MatmulChecklistConfig) -> list[str]:
    """Check both digests and every verdict; return the admitted kernels in order."""

    limits = section.get("limits") if type(section) is dict else None
    _require(
        type(limits) is dict and set(limits) == set(VARIANTS),
        "kernels must hold the limits of every registered kernel",
    )
    checked = {name: _limits(limits[name]) for name in VARIANTS}
    # Every naive call is a launch checked against its pipeline before it runs.
    group = math.prod(config.naive.group_size)
    for name in NAIVE_VARIANTS:
        _require(
            checked[name]["max_threads_per_threadgroup"] >= group,
            f"the {name} pipeline allows fewer threads than the naive group of "
            f"{group}",
        )
    admissions = [_admission(name, checked[name]) for name in LIBRARY_VARIANTS]
    expected = {
        "naive_source_sha256": _text_digest(SHADER_SOURCE),
        "library_sha256": _text_digest(checklist_library()),
        "limits": checked,
        "admissions": admissions,
    }
    _require(_exact(section, expected), "kernel section does not recompute")
    admitted = {entry["name"] for entry in admissions if entry["admitted"]}
    return [name for name in VARIANTS if name in NAIVE_VARIANTS or name in admitted]


def _verify(
    report: object, supplied: MatmulChecklistConfig | None
) -> dict[str, object]:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    _require(
        type(report["schema_version"]) is int
        and report["schema_version"] == REPORT_SCHEMA_VERSION,
        "unsupported report schema",
    )
    _require(report["kind"] == REPORT_KIND, "unexpected report kind")
    _require(
        report[DIGEST_FIELD] == _digest(report),
        "report digest does not match its content",
    )
    measured, scope, label = _labels(report)
    config = config_from_mapping(report["config"])
    _require(_exact(report["config"], config.to_mapping()), "config is not canonical")
    _require(
        isinstance(report["config_sha256"], str)
        and _SHA256.fullmatch(report["config_sha256"]) is not None
        and report["config_sha256"] == config.fingerprint(),
        "config digest does not match the embedded config",
    )
    if supplied is not None:
        _require(
            supplied.fingerprint() == config.fingerprint(),
            "report was not made from the supplied config",
        )
    _environment(report["environment"], measured)
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    names = _kernels(report["kernels"], config)
    _require(_exact(report["method"], method_mapping()), "method texts were changed")
    sides = [*names, _TORCH]
    floor = _point(report["floor"], FLOOR_INDEX, FLOOR_SHAPE, sides, config)
    shapes = config.benchmark.shapes
    recorded = report["points"]
    _require(
        type(recorded) is list and len(recorded) == len(shapes),
        "timing points do not cover the sweep",
    )
    points = [
        _point(entry, index, (shape.m, shape.n, shape.k), sides, config)
        for index, (entry, shape) in enumerate(zip(recorded, shapes))
    ]
    timed = [floor, *points]
    host = sum(
        sum(side["clocks"][clock]["samples_ns"])
        for point in timed
        for side in point["sides"].values()
        for clock in _HOST_CLOCKS
    )
    run = report["run"]
    limit = config.benchmark.max_runtime_seconds * _NS
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "host_timed_ns"}
        and _exact(run["max_runtime_ns"], limit)
        and _exact(run["host_timed_ns"], host)
        and type(run["elapsed_ns"]) is int
        and host <= run["elapsed_ns"] <= limit,
        "run time does not cover its host samples within the budget",
    )
    load = report["host_load"]
    refused = [name for name in LIBRARY_VARIANTS if name not in names]
    summary = {
        "label": label,
        "points": len(points),
        "sides": sides,
        "refused_variants": refused,
        "timed_samples": sum(
            len(clock["samples_ns"])
            for point in timed
            for side in point["sides"].values()
            for clock in side["clocks"].values()
        ),
        "fastest_host": _fastest(timed, sides),
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "points": len(points),
        "sides": sides,
        "refused_variants": refused,
        "timed_samples": summary["timed_samples"],
        "warnings": summary["warnings"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def verify_timing_report(
    report: object, *, config: MatmulChecklistConfig | None = None
) -> dict[str, object]:
    """Verify a sealed checklist timing report, optionally against its config.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 check, is reported as
    ``MatmulChecklistBoundVerificationError``.
    """

    if config is not None and not isinstance(config, MatmulChecklistConfig):
        raise TypeError("config must be a MatmulChecklistConfig")
    try:
        return _verify(report, config)
    except MatmulChecklistBoundVerificationError:
        raise
    except (MatmulTiledGapVerificationError, MatmulTiledVerificationError) as error:
        raise MatmulChecklistBoundVerificationError(str(error)) from error
    except (
        MatmulChecklistBoundError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulChecklistBoundVerificationError(
            f"report failed verification: {error}"
        ) from error
