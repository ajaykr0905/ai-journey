"""Independent verification of sealed one-call timing reports, on the CPU alone.

Timings cannot be observed again, so they are checked for structure, and
every number that derives from them is rebuilt with this module's own code:
each event sample from its raw milliseconds, each nearest-rank statistic,
the rule that a dispatch median stays within its synchronized median, the
agreement bound of each output check, the run totals and the summary. A
check that claims the naive bits must sit exactly 0 from them, because equal
bits subtract to 0. The kernel section must name the Day 74 source by its
digest and the tiled library that the registry rebuilds for the config's
variants. The naive pipeline must admit the naive group, because the run
checks every launch against its pipeline, and each verdict must follow from
the recorded pipeline limits. The shared helpers come from the tiled
verifier, not from a measuring module, and torch is never imported.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement. A matching digest shows internal consistency; it
cannot prove that a timing was observed or that a GPU ran.

The method texts must be those of the report's own schema_version, 1 or 2.
The versions differ only in the gpu-timing event-domain text, so a version 1
report still verifies, and neither version verifies with the other's.
"""

from __future__ import annotations

import math
import re

from .gpu_timing_clocks import CLOCK_DOMAINS, CLOCKS, MAX_EVENT_MS, NS_PER_MS
from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_source import SHADER_SOURCE
from .matmul_tiled_config import MatmulTiledConfig, config_from_mapping
from .matmul_tiled_gap_timing import (
    FLOOR_INDEX,
    FLOOR_SHAPE,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    SCHEMA_VERSIONS,
    SYNTHETIC_BOUNDARY,
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
    method_mapping,
)
from .matmul_tiled_source import HEADER
from .matmul_tiled_variants import VARIANTS
from .matmul_tiled_verify import (
    _admission,
    _digest,
    _environment,
    _exact,
    _gamma,
    _limits,
    _load,
    _switches,
    _text_digest,
    _timing,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_report import DIGEST_FIELD

_NAIVE, _TORCH = "matmul_x_columns", "torch_matmul"
_EVENT, _SYNCHRONIZED, _DISPATCH = CLOCKS
_U32 = 2.0**-24
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
        raise MatmulTiledGapVerificationError(message)


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


def _kernels(section: object, config: MatmulTiledConfig) -> list[str]:
    """Check both library digests and every verdict; return the admitted names."""

    names = config.variants
    limits = section.get("limits") if type(section) is dict else None
    _require(
        type(limits) is dict and set(limits) == {_NAIVE, *names},
        "kernels must hold the limits of the naive kernel and every variant",
    )
    checked = {name: _limits(limits[name]) for name in (_NAIVE, *names)}
    # Every naive call is a launch checked against this limit before it runs.
    group = math.prod(config.naive.group_size)
    _require(
        checked[_NAIVE]["max_threads_per_threadgroup"] >= group,
        f"the naive pipeline allows fewer threads than the naive group of {group}",
    )
    admissions = [
        _admission(name, VARIANTS[name].tile, checked[name]) for name in names
    ]
    source = HEADER + "".join("\n" + VARIANTS[name].source for name in names)
    expected = {
        "naive_source_sha256": _text_digest(SHADER_SOURCE),
        "tiled_source_sha256": _text_digest(source),
        "limits": checked,
        "admissions": admissions,
    }
    _require(_exact(section, expected), "kernel section does not recompute")
    return [entry["name"] for entry in admissions if entry["admitted"]]


def _clock(entry: object, name: str, repeats: int, label: str) -> dict:
    _require(type(entry) is dict, f"{label} {name} samples must be a JSON object")
    samples = entry.get("samples_ns")
    if name == _EVENT:
        raw = entry.get("elapsed_ms")
        _require(
            type(raw) is list
            and len(raw) == repeats
            and all(
                type(value) is float
                and math.isfinite(value)
                and 0.0 < value <= MAX_EVENT_MS
                for value in raw
            ),
            f"{label} needs {repeats} raw event intervals in (0, {MAX_EVENT_MS}] ms",
        )
        samples = [round(value * NS_PER_MS) for value in raw]
    _require(
        type(entry.get("samples_ns")) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{label} {name} samples_ns must hold {repeats} positive integers",
    )
    expected = {"clock": name, "domain": CLOCK_DOMAINS[name], "samples_ns": samples}
    if name == _EVENT:
        expected["elapsed_ms"] = entry["elapsed_ms"]
    expected["timing"] = _timing(samples)
    _require(_exact(entry, expected), f"{label} {name} does not recompute")
    return expected


def _side(entry: object, name: str, config: MatmulTiledConfig, label: str) -> dict:
    clocks = entry.get("clocks") if type(entry) is dict else None
    _require(
        type(clocks) is dict and set(clocks) == set(CLOCKS),
        f"{label} {name} must hold the samples of the three clocks",
    )
    repeats = config.benchmark.repeats
    rebuilt = {
        clock: _clock(clocks[clock], clock, repeats, f"{label} {name}")
        for clock in CLOCKS
    }
    _require(
        rebuilt[_DISPATCH]["timing"]["median_ns"]
        <= rebuilt[_SYNCHRONIZED]["timing"]["median_ns"],
        f"{label} {name}: the dispatch median exceeds the synchronized median",
    )
    expected = {
        "implementation": name,
        "warmup": config.benchmark.warmup,
        "repeats": repeats,
        "clocks": rebuilt,
    }
    _require(_exact(entry, expected), f"{label} {name} does not recompute")
    return expected


def _checks(entry: object, sides: list[str], bound: float, label: str) -> dict:
    _require(
        type(entry) is dict and set(entry) == set(sides[1:]),
        f"{label} must check every other side against the naive output",
    )
    for name, check in entry.items():
        _require(
            type(check) is dict
            and set(check) == {"same_bits_as_naive", "disagreement"}
            and type(check["same_bits_as_naive"]) is bool
            and type(check["disagreement"]) is float
            and math.isfinite(check["disagreement"])
            and 0.0 <= check["disagreement"] <= bound,
            f"{label} {name} must agree with the naive output within the bound",
        )
        # Equal bits subtract to 0, so a record cannot claim both.
        _require(
            not check["same_bits_as_naive"] or check["disagreement"] == 0.0,
            f"{label} {name} cannot hold the naive bits and sit "
            f"{check['disagreement']!r} from them",
        )
    return entry


def _point(
    entry: object,
    index: int,
    shape: tuple[int, int, int],
    sides: list[str],
    config: MatmulTiledConfig,
) -> dict:
    m, n, k = shape
    label = f"point {index} at {[m, n, k]}"
    _require(type(entry) is dict, f"{label} must be a JSON object")
    recorded = entry.get("sides")
    _require(
        type(recorded) is dict and set(recorded) == set(sides),
        f"{label} must time the naive kernel, the admitted tiles and torch.matmul",
    )
    bound = 4 * _gamma(k, _U32)
    expected = {
        "index": index,
        "m": m,
        "n": n,
        "k": k,
        "flops_per_call": 2 * m * n * k,
        "side_order": sides,
        "sides": {name: _side(recorded[name], name, config, label) for name in sides},
        "checks": _checks(entry.get("checks"), sides, bound, label),
        "agreement_bound": bound,
    }
    _require(_exact(entry, expected), f"{label} does not recompute")
    return expected


def _fastest(points: list[dict], sides: list[str]) -> dict:
    fastest = {}
    for side in sides:
        hosts = [
            point["sides"][side]["clocks"][_SYNCHRONIZED]["timing"]["median_ns"]
            for point in points
        ]
        first = hosts.index(min(hosts))
        top = points[first]
        fastest[side] = {
            "m": top["m"],
            "n": top["n"],
            "k": top["k"],
            "median_host_ns": hosts[first],
        }
    return fastest


def _verify(report: object, supplied: MatmulTiledConfig | None) -> dict[str, object]:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    version = report["schema_version"]
    _require(
        type(version) is int and version in SCHEMA_VERSIONS,
        "report schema_version must be one of "
        + ", ".join(str(known) for known in SCHEMA_VERSIONS),
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
    admitted = _kernels(report["kernels"], config)
    _require(
        _exact(report["method"], method_mapping(version)),
        f"method texts are not those of schema_version {version}",
    )
    sides = [_NAIVE, *admitted, _TORCH]
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
        for clock in (_SYNCHRONIZED, _DISPATCH)
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
    summary = {
        "label": label,
        "points": len(points),
        "sides": sides,
        "timed_samples": sum(
            len(clock["samples_ns"])
            for point in timed
            for side in point["sides"].values()
            for clock in side["clocks"].values()
        ),
        "fastest_host": _fastest(timed, sides),
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
        "timed_samples": summary["timed_samples"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def verify_timing_report(
    report: object, *, config: MatmulTiledConfig | None = None
) -> dict[str, object]:
    """Verify a sealed one-call timing report, optionally against its config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulTiledGapVerificationError``.
    """

    if config is not None and not isinstance(config, MatmulTiledConfig):
        raise TypeError("config must be a MatmulTiledConfig")
    try:
        return _verify(report, config)
    except MatmulTiledGapVerificationError:
        raise
    except (
        MatmulTiledGapError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulTiledGapVerificationError(
            f"report failed verification: {error}"
        ) from error
