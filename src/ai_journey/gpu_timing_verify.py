"""Independent verification of sealed three-clock timing reports, on the CPU.

Timings cannot be observed again, so the verifier checks their structure and
recomputes everything derived from them with its own code:

- Every side holds ``repeats`` positive integer samples on each of the three
  clocks, with the warmup and repeats of the config. Event samples carry raw
  float ms values in ``(0, 60000]``, and each ``samples_ns`` must equal
  ``round(ms * 1e6)``. Each clock names its own domain.
- Every nearest-rank statistic is recomputed from the raw samples. A side
  whose dispatch median exceeds its synchronized median is rejected, because
  a dispatch sample covers a strict part of a synchronized one.
- Every fit, split, ratio, rate, fraction, cross-domain record and summary
  entry is rebuilt from the medians with a separate implementation of the
  weighted least squares, whose ``math.fsum`` sums give the same bits.
- Launch geometry, shapes, planned work, the shader digests, the clock
  rules, the load averages, the environment tokens and the labels are
  checked too, and run totals must recompute.

Every rebuilt section must match the stored one byte for byte as canonical
JSON, so ``1``, ``1.0`` and ``true`` stay distinct. No measuring module is
called, and torch is never imported, so a host without torch or MPS can
verify a report. Only a ``time.perf_counter_ns`` report from
``torch.mps.compile_shader`` verifies as a measurement. A matching digest
shows internal consistency; it cannot prove that a timing was observed.
"""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE
from .gpu_timing_analysis import FRACTION_RULE, INFERENCE_BASIS
from .gpu_timing_benchmark import SCHEDULE
from .gpu_timing_clocks import (
    CLOCK_DEFINITIONS,
    CROSS_DOMAIN_MEANING,
    DOMAIN_DEFINITIONS,
    GpuTimingError,
)
from .gpu_timing_config import GpuTimingConfig, config_from_mapping
from .gpu_timing_fit import FIT_METHOD, REGIME_RULE, SPLIT_RULE
from .gpu_timing_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
)
from .gpu_timing_source import EMPTY_SOURCE
from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_report import MATMUL_PRECISION_FIELD, TORCH_EQUIVALENT
from .matmul_metal_source import SHADER_SOURCE
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_MAX_EVENT_MS = 60_000.0
_U32 = 2.0**-24
_EVENT, _SYNC, _DISPATCH = "mps_event", "host_synchronized", "host_dispatch"
_CLOCKS = (_EVENT, _SYNC, _DISPATCH)
_DOMAINS = {
    _EVENT: "mps_event_stream",
    _SYNC: "host_perf_counter_ns",
    _DISPATCH: "host_perf_counter_ns",
}
_COLUMNS, _ROWS, _TORCH = "matmul_x_columns", "matmul_x_rows", "torch_matmul"
_SIDES = (_COLUMNS, _ROWS, _TORCH)
_MAPPINGS = {"x_columns": _COLUMNS, "x_rows": _ROWS}
_EMPTY = "empty_kernel"
_FIXED, _SIZE = "fixed_overhead_dominant", "size_dependent_dominant"
_AGREEMENT = "4 * gamma_K(2**-24) * (|A| @ |B|), on the device in float32"
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
    "host_load",
    "shaders",
    "clocks",
    "planned_work",
    "benchmark",
    "analysis",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_LIMIT_FIELDS = {
    "max_threads_per_threadgroup",
    "thread_execution_width",
    "static_threadgroup_memory_bytes",
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GpuTimingVerificationError(GpuTimingError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuTimingVerificationError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _exact(actual: object, expected: object, label: str) -> None:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 stay distinct."""

    try:
        same = _canonical(actual) == _canonical(expected)
    except (TypeError, ValueError) as error:
        raise GpuTimingVerificationError(f"{label} is not finite JSON") from error
    _require(same, f"{label} does not match its recomputation")


def _digest(report: dict[str, object]) -> str:
    unsigned = {
        key: value for key, value in report.items() if key != REPORT_DIGEST_FIELD
    }
    return sha256(_canonical(unsigned)).hexdigest()


def _rank(ordered: list[int], percentile: int) -> int:
    return ordered[(percentile * len(ordered) + 99) // 100 - 1]


def _statistics(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)
    p25, p75 = _rank(ordered, 25), _rank(ordered, 75)
    return {
        "count": len(ordered),
        "min_ns": ordered[0],
        "p25_ns": p25,
        "median_ns": _rank(ordered, 50),
        "p75_ns": p75,
        "max_ns": ordered[-1],
        "iqr_ns": p75 - p25,
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _geometry(extent: list[int], group: list[int], width: int) -> dict[str, object]:
    groups = [_ceil_div(size, size_group) for size, size_group in zip(extent, group)]
    threads = [count * size for count, size in zip(groups, group)]
    per_group = math.prod(group)
    simd = _ceil_div(per_group, width)
    return {
        "extent": extent,
        "group_size": group,
        "groups": groups,
        "threads": threads,
        "threads_per_group": per_group,
        "launched_threads": math.prod(threads),
        "active_threads": math.prod(extent),
        "padding_threads": math.prod(threads) - math.prod(extent),
        "simd_groups_per_threadgroup": simd,
        "idle_lanes_per_threadgroup": simd * width - per_group,
    }


def _limits(shaders: dict[str, object], group_threads: int) -> dict[str, int]:
    limits = shaders["limits"]
    _require(
        type(limits) is dict and set(limits) == {_COLUMNS, _ROWS, _EMPTY},
        "shader limits must cover both matmul kernels and the empty kernel",
    )
    widths = {}
    for name, entry in limits.items():
        _require(
            type(entry) is dict
            and set(entry) == _LIMIT_FIELDS
            and all(type(value) is int for value in entry.values()),
            f"{name} limits must be three integers",
        )
        _require(
            1
            <= entry["thread_execution_width"]
            <= entry["max_threads_per_threadgroup"]
            <= 1024
            and 0 <= entry["static_threadgroup_memory_bytes"] <= 1 << 20,
            f"{name} limits are out of range",
        )
        needed = 1 if name == _EMPTY else group_threads
        _require(
            entry["max_threads_per_threadgroup"] >= needed,
            f"{name} cannot hold its threadgroup",
        )
        widths[name] = entry["thread_execution_width"]
    return widths


def _clock_entry(
    entry: object, clock: str, repeats: int, label: str
) -> tuple[dict[str, object], int]:
    """Check one clock's raw samples; return the rebuilt entry and its median."""

    _require(type(entry) is dict, f"{label} must be an object")
    samples = entry.get("samples_ns")
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(value) is int and value > 0 for value in samples),
        f"{label} needs {repeats} positive integer samples",
    )
    rebuilt: dict[str, object] = {
        "clock": clock,
        "domain": _DOMAINS[clock],
        "samples_ns": samples,
    }
    if clock == _EVENT:
        raw = entry.get("elapsed_ms")
        _require(
            type(raw) is list
            and len(raw) == repeats
            and all(
                type(value) is float
                and math.isfinite(value)
                and 0.0 < value <= _MAX_EVENT_MS
                for value in raw
            ),
            f"{label} needs {repeats} raw float ms values in (0, 60000]",
        )
        _require(
            [round(value * 1_000_000) for value in raw] == samples,
            f"{label} samples_ns are not round(ms * 1e6)",
        )
        rebuilt["elapsed_ms"] = raw
    rebuilt["timing"] = _statistics(samples)
    _exact(entry, rebuilt, label)
    return rebuilt, rebuilt["timing"]["median_ns"]


def _side(
    entry: object, name: str, warmup: int, repeats: int, label: str
) -> dict[str, int]:
    """Check one side's three clocks; return its median on each clock."""

    _require(type(entry) is dict, f"{label} must be an object")
    clocks = entry.get("clocks")
    _require(
        type(clocks) is dict and set(clocks) == set(_CLOCKS),
        f"{label} must hold {', '.join(_CLOCKS)}",
    )
    medians = {}
    rebuilt = {}
    for clock in _CLOCKS:
        rebuilt[clock], medians[clock] = _clock_entry(
            clocks[clock], clock, repeats, f"{label} {clock}"
        )
    _exact(
        entry,
        {
            "implementation": name,
            "warmup": warmup,
            "repeats": repeats,
            "clocks": rebuilt,
        },
        label,
    )
    _require(
        medians[_DISPATCH] <= medians[_SYNC],
        f"{label}: the dispatch median exceeds the synchronized median",
    )
    return medians


def _fit(flops: list[int], medians: list[int]) -> tuple[float, float, float]:
    weights = [1.0 / (median * median) for median in medians]
    total = math.fsum(weights)
    sum_x = math.fsum(w * x for w, x in zip(weights, flops))
    sum_y = math.fsum(w * y for w, y in zip(weights, medians))
    sum_xx = math.fsum(w * x * x for w, x in zip(weights, flops))
    sum_xy = math.fsum(w * x * y for w, x, y in zip(weights, flops, medians))
    determinant = total * sum_xx - sum_x * sum_x
    _require(determinant > 0.0, "a stored fit is singular")
    slope = (total * sum_xy - sum_x * sum_y) / determinant
    intercept = (sum_y - slope * sum_x) / total
    residual = max(abs(y - (intercept + slope * x)) / y for x, y in zip(flops, medians))
    return intercept, slope, residual


def _cross(medians: dict[str, int]) -> dict[str, object]:
    return {
        "value_ns": medians[_SYNC] - medians[_EVENT],
        "cross_domain": True,
        "minuend": _SYNC,
        "subtrahend": _EVENT,
    }


def _analysis(
    config: GpuTimingConfig,
    floor: dict[str, int],
    medians: list[dict[str, dict[str, int]]],
) -> dict[str, object]:
    shapes = config.sweep.shapes
    ceiling = config.reference.fma_ceiling_gflops_per_second
    flops = [shape.flops for shape in shapes]
    fits = {
        side: {
            clock: _fit(flops, [point[side][clock] for point in medians])
            for clock in _CLOCKS
        }
        for side in _SIDES
    }
    points = []
    for index, (shape, point) in enumerate(zip(shapes, medians)):
        sides = {}
        for side in _SIDES:
            clocks = {}
            for clock in _CLOCKS:
                median = point[side][clock]
                fixed = fits[side][clock][0]
                share = fixed / median
                clocks[clock] = {
                    "median_ns": median,
                    "fixed_ns": fixed,
                    "size_ns": median - fixed,
                    "fixed_share": share,
                    "regime": _FIXED if share >= 0.5 else _SIZE,
                    "iqr_ns": point[side][f"{clock}_iqr"],
                }
            rate = shape.flops / point[side][_EVENT]
            sides[side] = {
                "clocks": clocks,
                "dispatch_to_synchronized_ratio": point[side][_DISPATCH]
                / point[side][_SYNC],
                "host_minus_event": _cross(point[side]),
                "event_gflops_per_second": rate,
                "event_fraction_of_fma_ceiling": rate / ceiling,
            }
        points.append(
            {
                "index": index,
                "m": shape.m,
                "n": shape.n,
                "k": shape.k,
                "flops_per_call": shape.flops,
                "sides": sides,
            }
        )
    smallest = shapes[0]
    return {
        "fit_method": FIT_METHOD,
        "split_rule": SPLIT_RULE,
        "regime_rule": REGIME_RULE,
        "cross_domain_meaning": CROSS_DOMAIN_MEANING,
        "inference_basis": INFERENCE_BASIS,
        "fraction_rule": FRACTION_RULE,
        "fma_ceiling_gflops_per_second": ceiling,
        "overhead_estimate": {
            "empty_kernel_median_ns": {clock: floor[clock] for clock in _CLOCKS},
            "empty_kernel_dispatch_to_synchronized_ratio": floor[_DISPATCH]
            / floor[_SYNC],
            "empty_kernel_host_minus_event": _cross(floor),
            "smallest_shape": {
                "index": 0,
                "m": smallest.m,
                "n": smallest.n,
                "k": smallest.k,
                "median_ns": {
                    side: {clock: medians[0][side][clock] for clock in _CLOCKS}
                    for side in _SIDES
                },
            },
        },
        "fits": {
            side: {
                clock: {
                    "intercept_ns": fit[0],
                    "slope_ns_per_flop": fit[1],
                    "points": len(shapes),
                    "max_relative_residual": fit[2],
                    "intercept_to_empty_kernel_ratio": fit[0] / floor[clock],
                }
                for clock, fit in by_clock.items()
            }
            for side, by_clock in fits.items()
        },
        "points": points,
    }


def _summary(
    label: str, analysis: dict[str, object], timed: int, load: dict[str, object]
) -> dict[str, object]:
    points = analysis["points"]
    largest = points[-1]
    sides = {}
    for side in _SIDES:
        crossover = next(
            (
                point["index"]
                for point in points
                if point["sides"][side]["clocks"][_EVENT]["regime"] == _SIZE
            ),
            None,
        )
        entry = largest["sides"][side]
        sides[side] = {
            "intercept_ns": {
                clock: analysis["fits"][side][clock]["intercept_ns"]
                for clock in _CLOCKS
            },
            "smallest_shape_median_ns": analysis["overhead_estimate"]["smallest_shape"][
                "median_ns"
            ][side],
            "first_size_dominant_index": crossover,
            "largest_event_regime": entry["clocks"][_EVENT]["regime"],
            "largest_event_gflops_per_second": entry["event_gflops_per_second"],
            "largest_event_fraction_of_fma_ceiling": entry[
                "event_fraction_of_fma_ceiling"
            ],
            "largest_dispatch_to_synchronized_ratio": entry[
                "dispatch_to_synchronized_ratio"
            ],
        }
    return {
        "label": label,
        "sweep_points": len(points),
        "timed_samples": timed,
        "empty_kernel_median_ns": analysis["overhead_estimate"][
            "empty_kernel_median_ns"
        ],
        "largest_point": {key: largest[key] for key in ("index", "m", "n", "k")},
        "sides": sides,
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


def _planned(config: GpuTimingConfig) -> dict[str, int]:
    sweep = config.sweep
    per_side = 3 * (sweep.warmup + sweep.repeats)
    sides = 1 + 3 * len(sweep.shapes)
    return {
        "sweep_points": len(sweep.shapes),
        "sides": sides,
        "timed_samples": sides * 3 * sweep.repeats,
        "launches": sides * per_side,
        "flops": sum(3 * per_side * shape.flops for shape in sweep.shapes),
    }


def _labels(report: dict[str, object]) -> bool:
    measured = report["measured"]
    _require(type(measured) is bool, "measured must be a boolean")
    _require(report["timer"] in (TIMER, INJECTED_TIMER), "timer is not known")
    _require(report["backend"] in (BACKEND, INJECTED_BACKEND), "backend is not known")
    real = report["timer"] == TIMER and report["backend"] == BACKEND
    _require(measured == real, "measured must hold exactly for the real backend")
    _exact(
        {
            "measurement_scope": report["measurement_scope"],
            "label": report["label"],
            "evidence_boundary": report["evidence_boundary"],
            "percentile_method": report["percentile_method"],
        },
        {
            "measurement_scope": MEASUREMENT_SCOPE if measured else SYNTHETIC_SCOPE,
            "label": MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL,
            "evidence_boundary": MEASUREMENT_BOUNDARY
            if measured
            else SYNTHETIC_BOUNDARY,
            "percentile_method": PERCENTILE_METHOD,
        },
        "labels",
    )
    return measured


def _environment(environment: object) -> None:
    fields = (*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD)
    _require(
        type(environment) is dict
        and set(environment) == set(fields)
        and all(
            isinstance(value, str) and _TOKEN.fullmatch(value)
            for value in environment.values()
        ),
        "environment must hold only short version and platform tokens",
    )


def _load(load: object) -> None:
    def triple(values: object) -> bool:
        return (
            type(values) is list
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0.0
                for value in values
            )
        )

    _require(
        type(load) is dict
        and set(load) == {"source", "before", "after"}
        and load["source"] == LOAD_AVERAGE_SOURCE
        and triple(load["before"])
        and triple(load["after"]),
        "host_load must hold the source and two triples of load averages",
    )


def verify_report(
    report: object, *, config: GpuTimingConfig | None = None
) -> dict[str, object]:
    """Recompute a sealed report; return its summary or raise on any mismatch."""

    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    digest = report[REPORT_DIGEST_FIELD]
    _require(
        isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
        "report_sha256 must be 64 lowercase hex digits",
    )
    _require(_digest(report) == digest, "report digest does not match its content")
    _require(
        report["schema_version"] == 1 and type(report["schema_version"]) is int,
        "report schema_version must be 1",
    )
    _require(report["kind"] == REPORT_KIND, f"report kind must be {REPORT_KIND}")
    try:
        parsed = config_from_mapping(report["config"])
    except ValueError as error:
        raise GpuTimingVerificationError(f"embedded config: {error}") from error
    _exact(report["config"], parsed.to_mapping(), "config")
    _require(
        report["config_sha256"] == sha256(_canonical(parsed.to_mapping())).hexdigest(),
        "config_sha256 does not match the embedded config",
    )
    if config is not None:
        _require(
            isinstance(config, GpuTimingConfig)
            and config.to_mapping() == parsed.to_mapping(),
            "report was not made from the given config",
        )
    measured = _labels(report)
    _environment(report["environment"])
    _load(report["host_load"])
    sweep = parsed.sweep
    group = list(parsed.launch.group_size)
    shaders = report["shaders"]
    _require(
        type(shaders) is dict
        and set(shaders)
        == {"matmul_source_sha256", "empty_source", "empty_source_sha256", "limits"},
        "shaders fields do not match the schema",
    )
    _require(
        shaders["matmul_source_sha256"]
        == sha256(SHADER_SOURCE.encode("ascii")).hexdigest(),
        "the matmul source digest is not the Day 74 source",
    )
    _require(
        shaders["empty_source"] == EMPTY_SOURCE
        and shaders["empty_source_sha256"]
        == sha256(EMPTY_SOURCE.encode("ascii")).hexdigest(),
        "the empty kernel source or its digest changed",
    )
    widths = _limits(shaders, math.prod(group))
    _exact(
        report["clocks"],
        {
            "order": list(_CLOCKS),
            "definitions": CLOCK_DEFINITIONS,
            "domains": _DOMAINS,
            "domain_definitions": DOMAIN_DEFINITIONS,
            "schedule": SCHEDULE,
        },
        "clocks",
    )
    _exact(report["planned_work"], _planned(parsed), "planned_work")

    benchmark = report["benchmark"]
    _require(
        type(benchmark) is dict
        and set(benchmark)
        == {"device", "torch_equivalent", "agreement", "empty", "points"},
        "benchmark fields do not match the schema",
    )
    _exact(
        [benchmark["device"], benchmark["torch_equivalent"], benchmark["agreement"]],
        ["mps", TORCH_EQUIVALENT, _AGREEMENT],
        "benchmark text",
    )
    empty = benchmark["empty"]
    _require(
        type(empty) is dict and set(empty) == {"kernel", "launch", "timing"},
        "the empty point fields do not match the schema",
    )
    one = _geometry([1], [1], widths[_EMPTY])
    plain = {
        key: value
        for key, value in one.items()
        if key not in ("simd_groups_per_threadgroup", "idle_lanes_per_threadgroup")
    }
    _exact(
        empty["kernel"],
        {
            "name": _EMPTY,
            "buffers": ["sink"],
            "never_position": 0xFFFFFFFF,
            "launch": plain,
        },
        "empty kernel",
    )
    _exact(empty["launch"], one, "empty launch")
    floor = _side(empty["timing"], _EMPTY, sweep.warmup, sweep.repeats, "empty")

    points = benchmark["points"]
    _require(
        type(points) is list and len(points) == len(sweep.shapes),
        "benchmark points must follow the sweep shapes",
    )
    medians = []
    for index, (point, shape) in enumerate(zip(points, sweep.shapes)):
        label = f"point {index}"
        _require(type(point) is dict, f"{label} must be an object")
        sides = point.get("sides")
        _require(
            type(sides) is dict and set(sides) == set(_SIDES),
            f"{label} must time {', '.join(_SIDES)}",
        )
        by_side = {}
        for side in _SIDES:
            values = _side(
                sides[side], side, sweep.warmup, sweep.repeats, f"{label} {side}"
            )
            for clock in _CLOCKS:
                timing = sides[side]["clocks"][clock]["timing"]
                values[f"{clock}_iqr"] = timing["iqr_ns"]
            by_side[side] = values
        medians.append(by_side)
        disagreement = point.get("torch_disagreement")
        product = shape.k * _U32
        bound = 4 * (product / (1 - product))
        _require(
            type(disagreement) is float
            and math.isfinite(disagreement)
            and 0.0 <= disagreement <= bound,
            f"{label}: torch_disagreement is outside the agreement bound",
        )
        extents = {"x_columns": [shape.n, shape.m], "x_rows": [shape.m, shape.n]}
        _exact(
            point,
            {
                "index": index,
                "m": shape.m,
                "n": shape.n,
                "k": shape.k,
                "flops_per_call": shape.flops,
                "launches": {
                    mapping: _geometry(
                        extents[mapping], group, widths[_MAPPINGS[mapping]]
                    )
                    for mapping in _MAPPINGS
                },
                "sides": sides,
                "torch_disagreement": disagreement,
                "agreement_bound": bound,
            },
            label,
        )

    analysis = _analysis(parsed, floor, medians)
    _exact(report["analysis"], analysis, "analysis")
    run = report["run"]
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "timed_host_ns"},
        "run fields do not match the schema",
    )
    timings = [empty["timing"]] + [
        point["sides"][side] for point in points for side in _SIDES
    ]
    timed = sum(
        sum(timing["clocks"][clock]["samples_ns"])
        for timing in timings
        for clock in (_SYNC, _DISPATCH)
    )
    limit = sweep.max_runtime_seconds * _NS
    _exact([run["max_runtime_ns"], run["timed_host_ns"]], [limit, timed], "run totals")
    elapsed = run["elapsed_ns"]
    _require(
        type(elapsed) is int and timed <= elapsed <= limit,
        "run elapsed_ns must cover the timed samples and stay inside the budget",
    )
    timed_samples = len(timings) * 3 * sweep.repeats
    _exact(
        report["summary"],
        _summary(report["label"], analysis, timed_samples, report["host_load"]),
        "summary",
    )
    return {
        "report_sha256": digest,
        "measured": measured,
        "scope": report["measurement_scope"],
        "label": report["label"],
        "sweep_points": len(points),
        "timed_samples": timed_samples,
        "load_average_before": report["host_load"]["before"],
        "load_average_after": report["host_load"]["after"],
    }
