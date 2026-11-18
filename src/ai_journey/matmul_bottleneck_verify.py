"""Independent verification of sealed bottleneck reports, on the CPU alone.

A bottleneck report holds the inputs of every decision: the config, the two
ceilings, and for each point the shape, the batch, the Day 74 median sample
times and the one-call medians. The verifier rebuilds every derived number
from those inputs with its own code: the FLOPs and intensities in exact
integers, each share with the same float operations, each name with its own
copy of the ordered rule, and the warp model of both mappings with NumPy
from the kernels' index formulas. Each section must match its rebuild byte
for byte as canonical JSON.

Like the gap report, the embedded inputs cannot prove themselves: a resealed
report with other medians still adds up. ``rebuilt`` closes that gap. It is
a report built again from the source files and the timing input, and the
given report must equal it byte for byte. The summary says whether that
check ran. Nothing here imports torch.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from fractions import Fraction
from hashlib import sha256

import numpy

from .gpu_ceiling_report import REPORT_KIND as CEILING_KIND
from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .matmul_bottleneck_config import (
    BottleneckConfig,
    MatmulBottleneckError,
    MatmulBottleneckVerificationError,
    config_from_mapping,
)
from .matmul_bottleneck_mapping import MODEL_SCOPE, PROBE_WARP, VERDICTS
from .matmul_bottleneck_report import (
    DEFINITIONS,
    GAP_SOURCE,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
    TIMING_SOURCE,
    reasons_mapping,
)
from .matmul_bottleneck_rule import BOTTLENECKS, REASONS
from .matmul_bottleneck_timing import TIMING_KIND
from .matmul_metal_report import REPORT_KIND as MATMUL_KIND
from .matmul_roofline_gap import REPORT_KIND as GAP_KIND
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD, report_digest

_COLUMNS, _ROWS = "matmul_x_columns", "matmul_x_rows"
_KERNELS = {_COLUMNS: "x_columns", _ROWS: "x_rows"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_LANES = 32
_SECTOR, _LINE = 32, 128
_MAX_DIMENSION = 8192
_MAX_POINTS = 64
_MAX_NS = 1 << 40
_CLASSES = ("single", "broadcast", "coalesced", "strided")
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "config",
    "config_sha256",
    "sources",
    "environment",
    "ceilings",
    "geometry",
    "model_scope",
    "definitions",
    "reasons",
    "points",
    "summary",
    DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulBottleneckVerificationError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _exact(actual: object, expected: object) -> bool:
    return _canonical(actual) == _canonical(expected)


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _positive_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0


def _loads(value: object) -> bool:
    return (
        type(value) is list
        and len(value) == 3
        and all(
            type(item) is float and math.isfinite(item) and item >= 0 for item in value
        )
    )


def _decimal(value: Fraction) -> str:
    """Six decimals, half to even, as the model lab writes a positive ratio."""

    scaled = round(value * 1_000_000)
    return f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}"


def _sources(sources: object) -> bool:
    """Check the shape of the source identities; return whether all measured."""

    _require(
        type(sources) is dict
        and set(sources) == {"matmul_metal", "gpu_ceilings", GAP_SOURCE, TIMING_SOURCE},
        "sources must name the three Day 74 reports and the timing input",
    )
    matmul, ceilings = sources["matmul_metal"], sources["gpu_ceilings"]
    gap, timing = sources[GAP_SOURCE], sources[TIMING_SOURCE]
    _require(
        type(matmul) is dict
        and set(matmul) == {"kind", "report_sha256"}
        and matmul["kind"] == MATMUL_KIND,
        "source matmul_metal does not match the schema",
    )
    _require(
        type(ceilings) is dict
        and set(ceilings)
        == {"kind", "report_sha256", "load_average_before", "load_average_after"}
        and ceilings["kind"] == CEILING_KIND
        and _loads(ceilings["load_average_before"])
        and _loads(ceilings["load_average_after"]),
        "source gpu_ceilings does not match the schema",
    )
    _require(
        type(gap) is dict
        and set(gap) == {"kind", "report_sha256", "measured", "label"}
        and gap["kind"] == GAP_KIND
        and type(gap["measured"]) is bool
        and gap["label"] == (MEASUREMENT_LABEL if gap["measured"] else SYNTHETIC_LABEL),
        "source matmul_roofline_gap does not match the schema",
    )
    _require(
        type(timing) is dict
        and set(timing)
        == {
            "kind",
            "input_sha256",
            "measured",
            "label",
            "producer",
            "repeats",
            "load_average_before",
            "load_average_after",
            "entries",
            "entries_used",
        }
        and timing["kind"] == TIMING_KIND
        and type(timing["measured"]) is bool
        and timing["label"]
        == (MEASUREMENT_LABEL if timing["measured"] else SYNTHETIC_LABEL)
        and type(timing["producer"]) is str
        and 1 <= len(timing["producer"]) <= 300
        and _integer(timing["repeats"], 1, 1 << 20)
        and _loads(timing["load_average_before"])
        and _loads(timing["load_average_after"])
        and _integer(timing["entries"], 1, 96)
        and _integer(timing["entries_used"], 1, timing["entries"]),
        "source timing does not match the schema",
    )
    for entry, field in (
        (matmul, "report_sha256"),
        (ceilings, "report_sha256"),
        (gap, "report_sha256"),
        (timing, "input_sha256"),
    ):
        _require(
            type(entry[field]) is str and _SHA256.fullmatch(entry[field]) is not None,
            f"source {field} must be a SHA-256 hex digest",
        )
    return gap["measured"] and timing["measured"]


def _environment(environment: object, measured: bool) -> None:
    _require(
        type(environment) is dict
        and set(environment) >= set(ENVIRONMENT_FIELDS)
        and len(environment) == len(ENVIRONMENT_FIELDS) + 1
        and "torch_float32_matmul_precision" in environment,
        "environment fields do not match the schema",
    )
    _require(
        all(
            type(value) is str and _TOKEN.fullmatch(value) is not None
            for value in environment.values()
        ),
        "environment values must be short tokens",
    )
    _require(
        not measured or environment["platform_system"] == "Darwin",
        "a measurement must come from macOS",
    )


def _ceilings(ceilings: object) -> tuple[float, float, bool]:
    _require(
        type(ceilings) is dict
        and set(ceilings)
        == {"fma_gflops_per_second", "bandwidth_gb_per_second", "fma_scales_with_work"}
        and _positive_float(ceilings["fma_gflops_per_second"])
        and _positive_float(ceilings["bandwidth_gb_per_second"])
        and type(ceilings["fma_scales_with_work"]) is bool,
        "ceilings do not match the schema",
    )
    return (
        ceilings["fma_gflops_per_second"],
        ceilings["bandwidth_gb_per_second"],
        ceilings["fma_scales_with_work"],
    )


def _geometry(geometry: object) -> tuple[int, int]:
    _require(
        type(geometry) is dict and set(geometry) == {"group_size", "simd_widths"},
        "geometry fields do not match the schema",
    )
    group = geometry["group_size"]
    _require(
        type(group) is list
        and len(group) == 2
        and all(_integer(size, 1, 1024) for size in group)
        and group[0] % _LANES == 0
        and group[0] * group[1] <= 1024,
        "the warp model needs a group whose x size is a multiple of 32",
    )
    _require(
        _exact(geometry["simd_widths"], {_COLUMNS: _LANES, _ROWS: _LANES}),
        "the warp model needs a SIMD width of 32 in both pipelines",
    )
    return group[0], group[1]


def _intensity(model: str, m: int, n: int, k: int) -> Fraction:
    flops = 2 * m * n * k
    if model == "no_reuse":
        return Fraction(flops, 4 * (2 * m * n * k + m * n))
    return Fraction(flops, 4 * (m * k + k * n + m * n))


def _rule(
    steady: float,
    rate: float,
    times: tuple[int, int, int],
    fma: float,
    scales: bool,
    roof: float,
    rule: dict,
) -> dict:
    """The ordered bottleneck rule, written again from its documented steps."""

    gpu, host, dispatch = times
    lower = roof if roof < fma else fma
    numbers = {
        "steady_ns_per_call": steady,
        "gpu_ns": float(gpu),
        "host_ns": float(host),
        "dispatch_ns": float(dispatch),
        "overhead_share": 1.0 - steady / host,
        "host_outside_gpu_share": 1.0 - gpu / host,
        "gpu_excess_share": (gpu - steady) / host,
        "dispatch_share": dispatch / host,
        "gpu_to_steady_ratio": gpu / steady,
        "rate_gflops_per_second": rate,
        "fma_ceiling_gflops_per_second": fma,
        "bandwidth_roof_gflops_per_second": roof,
        "fma_ceiling_share": rate / fma,
        "bandwidth_roof_share": rate / roof,
        "lower_roof_share": rate / lower,
    }
    slack = 1.0 + rule["timing_tolerance"]
    excess = 1.0 + rule["roof_excess_tolerance"]
    agreement = rule["gpu_agreement_max_ratio"]
    near = rule["near_roof_share_min"]
    ratio = numbers["gpu_to_steady_ratio"]
    steps = (
        (dispatch > host, "dispatch_time_above_host_time", "dispatch_ns", float(host)),
        (gpu > host * slack, "gpu_time_above_host_time", "gpu_ns", host * slack),
        (
            steady > host * slack,
            "steady_time_above_host_time",
            "steady_ns_per_call",
            host * slack,
        ),
        (
            numbers["overhead_share"] >= rule["dispatch_overhead_share_min"],
            "one_call_overhead_dominates",
            "overhead_share",
            rule["dispatch_overhead_share_min"],
        ),
        (
            ratio > agreement or ratio * agreement < 1.0,
            "gpu_time_disagrees_with_steady_time",
            "gpu_to_steady_ratio",
            agreement,
        ),
        (
            numbers["fma_ceiling_share"] > excess,
            "rate_above_fma_ceiling",
            "fma_ceiling_share",
            excess,
        ),
        (
            numbers["bandwidth_roof_share"] > excess,
            "rate_above_bandwidth_roof",
            "bandwidth_roof_share",
            excess,
        ),
        (
            numbers["lower_roof_share"] < near,
            "below_both_roofs",
            "lower_roof_share",
            near,
        ),
        (roof < fma, "near_bandwidth_roof", "lower_roof_share", near),
        (
            rule["require_fma_scales_with_work"] and not scales,
            "fma_ceiling_does_not_scale_with_work",
            "lower_roof_share",
            near,
        ),
        (True, "near_fma_ceiling", "lower_roof_share", near),
    )
    for applies, reason, quantity, limit in steps:
        if applies:
            break
    return {
        "bottleneck": REASONS[reason][0],
        "reason": reason,
        "explanation": REASONS[reason][1],
        "numbers": numbers,
        "deciding": {"quantity": quantity, "value": numbers[quantity], "limit": limit},
    }


def _requests(addresses: numpy.ndarray) -> dict:
    """Count sectors, lines and classes of each row of lane addresses."""

    count = addresses.shape[1]
    sectors = addresses // _SECTOR
    lines = addresses // _LINE

    def distinct(values: numpy.ndarray) -> numpy.ndarray:
        ordered = numpy.sort(values, axis=1)
        return 1 + numpy.count_nonzero(numpy.diff(ordered, axis=1), axis=1)

    unique = distinct(addresses)
    steps = numpy.diff(addresses, axis=1)
    affine = (steps == steps[:, :1]).all(axis=1) if count > 1 else None
    classes = []
    strides = []
    for row in range(addresses.shape[0]):
        if count == 1:
            classes.append("single")
            strides.append(None)
            continue
        step = int(steps[row, 0]) if affine[row] else None
        strides.append(step)
        if unique[row] == 1:
            classes.append("broadcast")
        elif step == 4:
            classes.append("coalesced")
        else:
            classes.append("strided")
    tally = Counter(classes)
    sector_counts = distinct(sectors)
    line_counts = distinct(lines)
    return {
        "requests": addresses.shape[0],
        "sectors": int(sector_counts.sum()),
        "lines": int(line_counts.sum()),
        "requested_bytes": 4 * count * addresses.shape[0],
        "unique_bytes": 4 * int(unique.sum()),
        "transferred_bytes": _SECTOR * int(sector_counts.sum()),
        "classes": {name: tally.get(name, 0) for name in _CLASSES},
        "first_request": {
            "active_lanes": count,
            "distinct_addresses": int(unique[0]),
            "sectors": int(sector_counts[0]),
            "lines": int(line_counts[0]),
            "class": classes[0],
            "lane_stride_bytes": strides[0],
        },
    }


def _warp(mapping: str, m: int, n: int, k: int) -> dict:
    """Lanes 0 to 31 of the first warp sit at x = lane and y = 0."""

    lanes = numpy.arange(_LANES, dtype=numpy.int64)
    rows, cols = (
        (numpy.zeros_like(lanes), lanes)
        if mapping == "x_columns"
        else (lanes, numpy.zeros_like(lanes))
    )
    active = (rows < m) & (cols < n)
    rows, cols = rows[active], cols[active]
    steps = numpy.arange(k, dtype=numpy.int64)[:, None]
    operands = {
        "a": (rows[None, :] * k + steps) * 4,
        "b": (steps * n + cols[None, :]) * 4,
        "c": ((rows * n + cols) * 4)[None, :],
    }
    counted = {name: _requests(values) for name, values in operands.items()}
    return {
        "mapping": mapping,
        "active_lanes": int(active.sum()),
        "operands": counted,
        "sectors": sum(item["sectors"] for item in counted.values()),
        "lines": sum(item["lines"] for item in counted.values()),
    }


def _mapping(m: int, n: int, k: int, rates: dict, slower_min: float) -> dict:
    warps = {mapping: _warp(mapping, m, n, k) for mapping in ("x_columns", "x_rows")}
    columns, rows = warps["x_columns"], warps["x_rows"]
    more = rows["sectors"] > columns["sectors"]
    ratio = rates[_COLUMNS] / rates[_ROWS]
    slower = ratio >= slower_min
    verdict = VERDICTS[0 if more and slower else 1 if more else 2 if slower else 3]
    return {
        "probe_warp": dict(PROBE_WARP),
        "warps": warps,
        "model_sector_ratio": rows["sectors"] / columns["sectors"],
        "model_line_ratio": rows["lines"] / columns["lines"],
        "model_predicts_x_rows_worse": more,
        "x_columns_to_x_rows_rate_ratio": ratio,
        "timing_shows_x_rows_slower": slower,
        "verdict": verdict,
    }


def _point(
    point: object,
    index: int,
    config: dict,
    ceilings: tuple[float, float, bool],
    group: tuple[int, int],
) -> dict:
    _require(type(point) is dict, f"point {index} must be a JSON object")
    m, n, k = point.get("m"), point.get("n"), point.get("k")
    calls = point.get("calls_per_sample")
    _require(
        all(_integer(value, 1, _MAX_DIMENSION) for value in (m, n, k))
        and _integer(calls, 1, 1 << 20),
        f"point {index} has an invalid shape or batch",
    )
    medians = point.get("median_sample_ns")
    kernels = point.get("kernels")
    _require(
        type(medians) is dict
        and set(medians) == set(_KERNELS)
        and all(_integer(value, 1, _MAX_NS) for value in medians.values()),
        f"point {index} median sample times must be positive integers",
    )
    _require(
        type(kernels) is dict
        and set(kernels) == set(_KERNELS)
        and all(type(entry) is dict for entry in kernels.values()),
        f"point {index} must hold both naive kernels",
    )
    rule = config["rule"]
    model = rule["intensity_model"]
    fma, bandwidth, scales = ceilings
    flops = 2 * m * n * k
    intensity = _intensity(model, m, n, k)
    roof = bandwidth * (intensity.numerator / intensity.denominator)
    rates = {kernel: flops * calls / medians[kernel] for kernel in _KERNELS}
    expected_kernels = {}
    for kernel, mapping in _KERNELS.items():
        inputs = kernels[kernel].get("inputs")
        _require(
            type(inputs) is dict
            and set(inputs) == {"median_gpu_ns", "median_host_ns", "median_dispatch_ns"}
            and all(_integer(value, 1, _MAX_NS) for value in inputs.values()),
            f"point {index} {kernel} timing inputs must be positive integers",
        )
        times = (
            inputs["median_gpu_ns"],
            inputs["median_host_ns"],
            inputs["median_dispatch_ns"],
        )
        expected_kernels[kernel] = {
            "mapping": mapping,
            "inputs": dict(inputs),
            "decision": _rule(
                medians[kernel] / calls, rates[kernel], times, fma, scales, roof, rule
            ),
        }
    block_x, block_y = group
    extents = {_COLUMNS: (n, m), _ROWS: (m, n)}
    expected = {
        "index": index,
        "m": m,
        "n": n,
        "k": k,
        "flops_per_call": flops,
        "calls_per_sample": calls,
        "intensity": {
            "model": model,
            "flops_per_byte": {
                "numerator": intensity.numerator,
                "denominator": intensity.denominator,
                "decimal": _decimal(intensity),
            },
        },
        "threadgroups": {
            kernel: [-(-x // block_x), -(-y // block_y)]
            for kernel, (x, y) in extents.items()
        },
        "median_sample_ns": dict(medians),
        "kernels": expected_kernels,
        "mapping": _mapping(m, n, k, rates, config["mapping"]["slower_ratio_min"]),
    }
    _require(_exact(point, expected), f"point {index} does not recompute")
    return expected


def _top(names: list[str]) -> dict:
    counts = Counter(names)
    most = max(counts.values())
    leaders = [name for name in BOTTLENECKS if counts.get(name, 0) == most]
    return {
        "counts": {name: counts.get(name, 0) for name in BOTTLENECKS},
        "top": leaders[0] if len(leaders) == 1 else "indeterminate",
        "top_points": most,
        "reason": "most_points" if len(leaders) == 1 else "tie",
    }


def _summary(label: str, points: list[dict]) -> dict:
    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            kernel: _top(
                [point["kernels"][kernel]["decision"]["bottleneck"] for point in points]
            )
            for kernel in _KERNELS
        },
        "mapping_verdicts": {
            verdict: len(
                [point for point in points if point["mapping"]["verdict"] == verdict]
            )
            for verdict in VERDICTS
        },
    }


def _verify(
    report: object, config: BottleneckConfig | None, rebuilt: object
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
        report[DIGEST_FIELD] == report_digest(report),
        "report digest does not match its content",
    )
    try:
        embedded = config_from_mapping(report["config"])
    except MatmulBottleneckError as error:
        raise MatmulBottleneckVerificationError(
            f"embedded config is invalid: {error}"
        ) from error
    _require(
        report["config_sha256"]
        == sha256(canonical_json_bytes(report["config"])).hexdigest(),
        "config digest does not match the embedded config",
    )
    _require(
        config is None or embedded == config,
        "report was not made from the supplied config",
    )
    measured = _sources(report["sources"])
    _require(
        report["measured"] is measured,
        "a report is a measurement only when the gap and the timing input are",
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
    _environment(report["environment"], measured)
    ceilings = _ceilings(report["ceilings"])
    group = _geometry(report["geometry"])
    _require(report["model_scope"] == MODEL_SCOPE, "model scope was changed")
    _require(_exact(report["definitions"], DEFINITIONS), "definitions were changed")
    _require(_exact(report["reasons"], reasons_mapping()), "reasons were changed")
    points = report["points"]
    _require(
        type(points) is list and 1 <= len(points) <= _MAX_POINTS,
        f"a report holds 1 to {_MAX_POINTS} points",
    )
    expected = [
        _point(point, index, report["config"], ceilings, group)
        for index, point in enumerate(points)
    ]
    _require(
        report["sources"][TIMING_SOURCE]["entries_used"] == 2 * len(points),
        "the timing input must have given one entry per naive kernel and point",
    )
    _require(
        _exact(report["summary"], _summary(label, expected)),
        "summary does not reconcile",
    )
    if rebuilt is not None:
        _require(type(rebuilt) is dict, "a rebuilt report must be a JSON object")
        _require(
            _exact(report["sources"], rebuilt.get("sources")),
            "report was not made from the supplied sources and timing input",
        )
        _require(
            canonical_json_bytes(report) == canonical_json_bytes(rebuilt),
            "report does not match the report rebuilt from its sources",
        )
    summary = report["summary"]["top_bottleneck"]
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "points": len(points),
        "top_bottleneck": {kernel: summary[kernel]["top"] for kernel in _KERNELS},
        "sources_checked": rebuilt is not None,
    }


def verify_report(
    report: object,
    *,
    config: BottleneckConfig | None = None,
    rebuilt: object | None = None,
) -> dict[str, object]:
    """Verify a sealed report; bind it to a config and to its rebuilt sources.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulBottleneckVerificationError``.
    """

    try:
        return _verify(report, config, rebuilt)
    except MatmulBottleneckVerificationError:
        raise
    except (
        MatmulBottleneckError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulBottleneckVerificationError(
            f"report failed verification: {error}"
        ) from error
