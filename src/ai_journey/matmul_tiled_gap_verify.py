"""Independent verification of sealed tiled gap reports, on the CPU alone.

A gap report holds its inputs: the shape and batch of each point, the median
sample time and rate of each side from the tiled report, the three one-call
medians of each side, the fastest host call of each side, the two ceilings
and their flag, and the full rule config. The verifier rebuilds every
derived number from those inputs with its own code: the FLOPs and the bytes
of each traffic model in exact integers, each intensity, roof, attainable
rate, share and bound with the same float operations, and each name with
the Day 75 verifier's own copy of the ordered rule. Each section must match
its rebuild byte for byte as canonical JSON, so ``1``, ``1.0`` and ``true``
stay distinct.

The embedded inputs cannot prove themselves: a resealed report with other
medians still adds up. ``rebuilt`` closes that gap. It is the gap built
again from the three source files, whose own verifiers recompute them, and
the given report must equal it byte for byte. The summary says whether that
check ran. Nothing here imports torch.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from .gpu_ceiling_report import REPORT_KIND as CEILING_KIND
from .kernel_metal_benchmark import MAX_CALLS_PER_SAMPLE, MAX_POINTS
from .kernel_metal_report import (
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .matmul_bottleneck_config import BottleneckConfig, config_from_mapping
from .matmul_bottleneck_report import reasons_mapping
from .matmul_bottleneck_verify import _rule, _top
from .matmul_metal_mapping import MAX_DIMENSION
from .matmul_tiled_gap import (
    CEILING_SOURCE,
    DEFINITIONS,
    GAP_ENVIRONMENT_FIELDS,
    MEASUREMENT_BOUNDARY,
    NO_REUSE,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    RULE_MODEL,
    SYNTHETIC_BOUNDARY,
    TILED_MODEL,
    TILED_SOURCE,
    TIMING_SOURCE,
)
from .matmul_tiled_gap_timing import REPORT_KIND as TIMING_KIND
from .matmul_tiled_gap_timing import (
    MatmulTiledGapError,
    MatmulTiledGapVerificationError,
)
from .matmul_tiled_report import REPORT_KIND as TILED_KIND
from .matmul_tiled_variants import VARIANTS
from .matmul_tiled_verify import _exact
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD, report_digest

_NAIVE, _TORCH = "matmul_x_columns", "torch_matmul"
_SWITCHES = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")
_TIMES = ("median_gpu_ns", "median_host_ns", "median_dispatch_ns")
_MAX_NS = 1 << 40
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_KINDS = {
    TILED_SOURCE: TILED_KIND,
    CEILING_SOURCE: CEILING_KIND,
    TIMING_SOURCE: TIMING_KIND,
}
_IDENTITY = {
    "kind",
    "report_sha256",
    "config_sha256",
    "measured",
    "label",
    "load_average_before",
    "load_average_after",
}
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "sources",
    "rule",
    "environment",
    "torch_mps_switches",
    "ceilings",
    "sides",
    "fastest_calls",
    "definitions",
    "reasons",
    "points",
    "summary",
    DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulTiledGapVerificationError(message)


def _positive_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _decimal(value: Fraction) -> str:
    """Six decimals, half to even, as the model lab writes a positive ratio."""

    scaled = round(value * 1_000_000)
    return f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}"


def _ratio(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": _decimal(value),
    }


def _source(entry: object, name: str) -> bool:
    _require(
        type(entry) is dict and set(entry) == _IDENTITY,
        f"source {name} fields do not match the schema",
    )
    _require(entry["kind"] == _KINDS[name], f"source {name} has another kind")
    for field in ("report_sha256", "config_sha256"):
        _require(
            type(entry[field]) is str and _SHA256.fullmatch(entry[field]) is not None,
            f"source {name} {field} must be a SHA-256 hex digest",
        )
    measured = entry["measured"]
    _require(type(measured) is bool, f"source {name} measured must be a boolean")
    _require(
        entry["label"] == (MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL),
        f"source {name} label does not match whether it measured",
    )
    for field in ("load_average_before", "load_average_after"):
        values = entry[field]
        _require(
            type(values) is list
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0
                for value in values
            ),
            f"source {name} {field} must be three nonnegative floats",
        )
    return measured


def _sources(sources: object) -> bool:
    _require(
        type(sources) is dict and set(sources) == set(_KINDS),
        "sources must name one tiled, one ceiling and one timing report",
    )
    measured = [_source(sources[name], name) for name in _KINDS]
    _require(
        sources[TILED_SOURCE]["config_sha256"]
        == sources[TIMING_SOURCE]["config_sha256"],
        "the tiled and timing sources name other configs",
    )
    return all(measured)


def _environment(environment: object, measured: bool) -> None:
    _require(
        type(environment) is dict and set(environment) == set(GAP_ENVIRONMENT_FIELDS),
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


def _switches(values: object) -> None:
    _require(
        type(values) is dict
        and set(values) == set(_SWITCHES)
        and all(
            value is None
            or (type(value) is str and _TOKEN.fullmatch(value) is not None)
            for value in values.values()
        ),
        "the MPS switches must be tokens or null",
    )


def _rule_config(section: object, supplied: BottleneckConfig | None) -> dict:
    _require(
        type(section) is dict and set(section) == {"config", "config_sha256"},
        "rule must hold the config and its digest",
    )
    config = config_from_mapping(section["config"])
    _require(_exact(section["config"], config.to_mapping()), "rule config changed")
    _require(
        section["config_sha256"] == config.fingerprint(),
        "rule config digest does not match the embedded config",
    )
    _require(
        config.rule.intensity_model == RULE_MODEL,
        "the rule must use the ideal_reuse roof",
    )
    if supplied is not None:
        _require(
            supplied.fingerprint() == config.fingerprint(),
            "report was not made with the supplied rule config",
        )
    return section["config"]["rule"]


def _ceilings(ceilings: object) -> tuple[float, float, bool]:
    _require(
        type(ceilings) is dict
        and set(ceilings)
        == {
            "fma_gflops_per_second",
            "bandwidth_gb_per_second",
            "ridge_flops_per_byte",
            "fma_scales_with_work",
        },
        "ceilings fields do not match the schema",
    )
    fma, bandwidth = (
        ceilings["fma_gflops_per_second"],
        ceilings["bandwidth_gb_per_second"],
    )
    scales = ceilings["fma_scales_with_work"]
    _require(
        _positive_float(fma) and _positive_float(bandwidth) and type(scales) is bool,
        "each ceiling must be a positive finite float, with a boolean flag",
    )
    _require(
        _exact(ceilings["ridge_flops_per_byte"], fma / bandwidth),
        "ceilings section does not recompute",
    )
    return fma, bandwidth, scales


def _sides(sides: object) -> list[str]:
    tiles = sides[1:-1] if type(sides) is list else None
    _require(
        type(sides) is list
        and len(sides) >= 2
        and sides[0] == _NAIVE
        and sides[-1] == _TORCH
        and all(type(name) is str and name in VARIANTS for name in tiles)
        and len(set(tiles)) == len(tiles),
        "sides must be the naive kernel, registered tiles and torch.matmul",
    )
    return sides


def _fastest(section: object, sides: list[str]) -> dict[str, int]:
    _require(
        type(section) is dict and set(section) == set(sides),
        "fastest_calls must name every side",
    )
    hosts = {}
    for side in sides:
        entry = section[side]
        _require(
            type(entry) is dict
            and set(entry) == {"m", "n", "k", "median_host_ns"}
            and all(_integer(entry[axis], 1, MAX_DIMENSION) for axis in "mnk")
            and _integer(entry["median_host_ns"], 1, _MAX_NS),
            f"fastest call of {side} must hold a shape and a host time",
        )
        hosts[side] = entry["median_host_ns"]
    return hosts


def _roof(flops: int, model_bytes: int, fma: float, bandwidth: float) -> dict:
    exact = Fraction(flops, model_bytes)
    roof = bandwidth * (exact.numerator / exact.denominator)
    return {
        "flops_per_byte": _ratio(exact),
        "bandwidth_roof_gflops_per_second": roof,
        "attainable_gflops_per_second": min(fma, roof),
        "bound": "bandwidth" if roof < fma else "compute",
    }


def _traffic(side: str, m: int, n: int, k: int, fma: float, bandwidth: float, rate):
    if side == _TORCH:
        return None
    tile = None if side == _NAIVE else VARIANTS[side].tile
    if tile is None:
        loads = 2 * m * n * k
    else:
        loads = -(-n // tile) * m * k + -(-m // tile) * k * n
    load_bytes, store_bytes = 4 * loads, 4 * m * n
    roof = _roof(2 * m * n * k, load_bytes + store_bytes, fma, bandwidth)
    return {
        "model": NO_REUSE if tile is None else TILED_MODEL,
        "tile": tile,
        "load_bytes": load_bytes,
        "store_bytes": store_bytes,
        "bytes": load_bytes + store_bytes,
        "flops_per_byte": roof["flops_per_byte"],
        "bandwidth_roof_gflops_per_second": roof["bandwidth_roof_gflops_per_second"],
        "attainable_gflops_per_second": roof["attainable_gflops_per_second"],
        "attainable_share": rate / roof["attainable_gflops_per_second"],
        "bound": roof["bound"],
    }


def _point(
    point: object,
    index: int,
    sides: list[str],
    ceilings: tuple[float, float, bool],
    fastest: dict[str, int],
    rule: dict,
) -> dict:
    fma, bandwidth, scales = ceilings
    _require(type(point) is dict, f"point {index} must be a JSON object")
    m, n, k = point.get("m"), point.get("n"), point.get("k")
    calls, recorded = point.get("calls_per_sample"), point.get("sides")
    _require(
        all(_integer(value, 1, MAX_DIMENSION) for value in (m, n, k))
        and _integer(calls, 1, MAX_CALLS_PER_SAMPLE),
        f"point {index} has an invalid shape or batch",
    )
    _require(
        type(recorded) is dict
        and set(recorded) == set(sides)
        and all(type(entry) is dict for entry in recorded.values()),
        f"point {index} must hold every side",
    )
    flops = 2 * m * n * k
    ideal_bytes = 4 * (m * k + k * n + m * n)
    ideal = {"bytes": ideal_bytes, **_roof(flops, ideal_bytes, fma, bandwidth)}
    entries = {}
    for side in sides:
        entry = recorded[side]
        median, one_call = entry.get("median_sample_ns"), entry.get("one_call")
        _require(
            _integer(median, 1, _MAX_NS)
            and type(one_call) is dict
            and set(one_call) == set(_TIMES)
            and all(_integer(one_call[name], 1, _MAX_NS) for name in _TIMES),
            f"point {index} {side} needs a median sample time and one-call times",
        )
        _require(
            fastest[side] <= one_call["median_host_ns"],
            f"point {index} {side} is faster than the fastest call of the side",
        )
        steady = median / calls
        rate = flops * calls / median
        decision = _rule(
            steady,
            rate,
            (*(one_call[name] for name in _TIMES), fastest[side]),
            fma,
            scales,
            ideal["bandwidth_roof_gflops_per_second"],
            rule,
        )
        entries[side] = {
            "median_sample_ns": median,
            "steady_ns_per_call": steady,
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / fma,
            "ideal_reuse_share": rate / ideal["attainable_gflops_per_second"],
            "traffic": _traffic(side, m, n, k, fma, bandwidth, rate),
            "one_call": one_call,
            "decision": decision,
        }
    expected = {
        "index": index,
        "m": m,
        "n": n,
        "k": k,
        "flops_per_call": flops,
        "calls_per_sample": calls,
        "ideal_reuse": ideal,
        "sides": entries,
    }
    _require(_exact(point, expected), f"point {index} does not recompute")
    return expected


def _summary(label: str, sides: list[str], points: list[dict]) -> dict:
    def span(side: str, field: str) -> list[float]:
        values = sorted(point["sides"][side][field] for point in points)
        return [values[0], values[-1]]

    def modeled(side: str) -> list[float]:
        return sorted(
            point["sides"][side]["traffic"]["attainable_share"] for point in points
        )

    kernels = sides[:-1]
    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            side: _top(
                [point["sides"][side]["decision"]["bottleneck"] for point in points]
            )
            for side in sides
        },
        "fma_ceiling_share_range": {
            side: span(side, "fma_ceiling_share") for side in sides
        },
        "ideal_reuse_share_range": {
            side: span(side, "ideal_reuse_share") for side in sides
        },
        "modeled_roof_share_range": {
            side: [modeled(side)[0], modeled(side)[-1]] for side in kernels
        },
        "points_above_modeled_roof": {
            side: len([share for share in modeled(side) if share > 1])
            for side in kernels
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
    measured = _sources(report["sources"])
    _require(
        report["measured"] is measured,
        "a gap is a measurement only when all three sources are measurements",
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
    _switches(report["torch_mps_switches"])
    rule = _rule_config(report["rule"], config)
    ceilings = _ceilings(report["ceilings"])
    _require(_exact(report["definitions"], DEFINITIONS), "definitions were changed")
    _require(_exact(report["reasons"], reasons_mapping()), "reasons were changed")
    sides = _sides(report["sides"])
    fastest = _fastest(report["fastest_calls"], sides)
    points = report["points"]
    _require(
        type(points) is list and 1 <= len(points) <= MAX_POINTS,
        f"a gap holds 1 to {MAX_POINTS} points",
    )
    expected = [
        _point(point, index, sides, ceilings, fastest, rule)
        for index, point in enumerate(points)
    ]
    summary = _summary(label, sides, expected)
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    if rebuilt is not None:
        _require(type(rebuilt) is dict, "a rebuilt gap must be a JSON object")
        _require(
            _exact(report["sources"], rebuilt.get("sources")),
            "report was not made from the supplied source reports",
        )
        _require(
            canonical_json_bytes(report) == canonical_json_bytes(rebuilt),
            "report does not match the gap rebuilt from its source reports",
        )
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "measured": measured,
        "scope": scope,
        "label": label,
        "points": len(points),
        "sides": sides,
        "rule_config_sha256": report["rule"]["config_sha256"],
        "tiled_report_sha256": report["sources"][TILED_SOURCE]["report_sha256"],
        "ceiling_report_sha256": report["sources"][CEILING_SOURCE]["report_sha256"],
        "timing_report_sha256": report["sources"][TIMING_SOURCE]["report_sha256"],
        "top_bottleneck": {
            side: summary["top_bottleneck"][side]["top"] for side in sides
        },
        "sources_checked": rebuilt is not None,
    }


def verify_gap_report(
    report: object,
    *,
    config: BottleneckConfig | None = None,
    rebuilt: object | None = None,
) -> dict[str, object]:
    """Verify a sealed gap report; bind it to a rule config or its sources.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulTiledGapVerificationError``.
    """

    if config is not None and type(config) is not BottleneckConfig:
        raise TypeError("config must be a BottleneckConfig")
    try:
        return _verify(report, config, rebuilt)
    except MatmulTiledGapVerificationError:
        raise
    except (
        MatmulTiledGapError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulTiledGapVerificationError(
            f"report failed verification: {error}"
        ) from error
