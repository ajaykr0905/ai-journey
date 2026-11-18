"""Independent verification of sealed gap reports, on the CPU alone.

A gap report holds its inputs: the shape, calls per sample and median sample
times of each sweep point from a matmul-metal report, the two ceilings of a
gpu-ceilings report, and the identity of both sources. The verifier rebuilds
every derived number from those inputs with its own code: the FLOPs, the
no-reuse and ideal-reuse bytes and intensities in exact integers, and each
rate, roof, bound, share and torch ratio with the same float operations. Every
section must match its rebuild byte for byte as canonical JSON, so ``1``,
``1.0`` and ``true`` stay distinct.

The embedded inputs cannot prove themselves: a resealed report with other
medians or ceilings still adds up. ``rebuilt`` closes that gap. It is a gap
report built again from the two source files, whose own verifiers recompute
them, and the given report must equal it byte for byte. The summary says
whether that check ran. Nothing here imports torch.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

from .gpu_ceiling_report import REPORT_KIND as CEILING_REPORT_KIND
from .kernel_metal_benchmark import MAX_CALLS_PER_SAMPLE, MAX_POINTS
from .kernel_metal_report import (
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .matmul_metal_mapping import MAX_DIMENSION
from .matmul_metal_report import REPORT_KIND as MATMUL_REPORT_KIND
from .matmul_roofline_gap import (
    CEILING_DEFINITION,
    CEILING_SOURCE,
    DEFINITIONS,
    GAP_ENVIRONMENT_FIELDS,
    MATMUL_SOURCE,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    SYNTHETIC_BOUNDARY,
    MatmulRooflineError,
    MatmulRooflineVerificationError,
)
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD, report_digest

_COLUMNS, _ROWS, _TORCH = "matmul_x_columns", "matmul_x_rows", "torch_matmul"
_SIDES = (_COLUMNS, _ROWS, _TORCH)
_NAIVE = {_COLUMNS: "x_columns", _ROWS: "x_rows"}
_MODELS = ("no_reuse", "ideal_reuse")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "label",
    "evidence_boundary",
    "sources",
    "environment",
    "ceilings",
    "definitions",
    "points",
    "summary",
    DIGEST_FIELD,
}
_IDENTITY = {"kind", "report_sha256", "config_sha256", "measured", "label"}
_LOADS = {"load_average_before", "load_average_after"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulRooflineVerificationError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _exact(actual: object, expected: object) -> bool:
    return _canonical(actual) == _canonical(expected)


def _positive_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _decimal(value: Fraction) -> str:
    """Six decimals, half to even, as the model lab writes a positive ratio."""

    scaled = round(value * 1_000_000)
    return f"{scaled // 1_000_000}.{scaled % 1_000_000:06d}"


def _source(entry: object, kind: str, extra: set[str]) -> bool:
    _require(
        type(entry) is dict and set(entry) == _IDENTITY | extra,
        f"source {kind} fields do not match the schema",
    )
    _require(entry["kind"] == kind, f"source {kind} has another kind")
    for field in ("report_sha256", "config_sha256"):
        _require(
            type(entry[field]) is str and _SHA256.fullmatch(entry[field]) is not None,
            f"source {kind} {field} must be a SHA-256 hex digest",
        )
    measured = entry["measured"]
    _require(type(measured) is bool, f"source {kind} measured must be a boolean")
    _require(
        entry["label"] == (MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL),
        f"source {kind} label does not match whether it measured",
    )
    for field in extra:
        values = entry[field]
        _require(
            type(values) is list
            and len(values) == 3
            and all(
                type(value) is float and math.isfinite(value) and value >= 0
                for value in values
            ),
            f"source {kind} {field} must be three nonnegative floats",
        )
    return measured


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


def _ceilings(ceilings: object) -> tuple[float, float]:
    _require(
        type(ceilings) is dict
        and set(ceilings)
        == {
            "definition",
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
    _require(
        _positive_float(fma) and _positive_float(bandwidth),
        "each ceiling must be a positive finite float",
    )
    _require(
        _exact(
            ceilings,
            {
                "definition": CEILING_DEFINITION,
                "fma_gflops_per_second": fma,
                "bandwidth_gb_per_second": bandwidth,
                "ridge_flops_per_byte": fma / bandwidth,
                "fma_scales_with_work": ceilings["fma_scales_with_work"],
            },
        )
        and type(ceilings["fma_scales_with_work"]) is bool,
        "ceilings section does not recompute",
    )
    return fma, bandwidth


def _intensity(flops: int, model_bytes: int, fma: float, bandwidth: float) -> dict:
    exact = Fraction(flops, model_bytes)
    roof = bandwidth * (exact.numerator / exact.denominator)
    return {
        "bytes": model_bytes,
        "flops_per_byte": {
            "numerator": exact.numerator,
            "denominator": exact.denominator,
            "decimal": _decimal(exact),
        },
        "bandwidth_roof_gflops_per_second": roof,
        "bound": "bandwidth" if roof < fma else "compute",
    }


def _point(point: object, index: int, fma: float, bandwidth: float) -> dict:
    _require(type(point) is dict, f"point {index} must be a JSON object")
    m, n, k = point.get("m"), point.get("n"), point.get("k")
    calls, sides = point.get("calls_per_sample"), point.get("sides")
    _require(
        all(_integer(value, 1, MAX_DIMENSION) for value in (m, n, k))
        and _integer(calls, 1, MAX_CALLS_PER_SAMPLE),
        f"point {index} has an invalid shape or batch",
    )
    _require(
        type(sides) is dict
        and set(sides) == set(_SIDES)
        and all(type(entry) is dict for entry in sides.values()),
        f"point {index} must hold the three sides",
    )
    medians = {side: sides[side].get("median_sample_ns") for side in _SIDES}
    _require(
        all(type(value) is int and value > 0 for value in medians.values()),
        f"point {index} median sample times must be positive integers",
    )
    flops = 2 * m * n * k
    per_sample = flops * calls
    intensities = {
        "no_reuse": _intensity(flops, 4 * (2 * m * n * k + m * n), fma, bandwidth),
        "ideal_reuse": _intensity(flops, 4 * (m * k + k * n + m * n), fma, bandwidth),
    }
    expected_sides = {}
    for side in _SIDES:
        rate = per_sample / medians[side]
        entry = {
            "median_sample_ns": medians[side],
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / fma,
            "no_reuse_roof_share": rate
            / intensities["no_reuse"]["bandwidth_roof_gflops_per_second"],
            "ideal_reuse_roof_share": rate
            / intensities["ideal_reuse"]["bandwidth_roof_gflops_per_second"],
        }
        if side in _NAIVE:
            entry["mapping"] = _NAIVE[side]
            entry["torch_ratio"] = medians[_TORCH] / medians[side]
        expected_sides[side] = entry
    expected = {
        "index": index,
        "m": m,
        "n": n,
        "k": k,
        "flops_per_call": flops,
        "calls_per_sample": calls,
        "flops_per_sample": per_sample,
        "intensities": intensities,
        "sides": expected_sides,
    }
    _require(_exact(point, expected), f"point {index} does not recompute")
    return expected


def _summary(label: str, points: list[dict]) -> dict:
    def best(side: str) -> dict:
        top = points[0]
        for point in points[1:]:
            if (
                point["sides"][side]["fma_ceiling_share"]
                > top["sides"][side]["fma_ceiling_share"]
            ):
                top = point
        return {
            "index": top["index"],
            "fma_ceiling_share": top["sides"][side]["fma_ceiling_share"],
        }

    def span(side: str, field: str) -> list[float]:
        values = sorted(point["sides"][side][field] for point in points)
        return [values[0], values[-1]]

    return {
        "label": label,
        "points": len(points),
        "best_fma_ceiling_share": {side: best(side) for side in _SIDES},
        "no_reuse_roof_share_range": {
            side: span(side, "no_reuse_roof_share") for side in _NAIVE
        },
        "ideal_reuse_roof_share_range": {
            side: span(side, "ideal_reuse_roof_share") for side in _NAIVE
        },
        "points_above_no_reuse_roof": {
            side: len(
                [
                    point
                    for point in points
                    if point["sides"][side]["no_reuse_roof_share"] > 1
                ]
            )
            for side in _NAIVE
        },
        "torch_ratio_range": {side: span(side, "torch_ratio") for side in _NAIVE},
    }


def _verify(report: object, rebuilt: object) -> dict[str, object]:
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
    sources = report["sources"]
    _require(
        type(sources) is dict and set(sources) == {MATMUL_SOURCE, CEILING_SOURCE},
        "sources must name one matmul-metal and one ceiling report",
    )
    matmul_measured = _source(sources[MATMUL_SOURCE], MATMUL_REPORT_KIND, set())
    ceiling_measured = _source(sources[CEILING_SOURCE], CEILING_REPORT_KIND, _LOADS)
    measured = matmul_measured and ceiling_measured
    _require(
        report["measured"] is measured,
        "a gap is a measurement only when both sources are measurements",
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
    fma, bandwidth = _ceilings(report["ceilings"])
    _require(_exact(report["definitions"], DEFINITIONS), "definitions were changed")
    points = report["points"]
    _require(
        type(points) is list and 1 <= len(points) <= MAX_POINTS,
        f"a gap holds 1 to {MAX_POINTS} points",
    )
    expected = [
        _point(point, index, fma, bandwidth) for index, point in enumerate(points)
    ]
    _require(
        _exact(report["summary"], _summary(label, expected)),
        "summary does not reconcile",
    )
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
        "matmul_report_sha256": sources[MATMUL_SOURCE]["report_sha256"],
        "ceiling_report_sha256": sources[CEILING_SOURCE]["report_sha256"],
        "sources_checked": rebuilt is not None,
    }


def verify_gap_report(
    report: object, *, rebuilt: object | None = None
) -> dict[str, object]:
    """Verify a sealed gap report; with ``rebuilt``, bind it to its sources.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulRooflineVerificationError``.
    """

    try:
        return _verify(report, rebuilt)
    except MatmulRooflineVerificationError:
        raise
    except (
        MatmulRooflineError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulRooflineVerificationError(
            f"report failed verification: {error}"
        ) from error
