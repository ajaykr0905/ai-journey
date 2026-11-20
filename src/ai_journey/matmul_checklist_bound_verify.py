"""Independent verification of sealed checklist bound reports, on the CPU alone.

A bound report holds its inputs: the shape and batch of each point, the
median, p25 and p75 sample times and the rate of each side from the
checklist report, the three one-call medians of each side, the fastest host
call of each side, the two ceilings and their flag, the rule config and the
ratio minimum. The verifier rebuilds every derived number from those inputs
with its own code: the device and threadgroup counts of each kernel in
exact integers from the sizes in the registry, each intensity, roof,
attainable rate, share and modeled rate with the same float operations, each
name with the Day 75 verifier's copy of the ordered rule, each step ratio
and verdict with the Day 76 verifier's copy of the ratio rule, and each
hypothesis from the declared checklist. Each section must match its rebuild
byte for byte as canonical JSON, so ``1``, ``1.0`` and ``true`` stay
distinct.

The embedded inputs cannot prove themselves: a resealed report with other
medians still adds up. ``rebuilt`` closes that gap. It is the bound built
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
from .matmul_checklist_bound import (
    BOUND_ENVIRONMENT_FIELDS,
    CEILING_SOURCE,
    CHECKLIST_SOURCE,
    DEFINITIONS,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    REPORT_SCHEMA_VERSION,
    RULE_MODEL,
    SYNTHETIC_BOUNDARY,
    TIMING_SOURCE,
)
from .matmul_checklist_bound_timing import REPORT_KIND as TIMING_KIND
from .matmul_checklist_bound_traffic import (
    MODEL_TEXT,
    MatmulChecklistBoundError,
    MatmulChecklistBoundVerificationError,
)
from .matmul_checklist_report import REPORT_KIND as CHECKLIST_KIND
from .matmul_checklist_steps import CHECKLIST, checklist_sha256
from .matmul_checklist_variants import VARIANTS
from .matmul_metal_mapping import MAX_DIMENSION
from .matmul_tiled_comparison import RATIO_RULE
from .matmul_tiled_gap_verify import _environment, _ratio, _roof, _switches
from .matmul_tiled_gap_verify import _fastest as _fastest_calls
from .matmul_tiled_gap_timing import MatmulTiledGapVerificationError
from .matmul_tiled_verify import _exact, _step
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD, report_digest

_TORCH = "torch_matmul"
_TIMES = ("median_gpu_ns", "median_host_ns", "median_dispatch_ns")
_MAX_NS = 1 << 40
_MAX_RATIO = 4.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KINDS = {
    CHECKLIST_SOURCE: CHECKLIST_KIND,
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
    "checklist_sha256",
    "rule",
    "comparison",
    "environment",
    "torch_mps_switches",
    "ceilings",
    "sides",
    "fastest_calls",
    "traffic_model",
    "definitions",
    "reasons",
    "points",
    "steps",
    "summary",
    DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulChecklistBoundVerificationError(message)


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


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
        "sources must name one checklist, one ceiling and one timing report",
    )
    measured = [_source(sources[name], name) for name in _KINDS]
    _require(
        sources[CHECKLIST_SOURCE]["config_sha256"]
        == sources[TIMING_SOURCE]["config_sha256"],
        "the checklist and timing sources name other configs",
    )
    return all(measured)


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


def _comparison(section: object) -> float:
    minimum = section.get("min_faster_ratio") if type(section) is dict else None
    _require(
        type(section) is dict
        and set(section) == {"min_faster_ratio", "ratio_rule"}
        and section["ratio_rule"] == RATIO_RULE
        and type(minimum) is float
        and math.isfinite(minimum)
        and 1.0 < minimum <= _MAX_RATIO,
        "comparison must hold the Day 76 ratio rule and a minimum in (1, 4]",
    )
    return minimum


def _ceilings(ceilings: object) -> tuple[float, float, bool]:
    fields = {
        "fma_gflops_per_second",
        "bandwidth_gb_per_second",
        "ridge_flops_per_byte",
        "fma_scales_with_work",
    }
    _require(
        type(ceilings) is dict and set(ceilings) == fields,
        "ceilings fields do not match the schema",
    )
    fma, bandwidth = (
        ceilings["fma_gflops_per_second"],
        ceilings["bandwidth_gb_per_second"],
    )
    scales = ceilings["fma_scales_with_work"]
    _require(
        all(
            type(value) is float and math.isfinite(value) and value > 0
            for value in (fma, bandwidth)
        )
        and type(scales) is bool,
        "each ceiling must be a positive finite float, with a boolean flag",
    )
    _require(
        _exact(ceilings["ridge_flops_per_byte"], fma / bandwidth),
        "ceilings section does not recompute",
    )
    return fma, bandwidth, scales


def _sides(sides: object) -> list[str]:
    kernels = sides[:-1] if type(sides) is list else None
    _require(
        type(sides) is list
        and len(sides) >= 2
        and sides[0] == "matmul_x_columns"
        and sides[-1] == _TORCH
        and all(type(name) is str and name in VARIANTS for name in kernels)
        and kernels == [name for name in VARIANTS if name in kernels],
        "sides must be the S0 kernel, registered kernels in order and torch.matmul",
    )
    return sides


def _sizes(name: str) -> tuple[int, int, int, int, int, bool]:
    """The tile and thread block of a tiled kernel, from the registry."""

    variant = VARIANTS[name]
    if variant.design is None:
        tile = variant.group_size[0]
        return tile, tile, tile, 1, 1, False
    design = variant.design
    return (
        design.block_m,
        design.block_n,
        design.block_k,
        design.thread_m,
        design.thread_n,
        design.vector_loads,
    )


def _model(name: str, m: int, n: int, k: int) -> dict[str, object]:
    """Count the traffic again: no reuse for Day 74, tiles walked in integers."""

    stores = m * n
    if VARIANTS[name].group_size is None:
        a = b = m * n * k
        model = {
            "model": "no_reuse",
            "block": None,
            "outputs_per_thread": [1, 1],
            "vector_paths": {"a": False, "b": False},
            "tiles": None,
        }
        operations, shared, executed = a + b, None, m * n * k
    else:
        bm, bn, bk, tm, tn, vector = _sizes(name)
        rows, columns, phases = -(-m // bm), -(-n // bn), -(-k // bk)
        groups = rows * columns
        launched = groups * (bm // tm) * (bn // tn)
        a, b = columns * m * k, rows * k * n
        a_slots, b_slots = bm * bk * phases * groups, bk * bn * phases * groups
        vector_a, vector_b = vector and k % 4 == 0, vector and n % 4 == 0
        operations = (a // 4 if vector_a else a) + (b // 4 if vector_b else b)
        reads, executed = launched * k * (tm + tn), groups * bm * bn * k
        model = {
            "model": "blocked",
            "block": [bm, bn, bk],
            "outputs_per_thread": [tm, tn],
            "vector_paths": {"a": vector_a, "b": vector_b},
            "tiles": {
                "groups": groups,
                "phases": phases,
                "launched_threads": launched,
                "padded_slots": a_slots + b_slots - a - b,
            },
        }
        shared = {
            "store_elements": a_slots + b_slots,
            "store_operations": (a_slots // 4 if vector_a else a_slots)
            + (b_slots // 4 if vector_b else b_slots),
            "read_elements": reads,
            "reads_per_fma": _ratio(Fraction(reads, executed)),
            "barriers_per_thread": 2 * phases,
            "fma_per_thread_per_phase": bk * tm * tn,
        }
    total = 4 * (a + b + stores)
    return {
        **model,
        "device": {
            "a_loads": a,
            "b_loads": b,
            "load_operations": operations,
            "store_elements": stores,
            "load_bytes": 4 * (a + b),
            "store_bytes": 4 * stores,
            "bytes": total,
            "loads_per_output": _ratio(Fraction(a + b, stores)),
            "flops_per_byte": _ratio(Fraction(2 * m * n * k, total)),
        },
        "threadgroup": shared,
        "executed_fma": executed,
    }


def _traffic(
    side: str, m: int, n: int, k: int, fma: float, bandwidth: float, rate: float
) -> dict[str, object] | None:
    if side == _TORCH:
        return None
    model = _model(side, m, n, k)
    flops = 2 * m * n * k
    roof = _roof(flops, model["device"]["bytes"], fma, bandwidth)
    shared = model["threadgroup"]
    return {
        **model,
        "roof": {
            "bandwidth_roof_gflops_per_second": roof[
                "bandwidth_roof_gflops_per_second"
            ],
            "attainable_gflops_per_second": roof["attainable_gflops_per_second"],
            "attainable_share": rate / roof["attainable_gflops_per_second"],
            "bound": roof["bound"],
            "modeled_device_gb_per_second": model["device"]["bytes"] * rate / flops,
            "modeled_threadgroup_reads_per_ns": None
            if shared is None
            else shared["read_elements"] * rate / flops,
        },
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
        times = [entry.get(f"{name}_sample_ns") for name in ("p25", "median", "p75")]
        one_call = entry.get("one_call")
        _require(
            all(_integer(value, 1, _MAX_NS) for value in times)
            and times == sorted(times)
            and type(one_call) is dict
            and set(one_call) == set(_TIMES)
            and all(_integer(one_call[name], 1, _MAX_NS) for name in _TIMES),
            f"point {index} {side} needs ordered sample times and one-call times",
        )
        _require(
            fastest[side] <= one_call["median_host_ns"],
            f"point {index} {side} is faster than the fastest call of the side",
        )
        p25, median, p75 = times
        steady = median / calls
        rate = flops * calls / median
        entries[side] = {
            "median_sample_ns": median,
            "p25_sample_ns": p25,
            "p75_sample_ns": p75,
            "steady_ns_per_call": steady,
            "median_gflops_per_second": rate,
            "fma_ceiling_share": rate / fma,
            "ideal_reuse_share": rate / ideal["attainable_gflops_per_second"],
            "traffic": _traffic(side, m, n, k, fma, bandwidth, rate),
            "one_call": one_call,
            "decision": _rule(
                steady,
                rate,
                (*(one_call[name] for name in _TIMES), fastest[side]),
                fma,
                scales,
                ideal["bandwidth_roof_gflops_per_second"],
                rule,
            ),
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


def _timing(name: str, entry: dict) -> dict:
    """Shape a side entry as the Day 76 ratio rule reads it."""

    return {
        "implementation": name,
        "median_gflops_per_second": entry["median_gflops_per_second"],
        "timing": {"p25_ns": entry["p25_sample_ns"], "p75_ns": entry["p75_sample_ns"]},
    }


def _role(entry: dict | None, *path: str) -> object:
    for key in path:
        if entry is None:
            return None
        entry = entry[key]
    return entry


def _steps(points: list[dict], minimum: float) -> list[dict]:
    rows = []
    for step in CHECKLIST:
        for variant in step.variants:
            predicted = list(step.predicted(variant))
            entries = []
            for point in points:
                m, n, k = point["m"], point["n"], point["k"]
                first, second = _model(variant, m, n, k), _model(step.base, m, n, k)
                sides = point["sides"]
                roles = {"base": sides.get(step.base), "variant": sides.get(variant)}
                measured = None
                if None not in roles.values():
                    ratio = _step(
                        _timing(variant, roles["variant"]),
                        _timing(step.base, roles["base"]),
                        minimum,
                    )
                    measured = {
                        "ratio": ratio["ratio"],
                        "iqr_separated": ratio["iqr_separated"],
                        "verdict": ratio["verdict"],
                        "held": ratio["verdict"] in predicted,
                    }
                entries.append(
                    {
                        "index": point["index"],
                        "m": m,
                        "n": n,
                        "k": k,
                        "model": {
                            "device_bytes_ratio": _ratio(
                                Fraction(
                                    first["device"]["bytes"], second["device"]["bytes"]
                                )
                            ),
                            "load_operations_ratio": _ratio(
                                Fraction(
                                    first["device"]["load_operations"],
                                    second["device"]["load_operations"],
                                )
                            ),
                            "threadgroup_reads_per_fma": {
                                "base": _role(second["threadgroup"], "reads_per_fma"),
                                "variant": _role(first["threadgroup"], "reads_per_fma"),
                            },
                        },
                        "measured": measured,
                        "fma_ceiling_share": {
                            role: _role(entry, "fma_ceiling_share")
                            for role, entry in roles.items()
                        },
                        "attainable_share": {
                            role: _role(entry, "traffic", "roof", "attainable_share")
                            for role, entry in roles.items()
                        },
                        "bottleneck": {
                            role: _role(entry, "decision", "bottleneck")
                            for role, entry in roles.items()
                        },
                    }
                )
            rows.append(
                {
                    "step": step.step,
                    "item": step.item,
                    "variant": variant,
                    "base": step.base,
                    "predicted": predicted,
                    "points": entries,
                }
            )
    return rows


def _summary(label: str, sides: list[str], points: list[dict], steps: list[dict]):
    def values(side: str, *path: str) -> list:
        return [_role(point["sides"][side], *path) for point in points]

    def span(found: list[float]) -> list[float]:
        ordered = sorted(found)
        return [ordered[0], ordered[-1]]

    kernels = sides[:-1]
    shares = {
        side: values(side, "traffic", "roof", "attainable_share") for side in kernels
    }
    return {
        "label": label,
        "points": len(points),
        "top_bottleneck": {
            side: _top(values(side, "decision", "bottleneck")) for side in sides
        },
        "fma_ceiling_share_range": {
            side: span(values(side, "fma_ceiling_share")) for side in sides
        },
        "ideal_reuse_share_range": {
            side: span(values(side, "ideal_reuse_share")) for side in sides
        },
        "modeled_roof_share_range": {side: span(shares[side]) for side in kernels},
        "points_above_modeled_roof": {
            side: len([share for share in shares[side] if share > 1])
            for side in kernels
        },
        "modeled_bandwidth_bound_points": {
            side: len(
                [
                    bound
                    for bound in values(side, "traffic", "roof", "bound")
                    if bound == "bandwidth"
                ]
            )
            for side in kernels
        },
        "step_verdicts": {
            row["variant"]: [
                _role(entry["measured"], "verdict") for entry in row["points"]
            ]
            for row in steps
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
        "a bound is a measurement only when all three sources are measurements",
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
    _require(
        report["checklist_sha256"] == checklist_sha256(),
        "the report does not name the declared checklist",
    )
    _require(
        type(report["environment"]) is dict
        and set(report["environment"]) == set(BOUND_ENVIRONMENT_FIELDS),
        "environment fields do not match the schema",
    )
    _environment(report["environment"], measured)
    _switches(report["torch_mps_switches"])
    rule = _rule_config(report["rule"], config)
    minimum = _comparison(report["comparison"])
    ceilings = _ceilings(report["ceilings"])
    _require(report["traffic_model"] == MODEL_TEXT, "the traffic model text changed")
    _require(_exact(report["definitions"], DEFINITIONS), "definitions were changed")
    _require(_exact(report["reasons"], reasons_mapping()), "reasons were changed")
    sides = _sides(report["sides"])
    fastest = _fastest_calls(report["fastest_calls"], sides)
    points = report["points"]
    _require(
        type(points) is list and 1 <= len(points) <= MAX_POINTS,
        f"a bound holds 1 to {MAX_POINTS} points",
    )
    expected = [
        _point(point, index, sides, ceilings, fastest, rule)
        for index, point in enumerate(points)
    ]
    steps = _steps(expected, minimum)
    _require(_exact(report["steps"], steps), "steps do not recompute")
    summary = _summary(label, sides, expected, steps)
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    if rebuilt is not None:
        _require(type(rebuilt) is dict, "a rebuilt bound must be a JSON object")
        _require(
            _exact(report["sources"], rebuilt.get("sources")),
            "report was not made from the supplied source reports",
        )
        _require(
            canonical_json_bytes(report) == canonical_json_bytes(rebuilt),
            "report does not match the bound rebuilt from its source reports",
        )
    sources = report["sources"]
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "measured": measured,
        "scope": scope,
        "label": label,
        "points": len(points),
        "sides": sides,
        "rule_config_sha256": report["rule"]["config_sha256"],
        "checklist_report_sha256": sources[CHECKLIST_SOURCE]["report_sha256"],
        "ceiling_report_sha256": sources[CEILING_SOURCE]["report_sha256"],
        "timing_report_sha256": sources[TIMING_SOURCE]["report_sha256"],
        "top_bottleneck": {
            side: summary["top_bottleneck"][side]["top"] for side in sides
        },
        "sources_checked": rebuilt is not None,
    }


def verify_bound_report(
    report: object,
    *,
    config: BottleneckConfig | None = None,
    rebuilt: object | None = None,
) -> dict[str, object]:
    """Verify a sealed bound report; bind it to a rule config or its sources.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 75 or Day 76 check, is reported as
    ``MatmulChecklistBoundVerificationError``.
    """

    if config is not None and type(config) is not BottleneckConfig:
        raise TypeError("config must be a BottleneckConfig")
    try:
        return _verify(report, config, rebuilt)
    except MatmulChecklistBoundVerificationError:
        raise
    except MatmulTiledGapVerificationError as error:
        raise MatmulChecklistBoundVerificationError(str(error)) from error
    except (
        MatmulChecklistBoundError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulChecklistBoundVerificationError(
            f"report failed verification: {error}"
        ) from error
