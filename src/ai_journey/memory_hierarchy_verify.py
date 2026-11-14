"""Independent verification of sealed memory-hierarchy measurement reports.

Timings cannot be re-observed, so verification recomputes everything that
follows from them. The embedded config is parsed strictly and must match its
digest, and a supplied config if one is given. From the config alone the
verifier rebuilds the measurement list, the byte and FLOP accounting of every
kernel and access pattern from its own tables, and every batch size. From the
raw samples it recomputes nearest-rank statistics and every rate, the curves
and their summaries, the access contrast, both roofline ceilings, the ridge
points and placements, the Little's-law products, the run totals, the planned
work and the summary, and each section must match byte for byte as canonical
JSON, so ``1``, ``1.0`` and ``true`` stay distinct. None of the measuring
modules is imported, so a bug in one of them cannot vouch for itself.

The timer, ``measured`` flag, scope and boundary must agree: only a
``time.perf_counter_ns`` report verifies as a host measurement, and an
``injected`` report verifies only as synthetic timing. A matching digest
shows internal consistency; it cannot prove that a timing was observed.
"""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256

from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_hierarchy_config import MemoryHierarchyConfig, config_from_mapping
from .memory_hierarchy_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_SCOPE,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_SCOPE,
)
from .memory_hierarchy_stream import MemoryHierarchyError

_NS = 1_000_000_000
_MAX_CALLS = 1 << 16
_WIDTH = {"float32": 4, "float64": 8}
_UNIT = {"float32": 2.0**-24, "float64": 2.0**-53}
_INDEX_BYTES = 8
_SCALAR = 3.0
# name: (calls, arrays, reads per element, writes per element, FLOPs per element)
_KERNELS = {
    "copy": (["numpy.copyto(c, a)"], ["a", "c"], 1, 1, 0),
    "scale": (["numpy.multiply(q, a, out=c)"], ["a", "c"], 1, 1, 1),
    "add": (["numpy.add(a, b, out=c)"], ["a", "b", "c"], 2, 1, 1),
    "triad": (
        ["numpy.multiply(q, b, out=c)", "numpy.add(c, a, out=c)"],
        ["a", "b", "c"],
        3,
        2,
        2,
    ),
}
# name: (call, order, index bytes per element)
_PATTERNS = {
    "copy": ("numpy.copyto(output, source)", "contiguous, no index array", 0),
    "sequential": (
        "source.take(sequential, out=output, mode='clip')",
        "element i at position i",
        8,
    ),
    "strided": (
        "source.take(strided, out=output, mode='clip')",
        "stride passes: elements j, j + S, j + 2S, ... for j = 0 .. S - 1",
        8,
    ),
    "random": (
        "source.take(random, out=output, mode='clip')",
        "numpy.random.default_rng((seed, 1)).permutation(n)",
        8,
    ),
}
_MATMUL_CALL = "numpy.matmul(left, right, out=product)"
_MATMUL_BOUND = "(gamma_K(u) + gamma_K(2**-53)) * (|A| @ |B|)"
_LAW = "in_flight = measured_rate_per_ns * stated_latency_ns"
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "measurement_scope",
    "evidence_boundary",
    "config",
    "config_sha256",
    "timer",
    "percentile_method",
    "environment",
    "planned_work",
    "stream",
    "access",
    "matmul",
    "roofline",
    "littles_law",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_ENVIRONMENT = (
    "python_version",
    "python_implementation",
    "numpy_version",
    "numpy_blas",
    "numpy_blas_version",
    "platform_system",
    "platform_machine",
    "cpu_count",
)
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MemoryHierarchyVerificationError(MemoryHierarchyError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MemoryHierarchyVerificationError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _exact(actual: object, expected: object) -> bool:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 stay distinct."""

    return _canonical(actual) == _canonical(expected)


def _digest(report: dict[str, object]) -> str:
    unsigned = {
        key: value for key, value in report.items() if key != REPORT_DIGEST_FIELD
    }
    return sha256(_canonical(unsigned)).hexdigest()


def _timing(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)

    def rank(percentile: int) -> int:
        return ordered[(percentile * len(ordered) + 99) // 100 - 1]

    return {
        "count": len(ordered),
        "min_ns": ordered[0],
        "p25_ns": rank(25),
        "median_ns": rank(50),
        "p75_ns": rank(75),
        "max_ns": ordered[-1],
        "iqr_ns": rank(75) - rank(25),
    }


def _batch(
    entry: object, calls: int, config: MemoryHierarchyConfig, label: str
) -> dict[str, object]:
    """Rebuild the batch fields of one measurement around its raw samples."""

    _require(type(entry) is dict, f"{label} must be a JSON object")
    repeats = config.timing.repeats
    samples = entry.get("samples_ns")
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{label} samples_ns must hold {repeats} positive integers",
    )
    _require(1 <= calls <= _MAX_CALLS, f"{label} batch exceeds {_MAX_CALLS} calls")
    return {
        "calls_per_sample": calls,
        "warmup": config.timing.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": _timing(samples),
    }


def _calls(bytes_per_call: int, config: MemoryHierarchyConfig) -> int:
    target = config.timing.target_bytes_per_sample
    return max(1, -(-target // bytes_per_call))


def _stream_entry(
    entry: object, name: str, count: int, config: MemoryHierarchyConfig
) -> dict[str, object]:
    _, arrays, reads, writes, flops = _KERNELS[name]
    width = _WIDTH[config.dtype]
    per_call = (reads + writes) * count * width
    calls = _calls(per_call, config)
    batch = _batch(entry, calls, config, f"stream {name} at {count} elements")
    median, minimum = batch["timing"]["median_ns"], batch["timing"]["min_ns"]
    moved, work = per_call * calls, flops * count * calls
    return {
        "kernel": name,
        "elements": count,
        "working_set_bytes": len(arrays) * count * width,
        "bytes_per_call": per_call,
        "flops_per_call": flops * count,
        **batch,
        "bytes_per_sample": moved,
        "flops_per_sample": work,
        "median_ns_per_call": median / calls,
        "median_gb_per_second": moved / median,
        "best_gb_per_second": moved / minimum,
        "median_gflops_per_second": work / median,
    }


def _curve_summary(curve: list[dict[str, object]]) -> dict[str, object]:
    peak = curve[0]
    for point in curve[1:]:
        if point["median_gb_per_second"] > peak["median_gb_per_second"]:
            peak = point
    largest = curve[-1]
    return {
        "peak_working_set_bytes": peak["working_set_bytes"],
        "peak_median_gb_per_second": peak["median_gb_per_second"],
        "largest_working_set_bytes": largest["working_set_bytes"],
        "largest_median_gb_per_second": largest["median_gb_per_second"],
        "largest_to_peak_ratio": largest["median_gb_per_second"]
        / peak["median_gb_per_second"],
    }


def _stream_section(
    section: object, config: MemoryHierarchyConfig
) -> list[dict[str, object]]:
    _require(
        type(section) is dict and type(section.get("measurements")) is list,
        "stream section must hold a measurement list",
    )
    plan = [
        (name, count)
        for count in config.stream.elements
        for name in config.stream.kernels
    ]
    measurements = section["measurements"]
    _require(
        len(measurements) == len(plan), "stream measurements do not cover the sweep"
    )
    expected_entries = []
    for entry, (name, count) in zip(measurements, plan):
        expected = _stream_entry(entry, name, count, config)
        _require(
            _exact(entry, expected),
            f"stream {name} at {count} elements does not recompute",
        )
        expected_entries.append(expected)
    curves: dict[str, list[dict[str, object]]] = {}
    for item in expected_entries:
        curves.setdefault(item["kernel"], []).append(
            {
                "elements": item["elements"],
                "working_set_bytes": item["working_set_bytes"],
                "median_gb_per_second": item["median_gb_per_second"],
                "best_gb_per_second": item["best_gb_per_second"],
            }
        )
    expected_section = {
        "dtype": config.dtype,
        "element_bytes": _WIDTH[config.dtype],
        "scalar": _SCALAR,
        "accounting": [
            {
                "name": name,
                "calls": _KERNELS[name][0],
                "arrays": _KERNELS[name][1],
                "reads_per_element": _KERNELS[name][2],
                "writes_per_element": _KERNELS[name][3],
                "flops_per_element": _KERNELS[name][4],
            }
            for name in config.stream.kernels
        ],
        "measurements": expected_entries,
        "curves": curves,
        "curve_summaries": {
            name: _curve_summary(curve) for name, curve in curves.items()
        },
    }
    _require(
        _exact(section, expected_section),
        "stream accounting, curves or curve summaries do not recompute",
    )
    return expected_entries


def _access_section(
    section: object, config: MemoryHierarchyConfig
) -> list[dict[str, object]]:
    _require(
        type(section) is dict and type(section.get("measurements")) is list,
        "access section must hold a measurement list",
    )
    measurements = section["measurements"]
    _require(len(measurements) == len(_PATTERNS), "access patterns are missing")
    width = _WIDTH[config.dtype]
    count = config.access.elements
    expected_entries = []
    for entry, (name, (call, order, index)) in zip(measurements, _PATTERNS.items()):
        per_call = count * (index + 2 * width)
        calls = _calls(per_call, config)
        batch = _batch(entry, calls, config, f"access {name}")
        median, minimum = batch["timing"]["median_ns"], batch["timing"]["min_ns"]
        moved, elements = per_call * calls, count * calls
        expected = {
            "name": name,
            "call": call,
            "order": order,
            "index_bytes": index,
            "bytes_per_call": per_call,
            **batch,
            "bytes_per_sample": moved,
            "elements_per_sample": elements,
            "median_ns_per_element": median / elements,
            "median_elements_per_ns": elements / median,
            "median_gb_per_second": moved / median,
            "best_gb_per_second": moved / minimum,
        }
        _require(_exact(entry, expected), f"access {name} does not recompute")
        expected_entries.append(expected)
    per_element = {
        item["name"]: item["median_ns_per_element"] for item in expected_entries
    }
    expected_section = {
        "dtype": config.dtype,
        "elements": count,
        "stride_elements": config.access.stride_elements,
        "stride_bytes": config.access.stride_elements * width,
        "source_bytes": count * width,
        "index_bytes": _INDEX_BYTES,
        "measurements": expected_entries,
        "contrast": {
            "sequential_to_copy_time_ratio": per_element["sequential"]
            / per_element["copy"],
            "strided_to_sequential_time_ratio": per_element["strided"]
            / per_element["sequential"],
            "random_to_sequential_time_ratio": per_element["random"]
            / per_element["sequential"],
            "random_to_strided_time_ratio": per_element["random"]
            / per_element["strided"],
        },
    }
    _require(_exact(section, expected_section), "access contrast does not recompute")
    return expected_entries


def _tolerance(inner: int, dtype: str) -> float:
    total = 0.0
    for unit in (_UNIT[dtype], _UNIT["float64"]):
        product = inner * unit
        total += product / (1 - product)
    return total


def _matmul_section(section: object, config: MemoryHierarchyConfig) -> dict:
    m, n, k = config.matmul.m, config.matmul.n, config.matmul.k
    width = _WIDTH[config.dtype]
    flops = 2 * m * n * k
    batch = _batch(section, 1, config, "matmul")
    correctness = section.get("correctness")
    ratio = correctness.get("max_bound_ratio") if type(correctness) is dict else None
    _require(
        type(ratio) is float and math.isfinite(ratio) and 0.0 <= ratio <= 1.0,
        "matmul max_bound_ratio must be a finite float in [0, 1]",
    )
    expected = {
        "m": m,
        "n": n,
        "k": k,
        "dtype": config.dtype,
        "call": _MATMUL_CALL,
        "flops_per_call": flops,
        "compulsory_bytes_per_call": (m * k + k * n + m * n) * width,
        **batch,
        "median_gflops_per_second": flops / batch["timing"]["median_ns"],
        "best_gflops_per_second": flops / batch["timing"]["min_ns"],
        "correctness": {
            "reference": "float64",
            "bound": _MATMUL_BOUND,
            "unit_roundoff": _UNIT[config.dtype],
            "tolerance": _tolerance(k, config.dtype),
            "checked_elements": m * n,
            "max_bound_ratio": ratio,
        },
    }
    _require(_exact(section, expected), "matmul accounting or rates do not recompute")
    return expected


def _estimate(
    kind: str, flops: int, read: int, written: int, peak: float, bandwidth: float
) -> dict[str, object]:
    total = read + written
    intensity = flops / total
    compute = flops / peak
    memory = total / bandwidth
    if compute > memory:
        bottleneck = "compute"
    elif memory > compute:
        bottleneck = "memory"
    else:
        bottleneck = "ridge"
    return {
        "kind": kind,
        "flops": flops,
        "bytes_read": read,
        "bytes_written": written,
        "total_bytes": total,
        "arithmetic_intensity": intensity,
        "compute_seconds": compute,
        "memory_seconds": memory,
        "lower_bound_seconds": max(compute, memory),
        "attainable_flops_per_second": min(peak, bandwidth * intensity),
        "bottleneck": bottleneck,
    }


def _placement(
    source: str, estimate: dict[str, object], measured: float
) -> dict[str, object]:
    return {
        "source": source,
        **estimate,
        "measured_flops_per_second": measured,
        "fraction_of_attainable": measured / estimate["attainable_flops_per_second"],
    }


def _roofline_section(
    section: object,
    config: MemoryHierarchyConfig,
    stream: list[dict[str, object]],
    matmul: dict[str, object],
) -> dict[str, object]:
    width = _WIDTH[config.dtype]
    triads = [item for item in stream if item["kernel"] == "triad"]
    largest = triads[-1]
    peak_point = triads[0]
    for item in triads[1:]:
        if item["median_gb_per_second"] > peak_point["median_gb_per_second"]:
            peak_point = item
    peak = matmul["flops_per_call"] * _NS / matmul["timing"]["median_ns"]

    def bandwidth(item: dict[str, object]) -> float:
        return item["bytes_per_sample"] * _NS / item["timing"]["median_ns"]

    def ceiling(name: str, item: dict[str, object]) -> dict[str, object]:
        return {
            "name": name,
            "kernel": "triad",
            "elements": item["elements"],
            "working_set_bytes": item["working_set_bytes"],
            "bytes_per_second": bandwidth(item),
            "ridge_point_flops_per_byte": peak / bandwidth(item),
        }

    memory = bandwidth(largest)
    placements = []
    for item in stream:
        name = item["kernel"]
        if item["elements"] != largest["elements"] or not _KERNELS[name][4]:
            continue
        _, _, reads, writes, flops = _KERNELS[name]
        count = item["elements"]
        estimate = _estimate(
            name,
            flops * count,
            reads * count * width,
            writes * count * width,
            peak,
            memory,
        )
        measured = item["flops_per_sample"] * _NS / item["timing"]["median_ns"]
        placements.append(_placement("stream", estimate, measured))
    m, n, k = config.matmul.m, config.matmul.n, config.matmul.k
    estimate = _estimate(
        "matmul", 2 * m * n * k, (m * k + k * n) * width, m * n * width, peak, memory
    )
    placements.append(_placement("matmul", estimate, peak))
    expected = {
        "ceiling_statistic": "median",
        "compute_ceiling": {
            "source": "matmul",
            "dtype": config.dtype,
            "m": m,
            "n": n,
            "k": k,
            "flops_per_second": peak,
        },
        "bandwidth_ceilings": [
            ceiling("largest_working_set", largest),
            ceiling("peak_sweep_point", peak_point),
        ],
        "ridge_point_flops_per_byte": peak / memory,
        "placement_elements": largest["elements"],
        "placements": placements,
    }
    _require(
        _exact(section, expected), "roofline ceilings or placements do not recompute"
    )
    return expected


def _littles_law_section(
    section: object,
    config: MemoryHierarchyConfig,
    stream: list[dict[str, object]],
    access: list[dict[str, object]],
) -> None:
    latency = config.littles_law.latency_ns
    triad = [item for item in stream if item["kernel"] == "triad"][-1]
    rate = triad["bytes_per_sample"] / triad["timing"]["median_ns"]
    expected = {
        "law": _LAW,
        "stated_latency_ns": latency,
        "latency_note": config.littles_law.latency_note,
        "latency_measured": False,
        "bandwidth": {
            "source": "stream_triad_largest_working_set",
            "elements": triad["elements"],
            "measured_bytes_per_ns": rate,
            "derived_bytes_in_flight": rate * latency,
        },
        "access_patterns": [
            {
                "pattern": item["name"],
                "measured_elements_per_ns": item["median_elements_per_ns"],
                "derived_elements_in_flight": item["median_elements_per_ns"] * latency,
            }
            for item in access
        ],
    }
    _require(_exact(section, expected), "Little's-law rows do not recompute")


def _environment(environment: object) -> None:
    _require(
        type(environment) is dict and set(environment) == set(_ENVIRONMENT),
        "environment fields do not match the host record",
    )
    for name in _ENVIRONMENT[:-1]:
        value = environment[name]
        _require(
            isinstance(value, str) and _TOKEN.fullmatch(value) is not None,
            f"environment {name} must be a version or platform token",
        )
    count = environment["cpu_count"]
    _require(
        count is None or (type(count) is int and count >= 1),
        "environment cpu_count must be a positive integer or null",
    )


def _verify(
    report: object, supplied: MemoryHierarchyConfig | None
) -> dict[str, object]:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    _require(
        type(report["schema_version"]) is int and report["schema_version"] == 1,
        "unsupported report schema",
    )
    _require(report["kind"] == REPORT_KIND, "unexpected report kind")
    _require(
        report[REPORT_DIGEST_FIELD] == _digest(report),
        "report digest does not match its content",
    )
    _require(
        report["timer"] in (TIMER, INJECTED_TIMER)
        and report["percentile_method"] == PERCENTILE_METHOD,
        "measurement method fields were changed",
    )
    measured = report["timer"] == TIMER
    _require(
        report["measured"] is measured,
        "a host-timer report must be measured"
        if measured
        else "an injected-clock report must not claim a measurement",
    )
    scope, boundary = (
        (MEASUREMENT_SCOPE, MEASUREMENT_BOUNDARY)
        if measured
        else (SYNTHETIC_SCOPE, SYNTHETIC_BOUNDARY)
    )
    _require(
        report["measurement_scope"] == scope
        and _exact(report["evidence_boundary"], boundary),
        "evidence boundary was weakened",
    )
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
            "report was not measured from the supplied config",
        )
    _environment(report["environment"])
    stream = _stream_section(report["stream"], config)
    access = _access_section(report["access"], config)
    matmul = _matmul_section(report["matmul"], config)
    roofline = _roofline_section(report["roofline"], config, stream, matmul)
    _littles_law_section(report["littles_law"], config, stream, access)
    measurements = [*stream, *access, matmul]
    samples = config.timing.warmup + config.timing.repeats
    planned = {
        "peak_allocation_bytes": config.planned_work().peak_allocation_bytes,
        "accounted_bytes": sum(
            samples * item["bytes_per_call"] * item["calls_per_sample"]
            for item in (*stream, *access)
        ),
        "calls": sum(samples * item["calls_per_sample"] for item in measurements),
        "flops": sum(samples * item["flops_per_sample"] for item in stream)
        + samples * matmul["flops_per_call"],
        "measurements": len(measurements),
    }
    _require(
        _exact(report["planned_work"], planned),
        "planned work does not match the measurements",
    )
    run = report["run"]
    timed = sum(sum(item["samples_ns"]) for item in measurements)
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "timed_ns"}
        and _exact(run["max_runtime_ns"], config.timing.max_runtime_seconds * _NS)
        and _exact(run["timed_ns"], timed)
        and type(run["elapsed_ns"]) is int
        and timed <= run["elapsed_ns"] <= run["max_runtime_ns"],
        "run time does not cover its samples within the budget",
    )
    triad = report["stream"]["curve_summaries"]["triad"]
    summary = {
        "stream_measurements": len(stream),
        "access_patterns": len(access),
        "timed_samples": len(measurements) * config.timing.repeats,
        "triad_peak_working_set_bytes": triad["peak_working_set_bytes"],
        "triad_peak_gb_per_second": triad["peak_median_gb_per_second"],
        "triad_largest_working_set_bytes": triad["largest_working_set_bytes"],
        "triad_largest_gb_per_second": triad["largest_median_gb_per_second"],
        "random_to_sequential_time_ratio": report["access"]["contrast"][
            "random_to_sequential_time_ratio"
        ],
        "compute_ceiling_flops_per_second": roofline["compute_ceiling"][
            "flops_per_second"
        ],
        "ridge_point_flops_per_byte": roofline["ridge_point_flops_per_byte"],
    }
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[REPORT_DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "measured": measured,
        "scope": scope,
        "stream_measurements": len(stream),
        "access_patterns": len(access),
        "timed_samples": summary["timed_samples"],
        "ridge_point_flops_per_byte": summary["ridge_point_flops_per_byte"],
    }


def verify_report(
    report: object, *, config: MemoryHierarchyConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MemoryHierarchyVerificationError``.
    """

    if config is not None and not isinstance(config, MemoryHierarchyConfig):
        raise TypeError("config must be a MemoryHierarchyConfig")
    try:
        return _verify(report, config)
    except MemoryHierarchyVerificationError:
        raise
    except (
        MemoryHierarchyError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MemoryHierarchyVerificationError(
            f"report failed verification: {error}"
        ) from error
