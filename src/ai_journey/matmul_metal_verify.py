"""Independent verification of sealed Metal matmul reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
correctness fact from the seed and the stored digests. For each case the
verifier draws the operands again and checks their digests, recomputes the
float64 reference with its own code and checks its digest, and replays the
kernels' fma chain with its own exact rounding. Its replay rounds each
step to odd in float64 first; with 53 bits against 24, the final rounding to
float32 is then exact (S. Boldo and G. Melquiond, IEEE Transactions on
Computers 57(4), 2008). The run's replay repairs midpoints instead, so the
two methods check each other. Each naive buffer is the replay plus its
sentinel guard tail and must hash to the stored digest, and its largest
relative error is recomputed and held to the bound. A PyTorch output has an
unknown summation order and cannot be rebuilt, so its record is checked for
structure: a digest, an error within the bound, a ratio that recomputes, and
a count of differing elements that is zero exactly when its digest is the
replay's. A record with the replay's digest holds the replay's bits, so its
error must equal the replay's error.

Timings cannot be observed again either, so they are checked for structure
only: sample counts, positive integers, and the arithmetic of the statistics,
rates, ratios, batch sizes, ceiling, run totals and summary derived from
them. Launch geometry, address steps, planned work, the shader source and
its digest, the environment tokens and the labels are checked too. Every
section must match the rebuilt one byte for byte as canonical JSON, so
``1``, ``1.0`` and ``true`` stay distinct. No measuring module is called to
recompute anything (the shared config parser only validates bounds), and
torch is never imported, so a host without torch or MPS can verify a report.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement; any ``injected`` backend or timer verifies only as
synthetic. A matching digest shows internal consistency; it cannot prove that
a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256

import numpy

from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_config import MatmulMetalConfig, config_from_mapping
from .matmul_metal_mapping import MatmulMetalError
from .matmul_metal_report import (
    FLOPS_PER_CALL,
    GFLOPS,
    MATMUL_PRECISION_FIELD,
    MEASUREMENT_BOUNDARY,
    REPLAY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNCHRONIZATION,
    SYNTHETIC_BOUNDARY,
    TORCH_EQUIVALENT,
)
from .matmul_metal_source import SHADER_SOURCE
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_MAX_CALLS = 4096
_CORRECTNESS_STREAM = 0
_FLOAT_SENTINEL = 0x7FC00000
_U32 = 2.0**-24
_U64 = 2.0**-53
_COLUMNS, _ROWS, _TORCH = "matmul_x_columns", "matmul_x_rows", "torch_matmul"
_SIDES = (_COLUMNS, _ROWS, _TORCH)
_MAPPINGS = {"x_columns": _COLUMNS, "x_rows": _ROWS}
_OUTPUTS = ((_COLUMNS, "mps"), (_ROWS, "mps"), (_TORCH, "mps"), (_TORCH, "cpu"))
_KERNELS = [
    {
        "name": kernel,
        "mapping": mapping,
        "axes": 2,
        "buffers": ["c", "a", "b", "m", "n", "k"],
        "scalar_buffers": [3, 4, 5],
    }
    for mapping, kernel in _MAPPINGS.items()
]
_CORRECTNESS_TEXT = {
    "reference": (
        "float64 sum of the exact float32 products in increasing k order, on "
        "the CPU, rebuilt from the seed"
    ),
    "replay": REPLAY,
    "relative_error": "max over elements of |C - R| / (|A| @ |B|), in float64",
    "bound": "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)",
    "gamma": "gamma_K(u) = K * u / (1 - K * u)",
    "bound_source": (
        "N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., "
        "SIAM, 2002, section 3.1"
    ),
    "sentinel_float32_bits": _FLOAT_SENTINEL,
}
_AGREEMENT = "4 * gamma_K(2**-24) * (|A| @ |B|), on the device in float32"
_CEILING = (
    "the highest median GFLOP/s of torch.matmul on the MPS device over this "
    "sweep, measured on this host in the same run; not a hardware or vendor peak"
)
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
    "shaders",
    "planned_work",
    "correctness",
    "benchmark",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MatmulMetalVerificationError(MatmulMetalError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulMetalVerificationError(message)


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


def _hash(*arrays: numpy.ndarray) -> str:
    """Hash the little-endian C-order bytes of arrays laid end to end."""

    digest = sha256()
    for array in arrays:
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        digest.update(numpy.ascontiguousarray(little).tobytes())
    return digest.hexdigest()


def _gamma(inner: int, unit: float) -> float:
    product = inner * unit
    return product / (1 - product)


def _bound(inner: int) -> float:
    return _gamma(inner, _U32) + _gamma(inner, _U64)


def _operands(seed: int, index: int, m: int, n: int, k: int):
    generator = numpy.random.default_rng([seed, _CORRECTNESS_STREAM, index])
    left = generator.standard_normal((m, k), dtype=numpy.float32)
    return left, generator.standard_normal((k, n), dtype=numpy.float32)


def _reference(left: numpy.ndarray, right: numpy.ndarray):
    """Sum the exact products in float64 in k order, with their absolute values."""

    a, b = left.astype(numpy.float64), right.astype(numpy.float64)
    total = numpy.zeros((a.shape[0], b.shape[1]))
    scale = numpy.zeros_like(total)
    for step in range(a.shape[1]):
        term = a[:, step, None] * b[None, step, :]
        total += term
        scale += numpy.abs(term)
    return total, scale


def _replay(left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
    """Replay ``sum = fma(a, b, sum)`` with a round-to-odd float64 step.

    The float64 sum is moved one ulp toward the exact sum whenever it is
    inexact and its last significand bit is even. That makes it the round-to-
    odd value, and rounding it to float32 then rounds the exact sum.
    """

    a, b = left.astype(numpy.float64), right.astype(numpy.float64)
    total = numpy.zeros((a.shape[0], b.shape[1]), dtype=numpy.float32)
    for step in range(a.shape[1]):
        term = a[:, step, None] * b[None, step, :]
        wide = total.astype(numpy.float64)
        near = term + wide
        virtual = near - term
        error = (term - (near - virtual)) + (wide - virtual)
        even = (near.view(numpy.int64) & 1) == 0
        toward = numpy.where(error > 0, numpy.inf, -numpy.inf)
        near = numpy.where(even & (error != 0), numpy.nextafter(near, toward), near)
        total = near.astype(numpy.float32)
    return total


def _relative_error(output: numpy.ndarray, total: numpy.ndarray, scale) -> float:
    error = numpy.abs(output.astype(numpy.float64) - total)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        numpy.divide(error, scale, out=error)
    return float(numpy.fmax.reduce(error, axis=None, initial=0.0))


def _launch(extent: list[int], group: list[int], limits: dict) -> dict[str, object]:
    groups = [-(-size // count) for size, count in zip(extent, group)]
    threads = [blocks * count for blocks, count in zip(groups, group)]
    per_group = math.prod(group)
    width = limits["thread_execution_width"]
    simd = -(-per_group // width)
    _require(
        per_group <= limits["max_threads_per_threadgroup"],
        "a threadgroup exceeds its pipeline's recorded limit",
    )
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


def _launches(m: int, n: int, group: list[int], limits: dict) -> dict[str, dict]:
    extents = {"x_columns": [n, m], "x_rows": [m, n]}
    return {
        mapping: _launch(extents[mapping], group, limits[kernel])
        for mapping, kernel in _MAPPINGS.items()
    }


def _limits(entry: object) -> dict[str, int]:
    _require(
        type(entry) is dict
        and set(entry)
        == {
            "max_threads_per_threadgroup",
            "thread_execution_width",
            "static_threadgroup_memory_bytes",
        }
        and all(type(value) is int for value in entry.values()),
        "pipeline limits must be three integers",
    )
    most = entry["max_threads_per_threadgroup"]
    _require(
        1 <= most <= 1024
        and 1 <= entry["thread_execution_width"] <= most
        and 0 <= entry["static_threadgroup_memory_bytes"] <= 1 << 20,
        "pipeline limits are outside what the API can report",
    )
    return entry


def _shaders(section: object) -> dict[str, dict[str, int]]:
    _require(type(section) is dict, "shaders must be a JSON object")
    kernels = section.get("kernels")
    _require(
        type(kernels) is list and len(kernels) == len(_KERNELS),
        "shaders must list both kernels",
    )
    limits = {
        spec["name"]: _limits(entry.get("limits") if type(entry) is dict else None)
        for spec, entry in zip(_KERNELS, kernels)
    }
    _require(
        _exact(section.get("source"), SHADER_SOURCE),
        "the report was not made with this lab's Metal source",
    )
    expected = {
        "source": SHADER_SOURCE,
        "source_sha256": sha256(SHADER_SOURCE.encode("ascii")).hexdigest(),
        "kernels": [{**spec, "limits": limits[spec["name"]]} for spec in _KERNELS],
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return limits


def _naive_output(
    entry: object, name: str, replay, guard: int, error: float, bound: float
) -> None:
    tail = numpy.full(guard, _FLOAT_SENTINEL, dtype=numpy.uint32).view(numpy.float32)
    expected = {
        "implementation": name,
        "device": "mps",
        "buffer_sha256": _hash(replay, tail),
        "guard_elements": guard,
        "max_relative_error": error,
        "bound_ratio": error / bound,
        "differing_elements": 0,
    }
    _require(
        _exact(entry, expected),
        f"{name} does not rebuild from the replay, its guard tail and the bound",
    )


def _torch_output(
    entry: object,
    device: str,
    replay: tuple[str, float],
    elements: int,
    bound: float,
) -> float:
    """Check what a PyTorch record can show without its output; return its error.

    ``replay`` is the replay's digest and its error. A record with that
    digest holds the replay's bits, so its error must be the replay's too.
    """

    _require(type(entry) is dict, "an output must be a JSON object")
    error, digest = entry.get("max_relative_error"), entry.get("buffer_sha256")
    differing = entry.get("differing_elements")
    replay_digest, replay_error = replay
    _require(
        type(error) is float and math.isfinite(error) and 0.0 <= error <= bound,
        f"torch_matmul on {device} must record an error within the bound",
    )
    _require(
        isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        and type(differing) is int
        and 0 <= differing <= elements
        and (differing == 0) == (digest == replay_digest),
        f"torch_matmul on {device} must count its differing elements consistently",
    )
    _require(
        digest != replay_digest or error == replay_error,
        f"torch_matmul on {device} holds the replay's bits, so it must record "
        "the replay's error",
    )
    expected = {
        "implementation": _TORCH,
        "device": device,
        "buffer_sha256": digest,
        "guard_elements": 0,
        "max_relative_error": error,
        "bound_ratio": error / bound,
        "differing_elements": differing,
    }
    _require(_exact(entry, expected), f"torch_matmul on {device} is malformed")
    return error


def _cases(
    section: object, config: MatmulMetalConfig, limits: dict
) -> list[dict[str, object]]:
    shapes = config.correctness.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "correctness cases do not cover the config",
    )
    group = list(config.launch.group_size)
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        left, right = _operands(config.seed, index, m, n, k)
        total, scale = _reference(left, right)
        replay = _replay(left, right)
        bound = _bound(k)
        error = _relative_error(replay, total, scale)
        _require(error <= bound, f"{label}: the replay exceeds the bound")
        launches = _launches(m, n, group, limits)
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(_OUTPUTS),
            f"{label} must record four outputs",
        )
        for output, mapping in zip(outputs, _MAPPINGS):
            guard = launches[mapping]["padding_threads"]
            _naive_output(output, _MAPPINGS[mapping], replay, guard, error, bound)
        for output, (_, device) in zip(outputs[2:], _OUTPUTS[2:]):
            _torch_output(output, device, (_hash(replay), error), m * n, bound)
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "launches": launches,
            "a_sha256": _hash(left),
            "b_sha256": _hash(right),
            "reference_sha256": _hash(total),
            "replay_sha256": _hash(replay),
            "bound": bound,
            "outputs": outputs,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
    return section


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


def _calls(flops: int, target: int) -> int:
    calls = max(1, -(-target // flops))
    _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
    return calls


def _side(
    entry: object, name: str, calls: int, flops: int, config: MatmulMetalConfig
) -> dict[str, object]:
    repeats = config.benchmark.repeats
    samples = entry.get("samples_ns") if type(entry) is dict else None
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{name} samples_ns must hold {repeats} positive integers",
    )
    timing = _timing(samples)
    work = flops * calls
    return {
        "implementation": name,
        "calls_per_sample": calls,
        "warmup": config.benchmark.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": timing,
        "flops_per_sample": work,
        "median_ns_per_call": timing["median_ns"] / calls,
        "iqr_ns_per_call": timing["iqr_ns"] / calls,
        "median_gflops_per_second": work / timing["median_ns"],
        "best_gflops_per_second": work / timing["min_ns"],
        "iqr_gflops_per_second": [work / timing["p75_ns"], work / timing["p25_ns"]],
    }


def _steps(m: int, n: int, k: int) -> dict[str, dict[str, object]]:
    def pair(x_step: int, y_step: int) -> dict[str, int]:
        return {"x": 4 * x_step, "y": 4 * y_step}

    return {
        "x_columns": {
            "mapping": "x_columns",
            "a_step_bytes": pair(0, k),
            "b_step_bytes": pair(1, 0),
            "c_step_bytes": pair(1, n),
        },
        "x_rows": {
            "mapping": "x_rows",
            "a_step_bytes": pair(k, 0),
            "b_step_bytes": pair(0, 1),
            "c_step_bytes": pair(n, 1),
        },
    }


def _points(
    section: object, config: MatmulMetalConfig, limits: dict
) -> list[dict[str, object]]:
    shapes = config.benchmark.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "benchmark points do not cover the sweep",
    )
    group = list(config.launch.group_size)
    rebuilt = []
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        _require(type(entry) is dict, f"point {index} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        flops = 2 * m * n * k
        calls = _calls(flops, config.benchmark.target_flops_per_sample)
        recorded = entry.get("sides")
        _require(type(recorded) is dict, f"point {index} sides must be an object")
        sides = {
            name: _side(recorded.get(name), name, calls, flops, config)
            for name in _SIDES
        }
        median = {name: sides[name]["timing"]["median_ns"] for name in _SIDES}
        disagreement = entry.get("torch_disagreement")
        agreement = 4 * _gamma(k, _U32)
        _require(
            type(disagreement) is float
            and math.isfinite(disagreement)
            and 0.0 <= disagreement <= agreement,
            f"point {index}: the timed torch.matmul output must agree with the "
            "naive output",
        )
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "flops_per_call": flops,
            "launches": _launches(m, n, group, limits),
            "access_steps": _steps(m, n, k),
            "sides": sides,
            "naive_to_torch_gflops_ratio": {
                name: median[_TORCH] / median[name] for name in (_COLUMNS, _ROWS)
            },
            "x_rows_to_x_columns_gflops_ratio": median[_COLUMNS] / median[_ROWS],
            "torch_disagreement": disagreement,
            "agreement_bound": agreement,
        }
        _require(
            _exact(entry, expected),
            f"point {index} at {[m, n, k]} does not recompute from its samples",
        )
        rebuilt.append(expected)
    return rebuilt


def _ceiling(points: list[dict[str, object]]) -> dict[str, object]:
    def rate(point: dict[str, object], side: str) -> float:
        return point["sides"][side]["median_gflops_per_second"]

    def best(side: str) -> dict[str, object]:
        top = points[0]
        for point in points[1:]:
            if rate(point, side) > rate(top, side):
                top = point
        return top

    top = best(_TORCH)
    ceiling = rate(top, _TORCH)
    naive = {}
    for name in (_COLUMNS, _ROWS):
        point = best(name)
        naive[name] = {
            "best_median_gflops_per_second": rate(point, name),
            "best_index": point["index"],
            "fraction_of_measured_ceiling": rate(point, name) / ceiling,
        }
    return {
        "definition": _CEILING,
        "measured_ceiling_gflops_per_second": ceiling,
        "ceiling_index": top["index"],
        "naive": naive,
    }


def _benchmark(
    section: object, config: MatmulMetalConfig, limits: dict
) -> list[dict[str, object]]:
    _require(type(section) is dict, "benchmark must be a JSON object")
    points = _points(section.get("points"), config, limits)
    expected = {
        "device": "mps",
        "flops_per_call": FLOPS_PER_CALL,
        "gflops": GFLOPS,
        "torch_equivalent": TORCH_EQUIVALENT,
        "synchronization": SYNCHRONIZATION,
        "agreement": _AGREEMENT,
        "points": points,
        "ceiling": _ceiling(points),
    }
    _require(_exact(section, expected), "benchmark section does not recompute")
    return points


def _planned(config: MatmulMetalConfig) -> dict[str, int]:
    sweep = config.benchmark
    samples = len(_SIDES) * (sweep.warmup + sweep.repeats)
    flops = [2 * shape.m * shape.n * shape.k for shape in sweep.shapes]
    batches = [_calls(work, sweep.target_flops_per_sample) for work in flops]
    return {
        "correctness_cases": len(config.correctness.shapes),
        "replay_steps": sum(
            shape.m * shape.n * shape.k for shape in config.correctness.shapes
        ),
        "benchmark_points": len(flops),
        "timed_samples": len(_SIDES) * len(flops) * sweep.repeats,
        "calls": sum(samples * batch for batch in batches),
        "flops": sum(samples * batch * work for batch, work in zip(batches, flops)),
    }


def _environment(environment: object, measured: bool) -> None:
    fields = {*ENVIRONMENT_FIELDS, MATMUL_PRECISION_FIELD}
    _require(
        type(environment) is dict and set(environment) == fields,
        "environment fields do not match the host record",
    )
    for name, value in environment.items():
        _require(
            isinstance(value, str) and _TOKEN.fullmatch(value) is not None,
            f"environment {name} must be a version or platform token",
        )
    _require(
        not measured or environment["platform_system"] == "Darwin",
        "an Apple GPU measurement must come from macOS",
    )


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    repeats: int,
) -> dict[str, object]:
    keys = [f"{name}/{device}" for name, device in _OUTPUTS]
    largest = points[0]
    for point in points[1:]:
        if point["flops_per_call"] > largest["flops_per_call"]:
            largest = point
    ceiling = _ceiling(points)
    return {
        "label": label,
        "correctness_cases": len(cases),
        "naive_outputs_replay_exactly": True,
        "max_relative_error": {
            key: max(case["outputs"][position]["max_relative_error"] for case in cases)
            for position, key in enumerate(keys)
        },
        "max_bound_ratio": max(
            output["max_relative_error"] / case["bound"]
            for case in cases
            for output in case["outputs"]
        ),
        "torch_cases_with_other_bits": {
            key: sum(
                case["outputs"][position]["differing_elements"] > 0 for case in cases
            )
            for position, key in enumerate(keys)
            if key.startswith(_TORCH)
        },
        "benchmark_points": len(points),
        "timed_samples": len(_SIDES) * len(points) * repeats,
        "measured_ceiling_gflops_per_second": ceiling[
            "measured_ceiling_gflops_per_second"
        ],
        "naive_fraction_of_measured_ceiling": {
            name: entry["fraction_of_measured_ceiling"]
            for name, entry in ceiling["naive"].items()
        },
        "largest_point": {
            "m": largest["m"],
            "n": largest["n"],
            "k": largest["k"],
            "median_gflops_per_second": {
                side: largest["sides"][side]["median_gflops_per_second"]
                for side in _SIDES
            },
            "naive_to_torch_gflops_ratio": largest["naive_to_torch_gflops_ratio"],
        },
    }


def _verify(report: object, supplied: MatmulMetalConfig | None) -> dict[str, object]:
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
    limits = _shaders(report["shaders"])
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, limits)
    _require(
        _exact(correctness, {**_CORRECTNESS_TEXT, "cases": cases}),
        "correctness section does not match the lab's reference and bound",
    )
    points = _benchmark(report["benchmark"], config, limits)
    timed = sum(
        sum(point["sides"][side]["samples_ns"]) for point in points for side in _SIDES
    )
    run = report["run"]
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "timed_ns"}
        and _exact(run["max_runtime_ns"], config.benchmark.max_runtime_seconds * _NS)
        and _exact(run["timed_ns"], timed)
        and type(run["elapsed_ns"]) is int
        and timed <= run["elapsed_ns"] <= run["max_runtime_ns"],
        "run time does not cover its samples within the budget",
    )
    summary = _summary(label, cases, points, config.benchmark.repeats)
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[REPORT_DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "measured_ceiling_gflops_per_second": summary[
            "measured_ceiling_gflops_per_second"
        ],
        "naive_fraction_of_measured_ceiling": summary[
            "naive_fraction_of_measured_ceiling"
        ],
    }


def verify_report(
    report: object, *, config: MatmulMetalConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulMetalVerificationError``.
    """

    if config is not None and not isinstance(config, MatmulMetalConfig):
        raise TypeError("config must be a MatmulMetalConfig")
    try:
        return _verify(report, config)
    except MatmulMetalVerificationError:
        raise
    except (
        MatmulMetalError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulMetalVerificationError(
            f"report failed verification: {error}"
        ) from error
