"""Independent verification of sealed tiled matmul reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
correctness fact from the seed and the stored digests, with its own code.
For each case it draws the operands again, recomputes the float64 reference
and replays the fma chain with a round-to-odd float64 step (S. Boldo and G.
Melquiond, IEEE Transactions on Computers 57(4), 2008), while the run's
replay repairs midpoints, so the two methods check each other. An output
that records the replay's bits must hash to the replay plus its sentinel
guard tail and must record the replay's error. An output that records other
bits cannot be rebuilt; it must record a digest other than that one, a count
of 1 to M * N differing elements and an error within the bound.

The shader section must name the Day 74 source by its digest and hold the
tiled library that the registry rebuilds for the config's variants, with
each variant's digest, mode, limits and a verdict that this module derives
again from those limits. Timings cannot be observed again either, so they
are checked for structure: sample counts, positive integers, and the
statistics, rates, batch sizes, modeled loads, checks, three numbers, step
verdicts, run totals and summary derived from them. A check that claims the
naive bits must sit exactly 0 from them, because equal bits subtract to 0.
The warnings, the MPS switches, the load averages, the environment and the
labels are checked for form. Every section must match its rebuild byte for
byte as canonical JSON, so ``1``, ``1.0`` and ``true`` stay distinct. No
measuring module is called (the shared config parser only validates bounds),
and torch is never imported.

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

from .gpu_ceiling_report import LOAD_AVERAGE_SOURCE
from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_metal_reference import (
    BOUND_SOURCE,
    ERROR_BOUND,
    GAMMA,
    REFERENCE,
    RELATIVE_ERROR,
)
from .matmul_metal_report import (
    FLOPS_PER_CALL,
    GFLOPS,
    MATMUL_PRECISION_FIELD,
    TORCH_EQUIVALENT,
)
from .matmul_metal_source import SHADER_SOURCE
from .matmul_tiled_benchmark import AGREEMENT, LOADS_PER_OUTPUT
from .matmul_tiled_comparison import CUBLAS, NAIVE_LABEL, RATIO_RULE, TILED_RULE
from .matmul_tiled_config import MatmulTiledConfig, config_from_mapping
from .matmul_tiled_report import (
    CORRECTNESS_RULES,
    MAX_WARNING_TEXT,
    MAX_WARNINGS,
    MEASUREMENT_BOUNDARY,
    REPLAY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SWITCH_SOURCE,
    SYNCHRONIZATION,
    SYNTHETIC_BOUNDARY,
    WARNING_CAPTURE,
)
from .matmul_tiled_source import HEADER, MatmulTiledError
from .matmul_tiled_variants import VARIANTS
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_MAX_CALLS = 4096
_CORRECTNESS_STREAM = 0
_SENTINEL = 0x7FC00000
_U32 = 2.0**-24
_U64 = 2.0**-53
_NAIVE, _ROWS, _TORCH = "matmul_x_columns", "matmul_x_rows", "torch_matmul"
_MODES = ("k_order_fma", "reordered_sum")
_FEWER = "pipeline_allows_fewer_threads_than_the_tile"
_BYTES = "static_threadgroup_bytes_differ_from_the_tiles"
_VERDICTS = ("faster", "slower", "no_clear_difference")
_SWITCHES = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")
_BUFFERS = ["c", "a", "b", "m", "n", "k"]
_SCALARS = [3, 4, 5]
_DAY_74_KERNELS = [
    {
        "name": name,
        "mapping": mapping,
        "axes": 2,
        "buffers": _BUFFERS,
        "scalar_buffers": _SCALARS,
    }
    for name, mapping in ((_NAIVE, "x_columns"), (_ROWS, "x_rows"))
]
_CORRECTNESS_TEXT = {
    "reference": REFERENCE,
    "replay": REPLAY,
    "relative_error": RELATIVE_ERROR,
    "bound": ERROR_BOUND,
    "gamma": GAMMA,
    "bound_source": BOUND_SOURCE,
    "sentinel_float32_bits": _SENTINEL,
    "rules": CORRECTNESS_RULES,
}
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
    "shaders",
    "refusals",
    "planned_work",
    "correctness",
    "benchmark",
    "comparison",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MatmulTiledVerificationError(MatmulTiledError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulTiledVerificationError(message)


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


def _text_digest(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


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
    """Replay ``sum = fma(a, b, sum)`` in k order with a round-to-odd float64 step.

    An inexact float64 sum with an even last significand bit moves one ulp
    toward the exact sum, which makes it the round-to-odd value; rounding
    that to float32 then rounds the exact sum once.
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


def _admission(name: str, tile: int, limits: dict[str, int]) -> dict[str, object]:
    threads, declared = tile * tile, 2 * tile * tile * 4
    reason = None
    if limits["max_threads_per_threadgroup"] < threads:
        reason = _FEWER
    elif limits["static_threadgroup_memory_bytes"] != declared:
        reason = _BYTES
    return {
        "name": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": threads,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "declared_threadgroup_bytes": declared,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
    }


def _shaders(section: object, config: MatmulTiledConfig):
    """Check both libraries; return every kernel's limits and each verdict."""

    _require(
        type(section) is dict and set(section) == {"naive", "tiled"},
        "shaders must hold the naive and the tiled library",
    )
    naive, tiled = section["naive"], section["tiled"]
    kernels = naive.get("kernels") if type(naive) is dict else None
    entries = tiled.get("variants") if type(tiled) is dict else None
    names = config.variants
    _require(
        type(kernels) is list
        and len(kernels) == len(_DAY_74_KERNELS)
        and type(entries) is list
        and len(entries) == len(names),
        "shaders must list both Day 74 kernels and every config variant",
    )
    limits = {
        spec["name"]: _limits(entry.get("limits") if type(entry) is dict else None)
        for spec, entry in zip(_DAY_74_KERNELS, kernels)
    }
    rebuilt, admissions = [], {}
    for name, entry in zip(names, entries):
        variant = VARIANTS[name]
        tile = variant.tile
        _require(variant.correctness_mode in _MODES, f"{name} has no known mode")
        limits[name] = _limits(entry.get("limits") if type(entry) is dict else None)
        admissions[name] = _admission(name, tile, limits[name])
        rebuilt.append(
            {
                "name": name,
                "tile": tile,
                "group_size": [tile, tile],
                "threads_per_group": tile * tile,
                "threadgroup_bytes": 2 * tile * tile * 4,
                "correctness_mode": variant.correctness_mode,
                "buffers": _BUFFERS,
                "scalar_buffers": _SCALARS,
                "source_sha256": _text_digest(variant.source),
                "limits": limits[name],
                "admission": admissions[name],
            }
        )
    source = HEADER + "".join("\n" + VARIANTS[name].source for name in names)
    _require(
        type(tiled) is dict and _exact(tiled.get("source"), source),
        "the report was not made with this lab's tiled source",
    )
    expected = {
        "naive": {
            "source_sha256": _text_digest(SHADER_SOURCE),
            "kernels": [
                {**spec, "limits": limits[spec["name"]]} for spec in _DAY_74_KERNELS
            ],
        },
        "tiled": {
            "source": source,
            "source_sha256": _text_digest(source),
            "variants": rebuilt,
        },
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return limits, admissions


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


def _kernels(config: MatmulTiledConfig, admitted: list[str]) -> list[tuple]:
    """Return ``(name, group, mode)`` of every kernel that ran, in run order."""

    kernels = [(_NAIVE, list(config.naive.group_size), "k_order_fma")]
    for name in admitted:
        tile = VARIANTS[name].tile
        kernels.append((name, [tile, tile], VARIANTS[name].correctness_mode))
    return kernels


def _output(
    entry: object,
    name: str,
    mode: str,
    guard: int,
    replay: tuple[numpy.ndarray, float],
    elements: int,
    bound: float,
) -> None:
    """Check one output record; the replay's bits rebuild it, other bits cannot."""

    _require(type(entry) is dict, f"{name} output must be a JSON object")
    bits, replay_error = replay
    tail = numpy.full(guard, _SENTINEL, dtype=numpy.uint32).view(numpy.float32)
    rebuilt = _hash(bits, tail)
    differing, digest = entry.get("differing_elements"), entry.get("buffer_sha256")
    error = entry.get("max_relative_error")
    _require(type(differing) is int, f"{name} must count its differing elements")
    if differing == 0:
        error = replay_error
        digest = rebuilt
    else:
        _require(
            1 <= differing <= elements
            and isinstance(digest, str)
            and _SHA256.fullmatch(digest) is not None
            and digest != rebuilt
            and type(error) is float
            and math.isfinite(error)
            and 0.0 <= error <= bound,
            f"{name} records other bits, so it needs another digest, a count of "
            "differing elements and an error within the bound",
        )
    expected = {
        "kernel": name,
        "correctness_mode": mode,
        "buffer_sha256": digest,
        "guard_elements": guard,
        "max_relative_error": error,
        "bound_ratio": error / bound,
        "differing_elements": differing,
        "replay_bits": differing == 0,
    }
    _require(
        _exact(entry, expected),
        f"{name} does not rebuild from the replay, its guard tail and the bound",
    )


def _cases(
    section: object,
    config: MatmulTiledConfig,
    limits: dict,
    admitted: list[str],
) -> list[dict[str, object]]:
    shapes = config.correctness.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "correctness cases do not cover the config",
    )
    kernels = _kernels(config, admitted)
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        generator = numpy.random.default_rng([config.seed, _CORRECTNESS_STREAM, index])
        left = generator.standard_normal((m, k), dtype=numpy.float32)
        right = generator.standard_normal((k, n), dtype=numpy.float32)
        total, scale = _reference(left, right)
        replay = _replay(left, right)
        bound = _gamma(k, _U32) + _gamma(k, _U64)
        error = _relative_error(replay, total, scale)
        _require(error <= bound, f"{label}: the replay exceeds the bound")
        launches = {
            name: _launch([n, m], group, limits[name]) for name, group, _ in kernels
        }
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(kernels),
            f"{label} must record one output per kernel that ran",
        )
        for output, (name, _, mode) in zip(outputs, kernels):
            guard = launches[name]["padding_threads"]
            _output(output, name, mode, guard, (replay, error), m * n, bound)
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
    entry: object, name: str, calls: int, flops: int, config: MatmulTiledConfig
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


def _loads(m: int, n: int, k: int, tile: int | None) -> float:
    if tile is None:
        return 2 * m * n * k / (m * n)
    useful = -(-n // tile) * m * k + -(-m // tile) * k * n
    return useful / (m * n)


def _checks(entry: object, sides: list[str], agreement: float) -> dict:
    _require(
        type(entry) is dict and set(entry) == set(sides[1:]),
        "a point must check every side against the naive output",
    )
    for name, check in entry.items():
        _require(
            type(check) is dict
            and set(check) == {"same_bits_as_naive", "disagreement"}
            and type(check["same_bits_as_naive"]) is bool
            and type(check["disagreement"]) is float
            and math.isfinite(check["disagreement"])
            and 0.0 <= check["disagreement"] <= agreement,
            f"{name} must agree with the naive output within the agreement bound",
        )
        # Equal bits subtract to 0, so a record cannot claim both.
        _require(
            not check["same_bits_as_naive"] or check["disagreement"] == 0.0,
            f"{name} cannot hold the naive bits and sit "
            f"{check['disagreement']!r} from them",
        )
    return entry


def _points(
    section: object, config: MatmulTiledConfig, limits: dict, admitted: list[str]
) -> list[dict[str, object]]:
    shapes = config.benchmark.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "benchmark points do not cover the sweep",
    )
    kernels = _kernels(config, admitted)
    sides = [_NAIVE, *admitted, _TORCH]
    rebuilt = []
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        _require(type(entry) is dict, f"point {index} must be a JSON object")
        m, n, k = shape.m, shape.n, shape.k
        flops = 2 * m * n * k
        calls = _calls(flops, config.benchmark.target_flops_per_sample)
        recorded = entry.get("sides")
        _require(type(recorded) is dict, f"point {index} sides must be an object")
        agreement = 4 * _gamma(k, _U32)
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "flops_per_call": flops,
            "side_order": sides,
            "launches": {
                name: _launch([n, m], group, limits[name]) for name, group, _ in kernels
            },
            "global_loads_per_output": {
                name: _loads(m, n, k, None if name == _NAIVE else group[0])
                for name, group, _ in kernels
            },
            "sides": {
                name: _side(recorded.get(name), name, calls, flops, config)
                for name in sides
            },
            "checks": _checks(entry.get("checks"), sides, agreement),
            "agreement_bound": agreement,
        }
        _require(
            _exact(entry, expected),
            f"point {index} at {[m, n, k]} does not recompute from its samples",
        )
        rebuilt.append(expected)
    return rebuilt


def _number(point: dict, side: str) -> dict[str, object]:
    entry = point["sides"][side]
    return {
        "side": side,
        "median_gflops_per_second": entry["median_gflops_per_second"],
        "iqr_gflops_per_second": entry["iqr_gflops_per_second"],
    }


def _step(first: dict, second: dict, minimum: float) -> dict[str, object]:
    one, two = first["timing"], second["timing"]
    ratio = first["median_gflops_per_second"] / second["median_gflops_per_second"]
    ahead, behind = one["p75_ns"] < two["p25_ns"], two["p75_ns"] < one["p25_ns"]
    verdict = _VERDICTS[2]
    if ratio >= minimum and ahead:
        verdict = _VERDICTS[0]
    elif 1 / ratio >= minimum and behind:
        verdict = _VERDICTS[1]
    return {
        "first": first["implementation"],
        "second": second["implementation"],
        "ratio": ratio,
        "iqr_separated": ahead or behind,
        "verdict": verdict,
    }


def _comparison(
    points: list[dict[str, object]],
    config: MatmulTiledConfig,
    admitted: list[str],
    torch_version: str,
) -> dict[str, object]:
    minimum = config.comparison.min_faster_ratio
    entries = []
    for point in points:
        best = None
        for name in admitted:
            rate = point["sides"][name]["median_gflops_per_second"]
            if best is None or rate > point["sides"][best]["median_gflops_per_second"]:
                best = name
        sides = point["sides"]
        tiled = None if best is None else sides[best]
        entries.append(
            {
                "index": point["index"],
                "m": point["m"],
                "n": point["n"],
                "k": point["k"],
                "naive": _number(point, _NAIVE),
                "tiled": None
                if best is None
                else {**_number(point, best), "tile": VARIANTS[best].tile},
                "vendor": _number(point, _TORCH),
                "cublas": CUBLAS,
                "tiled_over_naive": None
                if tiled is None
                else _step(tiled, sides[_NAIVE], minimum),
                "vendor_over_tiled": None
                if tiled is None
                else _step(sides[_TORCH], tiled, minimum),
            }
        )
    return {
        "naive": NAIVE_LABEL,
        "tiled_rule": TILED_RULE,
        "vendor": f"vendor GPU matmul: torch.matmul on MPS, PyTorch {torch_version}",
        "cublas": CUBLAS,
        "ratio_rule": RATIO_RULE,
        "min_faster_ratio": minimum,
        "points": entries,
    }


def _planned(config: MatmulTiledConfig) -> dict[str, int]:
    sweep = config.benchmark
    sides = 2 + len(config.variants)
    samples = sides * (sweep.warmup + sweep.repeats)
    flops = [2 * shape.m * shape.n * shape.k for shape in sweep.shapes]
    batches = [_calls(work, sweep.target_flops_per_sample) for work in flops]
    cases = config.correctness.shapes
    return {
        "correctness_cases": len(cases),
        "replay_steps": sum(shape.m * shape.n * shape.k for shape in cases),
        "correctness_launches": len(cases) * (1 + len(config.variants)),
        "benchmark_points": len(flops),
        "sides_per_point": sides,
        "timed_samples": sides * len(flops) * sweep.repeats,
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


def _switches(section: object) -> None:
    values = section.get("values") if type(section) is dict else None
    _require(
        type(section) is dict
        and set(section) == {"source", "values"}
        and section["source"] == SWITCH_SOURCE
        and type(values) is dict
        and set(values) == set(_SWITCHES)
        and all(
            value is None
            or (isinstance(value, str) and _TOKEN.fullmatch(value) is not None)
            for value in values.values()
        ),
        "the MPS switches must be tokens or null, from the environment",
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


def _record(entry: object) -> tuple[str, str, str, int]:
    _require(
        type(entry) is dict
        and set(entry) == {"category", "message", "file", "line", "count"}
        and all(
            isinstance(entry[field], str) and len(entry[field]) <= MAX_WARNING_TEXT
            for field in ("category", "message", "file")
        )
        and type(entry["line"]) is int
        and entry["line"] >= 0
        and type(entry["count"]) is int
        and entry["count"] >= 1,
        "a warning record must hold its category, message, file, line and count",
    )
    return entry["category"], entry["message"], entry["file"], entry["line"]


def _warnings(section: object) -> dict[str, object]:
    _require(
        type(section) is dict
        and set(section)
        == {"capture", "torch_imported_before_capture", "count", "distinct", "records"}
        and section["capture"] == WARNING_CAPTURE
        and type(section["torch_imported_before_capture"]) is bool
        and type(section["count"]) is int
        and type(section["distinct"]) is int
        and type(section["records"]) is list,
        "warnings must hold the capture, the import flag, counts and records",
    )
    records = section["records"]
    keys = [_record(entry) for entry in records]
    total = sum(entry["count"] for entry in records)
    distinct, count = section["distinct"], section["count"]
    _require(
        len(set(keys)) == len(keys)
        and len(records) == min(distinct, MAX_WARNINGS)
        and distinct <= count
        and (total == count if distinct <= MAX_WARNINGS else total <= count),
        "warning counts do not reconcile with their records",
    )
    return section


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    numbers: dict[str, object],
    refused: list[str],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    kernels = [output["kernel"] for output in cases[0]["outputs"]]
    rows = []
    for entry in numbers["points"]:
        tiled = entry["tiled"]
        first, second = entry["tiled_over_naive"], entry["vendor_over_tiled"]
        rows.append(
            {
                "m": entry["m"],
                "n": entry["n"],
                "k": entry["k"],
                "naive_gflops_per_second": entry["naive"]["median_gflops_per_second"],
                "tiled_side": None if tiled is None else tiled["side"],
                "tiled_gflops_per_second": None
                if tiled is None
                else tiled["median_gflops_per_second"],
                "vendor_gflops_per_second": entry["vendor"]["median_gflops_per_second"],
                "tiled_over_naive": None if first is None else first["verdict"],
                "vendor_over_tiled": None if second is None else second["verdict"],
            }
        )
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "refused_variants": refused,
        "cases_with_other_bits": {
            name: [
                case["index"]
                for case in cases
                if not case["outputs"][position]["replay_bits"]
            ]
            for position, name in enumerate(kernels)
        },
        "max_bound_ratio": max(
            output["bound_ratio"] for case in cases for output in case["outputs"]
        ),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "three_numbers": rows,
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
    }


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


def _verify(report: object, supplied: MatmulTiledConfig | None) -> dict[str, object]:
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
    limits, admissions = _shaders(report["shaders"], config)
    refusals = [entry for entry in admissions.values() if not entry["admitted"]]
    _require(_exact(report["refusals"], refusals), "refusals do not match the limits")
    admitted = [name for name in config.variants if admissions[name]["admitted"]]
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, limits, admitted)
    _require(
        _exact(correctness, {**_CORRECTNESS_TEXT, "cases": cases}),
        "correctness section does not match the lab's reference and bound",
    )
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config, limits, admitted)
    expected = {
        "device": "mps",
        "flops_per_call": FLOPS_PER_CALL,
        "gflops": GFLOPS,
        "torch_equivalent": TORCH_EQUIVALENT,
        "synchronization": SYNCHRONIZATION,
        "agreement": AGREEMENT,
        "loads_per_output": LOADS_PER_OUTPUT,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    torch_version = report["environment"]["torch_version"]
    numbers = _comparison(points, config, admitted, torch_version)
    _require(
        _exact(report["comparison"], numbers),
        "the three numbers or their steps do not recompute from the samples",
    )
    timed = sum(
        sum(side["samples_ns"]) for point in points for side in point["sides"].values()
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
    summary = _summary(
        label,
        cases,
        points,
        numbers,
        [entry["name"] for entry in refusals],
        captured,
        report["host_load"],
    )
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
        "refused_variants": summary["refused_variants"],
        "cases_with_other_bits": summary["cases_with_other_bits"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "three_numbers": summary["three_numbers"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object, *, config: MatmulTiledConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MatmulTiledVerificationError``.
    """

    if config is not None and not isinstance(config, MatmulTiledConfig):
        raise TypeError("config must be a MatmulTiledConfig")
    try:
        return _verify(report, config)
    except MatmulTiledVerificationError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MatmulTiledVerificationError(
            f"report failed verification: {error}"
        ) from error
