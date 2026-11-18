"""Independent verification of sealed Metal kernel reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
correctness fact from the seed and the stored digests. For each case the
verifier draws the inputs again, checks their digests, recomputes the exact
integer reference with its own code and checks its digest. It then rebuilds
each recorded output from the reference: a vector-add output is the
reference itself, a grayscale output is the reference with every recorded
mismatched pixel set to its recorded level, and the Metal buffer adds its
sentinel guard tail. Each rebuilt buffer must hash to the stored digest, so
the recorded mismatches are complete, and the mismatch counts and maximum
differences are recomputed from them and held to the bound.

Timings cannot be observed again either, so they are checked for structure
only: sample counts, positive integers, and the arithmetic of the statistics,
rates, batch sizes, run totals and summary derived from them. Launch
geometry, planned work, the shader source and its digest, the contraction
probe, the environment tokens and the labels are checked too. Every section
must match the rebuilt one byte for byte as canonical JSON, so ``1``, ``1.0``
and ``true`` stay distinct. No measuring module is called to recompute
anything (the shared config parser only validates bounds), and torch is never
imported, so a host without torch or MPS can verify a report.

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

from .kernel_metal_config import KernelMetalConfig, config_from_mapping
from .kernel_metal_geometry import KernelMetalError
from .kernel_metal_report import (
    DEVICE,
    ENVIRONMENT_FIELDS,
    INJECTED_BACKEND,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    REFERENCE,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    ROUNDING_RULE,
    SYNCHRONIZATION,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
    TORCH_EQUIVALENTS,
)
from .kernel_metal_runtime import BACKEND
from .kernel_metal_source import SHADER_SOURCE
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_MAX_CALLS = 4096
_MAX_MISMATCHES = 1024
# The exact integer rule leaves no rounding to differ: outputs must match.
_BOUND_LEVELS = 0
_BENCHMARK_OFFSET = 32
_ALL_TRIPLES = (4096, 4096)
_CHUNK = 1 << 20
_FLOAT_SENTINEL = 0x7FC00000
_BYTE_SENTINEL = 0xA5
_WEIGHTS = (0.2989, 0.5870, 0.1140)
_WEIGHTS_PER_10000 = numpy.array([2989, 5870, 1140], dtype=numpy.int64)
_STREAMS = {"vector_add": 0, "rgb_to_gray": 1}
_KERNELS = {
    "vector_add": {
        "name": "vector_add",
        "axes": 1,
        "buffers": ["out", "a", "b", "n"],
        "scalar_buffers": [3],
        "bytes_per_item": 12,
    },
    "rgb_to_gray": {
        "name": "rgb_to_gray",
        "axes": 2,
        "buffers": ["out", "rgb", "width", "height"],
        "scalar_buffers": [2, 3],
        "bytes_per_item": 4,
    },
}
_OUTPUTS = (("metal_kernel", "mps"), ("torch", "mps"), ("torch", "cpu"))
_SIDES = ("metal_kernel", "torch")
_PROBE = {
    "expression": "x[0] * x[1] + x[2]",
    "inputs": [1.0 + 2.0**-12, 1.0 + 2.0**-12, -(1.0 + 2.0**-11)],
    "unfused_result": 0.0,
    "fused_result": 2.0**-24,
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


class KernelMetalVerificationError(KernelMetalError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KernelMetalVerificationError(message)


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


def _vector_inputs(
    seed: int, index: int, length: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    generator = numpy.random.default_rng([seed, _STREAMS["vector_add"], index])
    first = generator.standard_normal(length, dtype=numpy.float32)
    return first, generator.standard_normal(length, dtype=numpy.float32)


def _image(seed: int, index: int, height: int, width: int, pattern: str):
    if pattern == "all_rgb_triples":
        codes = numpy.arange(1 << 24, dtype=numpy.uint32)
        channels = (codes >> 16, (codes >> 8) & 0xFF, codes & 0xFF)
        return numpy.stack(channels, axis=-1).astype(numpy.uint8).reshape(4096, 4096, 3)
    generator = numpy.random.default_rng([seed, _STREAMS["rgb_to_gray"], index])
    return generator.integers(0, 256, size=(height, width, 3), dtype=numpy.uint8)


def _gray(image: numpy.ndarray) -> numpy.ndarray:
    """The exact rule, written independently of the run's reference.

    The weighted sum times 10000 is an integer. Its quotient by 10000 gains
    one when the remainder is at least 5000: round half up, with no float.
    """

    flat = image.reshape(-1, 3)
    gray = numpy.empty(flat.shape[0], dtype=numpy.uint8)
    for start in range(0, flat.shape[0], _CHUNK):
        block = flat[start : start + _CHUNK].astype(numpy.int64)
        quotient, remainder = numpy.divmod(block @ _WEIGHTS_PER_10000, 10000)
        level = numpy.minimum(quotient + (remainder >= 5000), 255)
        gray[start : start + block.shape[0]] = level.astype(numpy.uint8)
    return gray


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


def _shaders(section: object) -> tuple[dict[str, dict[str, int]], bool]:
    _require(type(section) is dict, "shaders must be a JSON object")
    kernels = section.get("kernels")
    _require(
        type(kernels) is list and len(kernels) == len(_KERNELS),
        "shaders must list both kernels",
    )
    limits = {
        name: _limits(entry.get("limits") if type(entry) is dict else None)
        for name, entry in zip(_KERNELS, kernels)
    }
    probe = section.get("contraction_probe")
    observed = probe.get("observed") if type(probe) is dict else None
    _require(
        type(observed) is float
        and observed in (_PROBE["unfused_result"], _PROBE["fused_result"]),
        "the contraction probe must hold the fused or the unfused result",
    )
    expected = {
        "source": SHADER_SOURCE,
        "source_sha256": sha256(SHADER_SOURCE.encode("ascii")).hexdigest(),
        "kernels": [{**_KERNELS[name], "limits": limits[name]} for name in _KERNELS],
        "contraction_probe": {
            **_PROBE,
            "observed": observed,
            "fused": observed == _PROBE["fused_result"],
        },
    }
    _require(
        _exact(section.get("source"), SHADER_SOURCE),
        "the report was not made with this lab's Metal source",
    )
    _require(_exact(section, expected), "shader section does not recompute")
    return limits, observed == _PROBE["fused_result"]


def _vector_output(
    entry: object, output: tuple[str, str], digest: str, guard: int, label: str
) -> None:
    """Require one vector-add output to hash to the reference, bit for bit."""

    implementation, device = output
    expected = {
        "implementation": implementation,
        "device": device,
        "buffer_sha256": digest,
        "guard_elements": guard,
    }
    _require(
        _exact(entry, expected),
        f"{label}: {implementation} on {device} is not exact",
    )


def _vector_cases(
    section: object, config: KernelMetalConfig, limits: dict
) -> list[dict[str, object]]:
    lengths = config.correctness.vector_add_lengths
    _require(
        type(section) is list and len(section) == len(lengths),
        "vector_add cases do not cover the config",
    )
    group = [config.launch.vector_add_group_size]
    for index, (entry, length) in enumerate(zip(section, lengths)):
        label = f"vector_add case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        first, second = _vector_inputs(config.seed, index, length)
        reference = first + second
        launch = _launch([length], group, limits["vector_add"])
        guard = launch["padding_threads"]
        tail = numpy.full(guard, _FLOAT_SENTINEL, dtype=numpy.uint32)
        reference_digest = _hash(reference)
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(_OUTPUTS),
            f"{label} must record three outputs",
        )
        _vector_output(outputs[0], _OUTPUTS[0], _hash(reference, tail), guard, label)
        for output, name in zip(outputs[1:], _OUTPUTS[1:]):
            _vector_output(output, name, reference_digest, 0, label)
        expected = {
            "index": index,
            "length": length,
            "launch": launch,
            "a_sha256": _hash(first),
            "b_sha256": _hash(second),
            "reference_sha256": reference_digest,
            "outputs": outputs,
        }
        _require(
            _exact(entry, expected),
            f"{label} does not rebuild from the seed",
        )
    return section


def _gray_output(
    entry: object,
    implementation: str,
    device: str,
    reference: numpy.ndarray,
    guard: int,
    label: str,
) -> int:
    """Rebuild one grayscale output from its mismatches; return its max difference."""

    _require(type(entry) is dict, f"{label}: an output must be a JSON object")
    mismatches = entry.get("mismatches")
    _require(
        type(mismatches) is list and len(mismatches) <= _MAX_MISMATCHES,
        f"{label}: mismatches must be a list of at most {_MAX_MISMATCHES}",
    )
    rebuilt = reference.copy()
    previous, largest = -1, 0
    for item in mismatches:
        _require(
            type(item) is list
            and len(item) == 2
            and all(type(value) is int for value in item),
            f"{label}: mismatches must be [index, level] integer pairs",
        )
        index, level = item
        _require(
            previous < index < reference.size and 0 <= level <= 255,
            f"{label}: mismatch indices must increase inside the image",
        )
        difference = abs(level - int(reference[index]))
        _require(difference > 0, f"{label}: a recorded mismatch matches the reference")
        largest = max(largest, difference)
        rebuilt[index] = level
        previous = index
    tail = numpy.full(guard, _BYTE_SENTINEL, dtype=numpy.uint8)
    expected = {
        "implementation": implementation,
        "device": device,
        "buffer_sha256": _hash(rebuilt, tail),
        "guard_elements": guard,
        "mismatches": mismatches,
        "mismatched_pixels": len(mismatches),
        "max_abs_difference_levels": largest,
    }
    _require(
        _exact(entry, expected),
        f"{label}: {implementation} on {device} does not rebuild from its "
        "mismatches and digest",
    )
    _require(
        largest <= _BOUND_LEVELS,
        f"{label}: {implementation} on {device} exceeds the {_BOUND_LEVELS}-level "
        "bound",
    )
    return largest


def _gray_cases(
    section: object, config: KernelMetalConfig, limits: dict
) -> list[dict[str, object]]:
    plan = [(shape, "uniform") for shape in config.correctness.rgb_to_gray_shapes]
    if config.correctness.rgb_to_gray_all_triples:
        plan.append((_ALL_TRIPLES, "all_rgb_triples"))
    _require(
        type(section) is list and len(section) == len(plan),
        "rgb_to_gray cases do not cover the config",
    )
    group = list(config.launch.rgb_to_gray_group_size)
    for index, (entry, ((height, width), pattern)) in enumerate(zip(section, plan)):
        label = f"rgb_to_gray case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        image = _image(config.seed, index, height, width, pattern)
        reference = _gray(image)
        launch = _launch([width, height], group, limits["rgb_to_gray"])
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(_OUTPUTS),
            f"{label} must record three outputs",
        )
        for position, (output, (implementation, device)) in enumerate(
            zip(outputs, _OUTPUTS)
        ):
            guard = launch["padding_threads"] if position == 0 else 0
            _gray_output(output, implementation, device, reference, guard, label)
        expected = {
            "index": index,
            "pattern": pattern,
            "height": height,
            "width": width,
            "launch": launch,
            "rgb_sha256": _hash(image),
            "reference_sha256": _hash(reference),
            "bound_levels": _BOUND_LEVELS,
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


def _calls(bytes_per_call: int, target: int) -> int:
    calls = max(1, -(-target // bytes_per_call))
    _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
    return calls


def _side(
    entry: object, name: str, calls: int, per_call: int, config: KernelMetalConfig
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
    moved = per_call * calls
    return {
        "implementation": name,
        "calls_per_sample": calls,
        "warmup": config.benchmark.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": timing,
        "bytes_per_sample": moved,
        "median_ns_per_call": timing["median_ns"] / calls,
        "iqr_ns_per_call": timing["iqr_ns"] / calls,
        "median_gb_per_second": moved / timing["median_ns"],
        "best_gb_per_second": moved / timing["min_ns"],
    }


def _points(
    section: object,
    kernel: str,
    extents: list[list[int]],
    group: list[int],
    config: KernelMetalConfig,
    limits: dict,
) -> list[dict[str, object]]:
    _require(
        type(section) is list and len(section) == len(extents),
        f"{kernel} benchmark points do not cover the sweep",
    )
    rebuilt = []
    for entry, extent in zip(section, extents):
        _require(type(entry) is dict, f"{kernel} point must be a JSON object")
        per_call = _KERNELS[kernel]["bytes_per_item"] * math.prod(extent)
        calls = _calls(per_call, config.benchmark.target_bytes_per_sample)
        sides = {
            name: _side(entry.get(name), name, calls, per_call, config)
            for name in _SIDES
        }
        shape = (
            {"length": extent[0]}
            if kernel == "vector_add"
            else {"height": extent[1], "width": extent[0]}
        )
        expected = {
            "kernel": kernel,
            **shape,
            "launch": _launch(extent, group, limits[kernel]),
            "bytes_per_call": per_call,
            **sides,
            "torch_to_metal_time_ratio": sides["torch"]["timing"]["median_ns"]
            / sides["metal_kernel"]["timing"]["median_ns"],
        }
        _require(
            _exact(entry, expected),
            f"{kernel} point at {extent} does not recompute from its samples",
        )
        rebuilt.append(expected)
    return rebuilt


def _benchmark(
    section: object, config: KernelMetalConfig, limits: dict
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _require(type(section) is dict, "benchmark must be a JSON object")
    sweep = config.benchmark
    vector = _points(
        section.get("vector_add"),
        "vector_add",
        [[length] for length in sweep.vector_add_lengths],
        [config.launch.vector_add_group_size],
        config,
        limits,
    )
    gray = _points(
        section.get("rgb_to_gray"),
        "rgb_to_gray",
        [[width, height] for height, width in sweep.rgb_to_gray_shapes],
        list(config.launch.rgb_to_gray_group_size),
        config,
        limits,
    )
    expected = {
        "device": DEVICE,
        "torch_equivalents": TORCH_EQUIVALENTS,
        "synchronization": SYNCHRONIZATION,
        "vector_add": vector,
        "rgb_to_gray": gray,
    }
    _require(_exact(section, expected), "benchmark section does not recompute")
    return vector, gray


def _planned(config: KernelMetalConfig) -> dict[str, int]:
    sweep = config.benchmark
    samples = sweep.warmup + sweep.repeats
    items = [12 * length for length in sweep.vector_add_lengths] + [
        4 * height * width for height, width in sweep.rgb_to_gray_shapes
    ]
    batches = [_calls(per_call, sweep.target_bytes_per_sample) for per_call in items]
    cases = len(config.correctness.vector_add_lengths) + len(
        config.correctness.rgb_to_gray_shapes
    )
    return {
        "correctness_cases": cases + int(config.correctness.rgb_to_gray_all_triples),
        "benchmark_points": len(items),
        "timed_samples": 2 * len(items) * sweep.repeats,
        "calls": sum(2 * samples * batch for batch in batches),
        "accounted_bytes": sum(
            2 * samples * batch * per_call for batch, per_call in zip(batches, items)
        ),
    }


def _environment(environment: object, measured: bool) -> None:
    _require(
        type(environment) is dict and set(environment) == set(ENVIRONMENT_FIELDS),
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


def _largest(point: dict[str, object]) -> dict[str, object]:
    shape = {key: point[key] for key in ("length", "height", "width") if key in point}
    return {
        **shape,
        "metal_kernel_gb_per_second": point["metal_kernel"]["median_gb_per_second"],
        "torch_gb_per_second": point["torch"]["median_gb_per_second"],
        "torch_to_metal_time_ratio": point["torch_to_metal_time_ratio"],
    }


def _verify(report: object, supplied: KernelMetalConfig | None) -> dict[str, object]:
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
    limits, fused = _shaders(report["shaders"])
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    vector_cases = _vector_cases(correctness.get("vector_add"), config, limits)
    gray_cases = _gray_cases(correctness.get("rgb_to_gray"), config, limits)
    expected_correctness = {
        "reference": REFERENCE,
        "rounding": ROUNDING_RULE,
        "weights": list(_WEIGHTS),
        "bound_levels": _BOUND_LEVELS,
        "sentinels": {"float32_bits": _FLOAT_SENTINEL, "uint8": _BYTE_SENTINEL},
        "vector_add": vector_cases,
        "rgb_to_gray": gray_cases,
    }
    _require(
        _exact(correctness, expected_correctness),
        "correctness section does not match the lab's reference and bound",
    )
    vector, gray = _benchmark(report["benchmark"], config, limits)
    points = [*vector, *gray]
    timed = sum(sum(point[side]["samples_ns"]) for point in points for side in _SIDES)
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
    metal_outputs = [case["outputs"][0] for case in gray_cases]
    summary = {
        "label": label,
        "correctness_cases": len(vector_cases) + len(gray_cases),
        "vector_add_exact_cases": len(vector_cases),
        "rgb_to_gray_max_abs_difference_levels": max(
            output["max_abs_difference_levels"]
            for case in gray_cases
            for output in case["outputs"]
        ),
        "rgb_to_gray_metal_mismatched_pixels": sum(
            output["mismatched_pixels"] for output in metal_outputs
        ),
        "contraction_fused": fused,
        "benchmark_points": len(points),
        "timed_samples": 2 * len(points) * config.benchmark.repeats,
        "vector_add_largest": _largest(vector[-1]),
        "rgb_to_gray_largest": _largest(gray[-1]),
        "peak_metal_kernel_gb_per_second": max(
            point["metal_kernel"]["median_gb_per_second"] for point in points
        ),
    }
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
        "rgb_to_gray_max_abs_difference_levels": summary[
            "rgb_to_gray_max_abs_difference_levels"
        ],
        "rgb_to_gray_metal_mismatched_pixels": summary[
            "rgb_to_gray_metal_mismatched_pixels"
        ],
    }


def verify_report(
    report: object, *, config: KernelMetalConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``KernelMetalVerificationError``.
    """

    if config is not None and not isinstance(config, KernelMetalConfig):
        raise TypeError("config must be a KernelMetalConfig")
    try:
        return _verify(report, config)
    except KernelMetalVerificationError:
        raise
    except (
        KernelMetalError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise KernelMetalVerificationError(
            f"report failed verification: {error}"
        ) from error
