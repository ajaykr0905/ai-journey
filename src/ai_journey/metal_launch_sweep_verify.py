"""Independent verification of sealed launch-sweep reports, on the CPU alone.

A GPU run cannot be observed again, so the verifier rebuilds every fact that
follows from the config, the recorded pipeline limits and the seed, with its
own code:

1. It admits or rejects each shape again from the recorded limits and the
   runtime's SIMD rule, and it rebuilds each SIMD footprint.
2. It rebuilds the mapping counts and the exact product of the simulated
   case, and the words that the layout probe must have returned.
3. It redraws the matmul inputs, checks their digests, and holds each
   recorded lattice value of ``C`` to its own float64 reference and bound.
4. It redraws the vector inputs and rebuilds every vector-add digest, with
   the sentinel tail of each Metal buffer.
5. It recomputes the batch sizes, statistics, rates, ranks, spreads and
   totals from the raw samples, and the planned work from the config.

A matmul output depends on the GPU's rounding, so the verifier cannot
rebuild its digest; the lattice values stand in for it. Timings cannot be
observed again either, so they are checked for structure and arithmetic
only. Every section must match the rebuilt one as canonical JSON, so ``1``,
``1.0`` and ``true`` stay different. The verifier shares the config parser,
which only checks bounds, and the Day 73 verifier's digest, statistics and
vector-case helpers. It calls no module that made the report and never
imports torch, so a host without torch or MPS can verify a report.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement. A matching digest shows internal consistency; it
cannot prove that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

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
from .kernel_metal_source import SHADER_SOURCE as DAY_73_SOURCE
from .kernel_metal_verify import _hash, _launch, _timing, _vector_inputs, _vector_output
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .memory_access_plan import canonical_json_bytes
from .memory_access_report import DIGEST_FIELD
from .metal_launch_sweep_config import MetalLaunchSweepConfig, config_from_mapping
from .metal_launch_sweep_report import (
    DEVICE,
    GRID_RULE,
    MEASUREMENT_BOUNDARY,
    REPORT_KIND,
    SYNCHRONIZATION,
    SYNTHETIC_BOUNDARY,
    TURN_ORDER,
)
from .metal_launch_sweep_shapes import MetalLaunchSweepError
from .metal_launch_sweep_source import SWEEP_SOURCE

_NS = 1_000_000_000
_MAX_CALLS = 4096
_OFFSET = 32
_MATMUL_STREAM = 2
_LATTICE = 5
# The run and the verifier sum the float64 reference of one element of C in
# different orders, so their errors can differ by 2 gamma_K(2**-53) |A| @ |B|,
# which is 2**-28 of the element's bound. A slack of 2**-20 of that bound
# covers the difference at every size; a fixed slack would exceed the whole
# error of a small case.
_REFERENCE_SLACK = 2.0**-20
_MAPPING_CASE = (13, 19, 3)
_FLOAT_SENTINEL = 0x7FC00000
_EXCEEDS = "exceeds_max_threads_per_threadgroup"
_NOT_WHOLE = "not_whole_simd_groups"
_MATMUL, _VECTOR, _PROBE = "matmul_naive", "vector_add", "simd_layout_probe"
_BOUND = "(gamma_K(2**-24) + gamma_K(2**-53)) * (|A| @ |B|)"
_REFERENCE = "float64 matmul on the CPU, rebuilt from the seed"
_KERNELS = [
    {
        "name": _MATMUL,
        "axes": 2,
        "buffers": ["c", "a", "b", "m", "n", "k"],
        "scalar_buffers": [3, 4, 5],
    },
    {"name": _PROBE, "axes": 2, "buffers": ["out"], "scalar_buffers": []},
]
_DAY_73_KERNEL = {
    "name": _VECTOR,
    "axes": 1,
    "buffers": ["out", "a", "b", "n"],
    "scalar_buffers": [3],
    "bytes_per_item": 12,
}
_OUTPUTS = (("metal_kernel", "mps"), ("torch", "mps"), ("torch", "cpu"))
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
    "pipelines",
    "dispatch",
    "admission",
    "mapping",
    "simd_layout",
    "planned_work",
    "correctness",
    "benchmark",
    "run",
    "summary",
    DIGEST_FIELD,
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MetalLaunchSweepVerificationError(MetalLaunchSweepError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetalLaunchSweepVerificationError(message)


def _exact(actual: object, expected: object) -> bool:
    """Compare JSON values by canonical bytes, so true, 1 and 1.0 stay distinct."""

    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _digest(report: dict[str, object]) -> str:
    unsigned = {key: value for key, value in report.items() if key != DIGEST_FIELD}
    return sha256(canonical_json_bytes(unsigned)).hexdigest()


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _limits(section: object) -> dict[str, dict[str, int]]:
    _require(
        type(section) is dict and set(section) == {_MATMUL, _VECTOR, _PROBE},
        "pipelines must list matmul_naive, vector_add and simd_layout_probe",
    )
    for name, entry in section.items():
        _require(
            type(entry) is dict
            and set(entry)
            == {
                "max_threads_per_threadgroup",
                "thread_execution_width",
                "static_threadgroup_memory_bytes",
            }
            and all(type(value) is int for value in entry.values()),
            f"{name} limits must be three integers",
        )
        most = entry["max_threads_per_threadgroup"]
        _require(
            1 <= most <= 1024
            and 1 <= entry["thread_execution_width"] <= most
            and 0 <= entry["static_threadgroup_memory_bytes"] <= 1 << 20,
            f"{name} limits are outside what the API can report",
        )
    return section


def _footprint(group: list[int], width: int) -> dict[str, int]:
    """Count the SIMD groups and the rows and columns that one of them spans."""

    columns_per_row = group[0]
    threads = math.prod(group)
    groups = _ceil(threads, width)
    rows = columns = 0
    for simd in range(groups):
        members = range(simd * width, min((simd + 1) * width, threads))
        rows = max(rows, len({index // columns_per_row for index in members}))
        columns = max(columns, len({index % columns_per_row for index in members}))
    return {
        "simd_groups": groups,
        "idle_lanes": groups * width - threads,
        "rows_per_simd_group": rows,
        "columns_per_simd_group": columns,
    }


def _admission(
    groups: list[list[int]], limits: dict[str, int], rule: bool
) -> list[dict[str, object]]:
    rebuilt = []
    for group in groups:
        threads = math.prod(group)
        width = limits["thread_execution_width"]
        reason = None
        if threads > limits["max_threads_per_threadgroup"]:
            reason = _EXCEEDS
        elif rule and threads % width:
            reason = _NOT_WHOLE
        rebuilt.append(
            {
                "group_size": group,
                "threads_per_group": threads,
                "admitted": reason is None,
                "reason": reason,
                **_footprint(group, width),
            }
        )
    return rebuilt


def _mapping_product() -> numpy.ndarray:
    """Rebuild the simulated case's exact product from its integer inputs."""

    m, n, k = _MAPPING_CASE
    rows, inner, columns = numpy.arange(m), numpy.arange(k), numpy.arange(n)
    a = (rows[:, None] + 2 * inner[None, :]) % 5 - 2
    b = (3 * inner[:, None] + columns[None, :]) % 7 - 3
    return (a @ b).astype(numpy.float32)


def _mapping(section: object, config: MetalLaunchSweepConfig, admitted: list) -> None:
    m, n, k = _MAPPING_CASE
    digest = _hash(_mapping_product())
    checks = []
    for group in admitted:
        gx, gy = group
        checks.append(
            {
                "group_size": group,
                "order_seed": config.seed,
                "launched_threads": _ceil(n, gx) * gx * _ceil(m, gy) * gy,
                "writer_threads": m * n,
                "reads": 2 * m * n * k,
                "writes": m * n,
                "out_of_bounds": 0,
                "write_conflicts": 0,
                "read_write_conflicts": 0,
                "output_sha256": digest,
                "exact": True,
            }
        )
    expected = {"case": list(_MAPPING_CASE), "checks": checks}
    _require(_exact(section, expected), "mapping checks do not rebuild")


def _layout_words(group: list[int], width: int) -> numpy.ndarray:
    linear = numpy.arange(math.prod(group), dtype=numpy.int64)
    words = numpy.stack(
        (
            linear // width,
            linear % width,
            linear % group[0],
            linear // group[0],
            numpy.full(linear.size, width),
        ),
        axis=1,
    )
    return words.astype("<u4")


def _layouts(section: object, admitted: list, limits: dict) -> None:
    probe_width = limits[_PROBE]["thread_execution_width"]
    expected = []
    for group, kernel in admitted:
        width = limits[kernel]["thread_execution_width"]
        _require(
            width == probe_width,
            "the probe pipeline must see the width of the admitting pipeline",
        )
        words = numpy.ascontiguousarray(_layout_words(group, width))
        expected.append(
            {
                "group_size": group,
                "layout": "linear_x_fastest",
                "simd_width": width,
                "observed_sha256": sha256(words.tobytes()).hexdigest(),
                **_footprint(group, width),
            }
        )
    _require(_exact(section, expected), "SIMD layout records do not rebuild")


def _matmul_inputs(
    seed: int, index: int, m: int, n: int, k: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    generator = numpy.random.default_rng([seed, _MATMUL_STREAM, index])
    a = generator.standard_normal((m, k), dtype=numpy.float32)
    return a, generator.standard_normal((k, n), dtype=numpy.float32)


def _lattice(extent: int) -> list[int]:
    last, steps = extent - 1, _LATTICE - 1
    return sorted({point * last // steps for point in range(_LATTICE)})


def _tolerance(k: int) -> float:
    return sum(k * unit / (1 - k * unit) for unit in (2.0**-24, 2.0**-53))


def _float32(value: float) -> bool:
    """Whether a finite float is exactly a value that a float32 buffer can hold."""

    with numpy.errstate(over="ignore"):
        return float(numpy.float32(value)) == value


def _accuracy(entry: object, a: numpy.ndarray, b: numpy.ndarray, label: str) -> None:
    """Hold the recorded lattice values of C to a float64 reference rebuilt here.

    C is a float32 buffer, so a lattice value with no float32 form, such as
    the float64 reference itself, cannot have come from the kernel.
    """

    _require(type(entry) is dict, f"{label}: accuracy must be a JSON object")
    samples = entry.get("samples")
    m, k = a.shape
    n = b.shape[1]
    positions = [[row, column] for row in _lattice(m) for column in _lattice(n)]
    _require(
        type(samples) is list
        and len(samples) == len(positions)
        and all(
            type(sample) is list
            and len(sample) == 3
            and sample[:2] == position
            and type(sample[2]) is float
            and math.isfinite(sample[2])
            and _float32(sample[2])
            for sample, position in zip(samples, positions)
        ),
        f"{label}: samples must be the float32 lattice values of C",
    )
    left, right = a.astype(numpy.float64), b.astype(numpy.float64)
    tolerance = _tolerance(k)
    # The recorded maxima span all of C, so they must reach every lattice
    # error, less the reference slack, and they can never be negative.
    least_error = least_ratio = 0.0
    for row, column, value in samples:
        reference = float(left[row] @ right[:, column])
        bound = tolerance * float(numpy.abs(left[row]) @ numpy.abs(right[:, column]))
        error = abs(value - reference)
        _require(error <= bound, f"{label}: C[{row}, {column}] exceeds the bound")
        least_error = max(least_error, error - _REFERENCE_SLACK * bound)
        if bound > 0:
            least_ratio = max(least_ratio, error / bound - _REFERENCE_SLACK)
    error, ratio = entry.get("max_abs_error"), entry.get("max_bound_ratio")
    _require(
        set(entry) == {"output_sha256", "max_abs_error", "max_bound_ratio", "samples"}
        and isinstance(entry["output_sha256"], str)
        and _SHA256.fullmatch(entry["output_sha256"]) is not None
        and type(error) is float
        and type(ratio) is float
        and least_error <= error < math.inf
        and least_ratio <= ratio <= 1.0,
        f"{label}: the error maxima do not cover the lattice or exceed the bound",
    )


def _matmul_cases(section: object, config, admitted: list) -> list[dict[str, object]]:
    cases = config.matmul.correctness_cases
    _require(
        type(section) is list and len(section) == len(cases),
        "matmul cases do not cover the config",
    )
    for index, (entry, (m, n, k)) in enumerate(zip(section, cases)):
        label = f"matmul case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        a, b = _matmul_inputs(config.seed, index, m, n, k)
        accuracy = {
            key: entry.get(key)
            for key in ("output_sha256", "max_abs_error", "max_bound_ratio", "samples")
        }
        _accuracy(accuracy, a, b, label)
        shapes = [
            {
                "group_size": [gx, gy],
                "guard_elements": _ceil(n, gx) * gx * _ceil(m, gy) * gy - m * n,
            }
            for gx, gy in admitted
        ]
        expected = {
            "index": index,
            "m": m,
            "n": n,
            "k": k,
            "a_sha256": _hash(a),
            "b_sha256": _hash(b),
            "shapes": shapes,
            **accuracy,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
    return section


def _vector_cases(
    section: object, config, admitted: list[tuple[int, int]], limits: dict
) -> list[dict[str, object]]:
    length = config.vector.correctness_length
    _require(
        type(section) is list and len(section) == len(admitted),
        "vector-add cases do not cover the admitted group sizes",
    )
    for entry, (index, size) in zip(section, admitted):
        label = f"vector_add case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        first, second = _vector_inputs(config.seed, index, length)
        reference = first + second
        launch = _launch([length], [size], limits[_VECTOR])
        guard = launch["padding_threads"]
        tail = numpy.full(guard, _FLOAT_SENTINEL, dtype=numpy.uint32)
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(_OUTPUTS),
            f"{label} must record three outputs",
        )
        _vector_output(outputs[0], _OUTPUTS[0], _hash(reference, tail), guard, label)
        for output, name in zip(outputs[1:], _OUTPUTS[1:]):
            _vector_output(output, name, _hash(reference), 0, label)
        expected = {
            "index": index,
            "length": length,
            "launch": launch,
            "a_sha256": _hash(first),
            "b_sha256": _hash(second),
            "reference_sha256": _hash(reference),
            "outputs": outputs,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
    return section


def _batch(per_call: int, target: int) -> int:
    calls = max(1, _ceil(target, per_call))
    _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
    return calls


def _point(
    entry: object,
    *,
    kernel: str,
    extent: dict[str, int],
    work: int,
    target: int,
    groups: list[list[int]],
    width: int,
    config: MetalLaunchSweepConfig,
) -> dict[str, object]:
    label = f"{kernel} point {list(extent.values())}"
    _require(type(entry) is dict, f"{label} must be a JSON object")
    calls = _batch(work, target)
    recorded = entry.get("shapes")
    repeats = config.timing.repeats
    _require(
        type(recorded) is list and len(recorded) == len(groups),
        f"{label} must time every admitted shape",
    )
    timings = []
    for shape in recorded:
        samples = shape.get("samples_ns") if type(shape) is dict else None
        _require(
            type(samples) is list
            and len(samples) == repeats
            and all(type(sample) is int and sample > 0 for sample in samples),
            f"{label}: samples_ns must hold {repeats} positive integers",
        )
        timings.append(_timing(samples))
    order = sorted(range(len(groups)), key=lambda at: (timings[at]["median_ns"], at))
    fastest, slowest = timings[order[0]], timings[order[-1]]
    rate = "gflops_per_second" if kernel == _MATMUL else "gb_per_second"
    shapes = []
    for position, (group, shape, timing) in enumerate(zip(groups, recorded, timings)):
        shapes.append(
            {
                "group_size": group,
                "threads_per_group": math.prod(group),
                **_footprint(group, width),
                "samples_ns": shape["samples_ns"],
                "timing": timing,
                "median_ns_per_call": timing["median_ns"] / calls,
                "iqr_ns_per_call": timing["iqr_ns"] / calls,
                f"median_{rate}": work * calls / timing["median_ns"],
                f"best_{rate}": work * calls / timing["min_ns"],
                "rank": order.index(position) + 1,
                "relative_to_fastest": timing["median_ns"] / fastest["median_ns"],
            }
        )
    expected = {
        "kernel": kernel,
        **extent,
        "work_per_call": work,
        "calls_per_sample": calls,
        "work_per_sample": work * calls,
        "warmup": config.timing.warmup,
        "repeats": repeats,
        "shapes": shapes,
        "ranking": [groups[at] for at in order],
        "spread": slowest["median_ns"] / fastest["median_ns"],
        "separated": fastest["p75_ns"] < slowest["p25_ns"],
        "output_sha256": entry.get("output_sha256"),
    }
    if kernel == _MATMUL:
        expected["accuracy"] = entry.get("accuracy")
    _require(_exact(entry, expected), f"{label} does not recompute from its samples")
    return expected


def _benchmark(
    section: object, config, matmul_groups: list, vector_groups: list, limits: dict
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _require(type(section) is dict, "benchmark must be a JSON object")
    sizes, lengths = config.matmul.sizes, config.vector.lengths
    matmul, vector = section.get("matmul"), section.get("vector")
    _require(
        type(matmul) is list
        and len(matmul) == len(sizes)
        and type(vector) is list
        and len(vector) == len(lengths),
        "benchmark points do not cover the sweep",
    )
    matmul_points = []
    for index, (entry, (m, n, k)) in enumerate(zip(matmul, sizes)):
        point = _point(
            entry,
            kernel=_MATMUL,
            extent={"m": m, "n": n, "k": k},
            work=2 * m * n * k,
            target=config.matmul.target_flops_per_sample,
            groups=matmul_groups,
            width=limits[_MATMUL]["thread_execution_width"],
            config=config,
        )
        a, b = _matmul_inputs(config.seed, _OFFSET + index, m, n, k)
        _accuracy(point["accuracy"], a, b, f"matmul point {index}")
        _require(
            point["output_sha256"] == point["accuracy"]["output_sha256"],
            f"matmul point {index}: the output digest is not the checked one",
        )
        matmul_points.append(point)
    vector_points = []
    for index, (entry, length) in enumerate(zip(vector, lengths)):
        point = _point(
            entry,
            kernel=_VECTOR,
            extent={"length": length},
            work=12 * length,
            target=config.vector.target_bytes_per_sample,
            groups=vector_groups,
            width=limits[_VECTOR]["thread_execution_width"],
            config=config,
        )
        first, second = _vector_inputs(config.seed, _OFFSET + index, length)
        _require(
            point["output_sha256"] == _hash(first + second),
            f"vector point {index}: the output is not the float32 sum",
        )
        vector_points.append(point)
    expected = {
        "device": DEVICE,
        "synchronization": SYNCHRONIZATION,
        "turn_order": TURN_ORDER,
        "matmul": matmul_points,
        "vector": vector_points,
    }
    _require(_exact(section, expected), "benchmark section does not recompute")
    return matmul_points, vector_points


def _planned(config: MetalLaunchSweepConfig) -> dict[str, int]:
    shapes = len(config.matmul.group_shapes)
    sizes = len(config.vector.group_sizes)
    samples = config.timing.warmup + config.timing.repeats
    matmul_calls = [
        shapes * samples * _batch(2 * m * n * k, config.matmul.target_flops_per_sample)
        for m, n, k in config.matmul.sizes
    ]
    vector_calls = [
        sizes * samples * _batch(12 * length, config.vector.target_bytes_per_sample)
        for length in config.vector.lengths
    ]
    return {
        "mapping_checks": shapes,
        "layout_probes": shapes + sizes,
        "correctness_launches": shapes * len(config.matmul.correctness_cases) + sizes,
        "benchmark_points": len(config.matmul.sizes) + len(config.vector.lengths),
        "timed_samples": config.timing.repeats
        * (shapes * len(config.matmul.sizes) + sizes * len(config.vector.lengths)),
        "calls": sum(matmul_calls) + sum(vector_calls),
        "matmul_flops": sum(
            calls * 2 * m * n * k
            for calls, (m, n, k) in zip(matmul_calls, config.matmul.sizes)
        ),
        "vector_bytes": sum(
            calls * 12 * length
            for calls, length in zip(vector_calls, config.vector.lengths)
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


def _load(value: object) -> bool:
    return value is None or (
        type(value) is list
        and len(value) == 3
        and all(
            type(item) is float and math.isfinite(item) and 0 <= item == round(item, 2)
            for item in value
        )
    )


def _summary_row(point: dict[str, object]) -> dict[str, object]:
    rate = (
        "median_gflops_per_second"
        if point["kernel"] == _MATMUL
        else "median_gb_per_second"
    )
    extent = {key: point[key] for key in ("m", "n", "k", "length") if key in point}
    return {
        **extent,
        "fastest": point["ranking"][0],
        "slowest": point["ranking"][-1],
        "spread": point["spread"],
        "separated": point["separated"],
        f"best_{rate}": max(shape[rate] for shape in point["shapes"]),
    }


def _verify(
    report: object, supplied: MetalLaunchSweepConfig | None
) -> dict[str, object]:
    _require(type(report) is dict, "report must be a JSON object")
    _require(set(report) == _FIELDS, "report fields do not match the schema")
    _require(
        type(report["schema_version"]) is int and report["schema_version"] == 1,
        "unsupported report schema",
    )
    _require(report["kind"] == REPORT_KIND, "unexpected report kind")
    _require(
        report[DIGEST_FIELD] == _digest(report),
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
    expected_shaders = {
        "source": SWEEP_SOURCE,
        "source_sha256": sha256(SWEEP_SOURCE.encode("ascii")).hexdigest(),
        "kernels": _KERNELS,
        "day_73_source_sha256": sha256(DAY_73_SOURCE.encode("ascii")).hexdigest(),
        "day_73_kernel": _DAY_73_KERNEL,
    }
    _require(
        _exact(report["shaders"], expected_shaders),
        "the report was not made with this lab's Metal sources",
    )
    limits = _limits(report["pipelines"])
    dispatch = report["dispatch"]
    rule = (
        dispatch.get("requires_whole_simd_groups") if type(dispatch) is dict else None
    )
    _require(type(rule) is bool, "the dispatch section must state the SIMD rule")
    _require(
        _exact(
            dispatch,
            {
                "grid": GRID_RULE,
                "requires_whole_simd_groups": rule,
                "rules": [_EXCEEDS, _NOT_WHOLE],
            },
        ),
        "dispatch section does not match the lab's rules",
    )
    _require(
        not measured or rule is False,
        "torch.mps.compile_shader pipelines do not require whole SIMD groups",
    )
    matmul_groups = [list(shape) for shape in config.matmul.group_shapes]
    vector_groups = [[size] for size in config.vector.group_sizes]
    admission = {
        "matmul": _admission(matmul_groups, limits[_MATMUL], rule),
        "vector": _admission(vector_groups, limits[_VECTOR], rule),
    }
    _require(
        _exact(report["admission"], admission),
        "admission does not follow from the recorded limits",
    )
    matmul_admitted = [
        entry["group_size"] for entry in admission["matmul"] if entry["admitted"]
    ]
    vector_admitted = [
        (index, entry["group_size"][0])
        for index, entry in enumerate(admission["vector"])
        if entry["admitted"]
    ]
    _require(
        bool(matmul_admitted) and bool(vector_admitted),
        "each sweep needs an admitted shape",
    )
    _mapping(report["mapping"], config, matmul_admitted)
    _layouts(
        report["simd_layout"],
        [(group, _MATMUL) for group in matmul_admitted]
        + [([size], _VECTOR) for _, size in vector_admitted],
        limits,
    )
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    matmul_cases = _matmul_cases(correctness.get("matmul"), config, matmul_admitted)
    vector_cases = _vector_cases(
        correctness.get("vector_add"), config, vector_admitted, limits
    )
    _require(
        _exact(
            correctness,
            {
                "reference": _REFERENCE,
                "bound": _BOUND,
                "matmul": matmul_cases,
                "vector_add": vector_cases,
            },
        ),
        "correctness section does not match the lab's reference and bound",
    )
    matmul, vector = _benchmark(
        report["benchmark"],
        config,
        matmul_admitted,
        [[size] for _, size in vector_admitted],
        limits,
    )
    points = [*matmul, *vector]
    timed = sum(
        sum(shape["samples_ns"]) for point in points for shape in point["shapes"]
    )
    run = report["run"]
    _require(
        type(run) is dict
        and set(run)
        == {
            "max_runtime_ns",
            "elapsed_ns",
            "timed_ns",
            "load_average_before",
            "load_average_after",
        }
        and _exact(run["max_runtime_ns"], config.timing.max_runtime_seconds * _NS)
        and _exact(run["timed_ns"], timed)
        and type(run["elapsed_ns"]) is int
        and timed <= run["elapsed_ns"] <= run["max_runtime_ns"]
        and _load(run["load_average_before"])
        and _load(run["load_average_after"]),
        "run time or load averages do not hold",
    )
    entries = [*admission["matmul"], *admission["vector"]]
    matmul_rows = [_summary_row(point) for point in matmul]
    vector_rows = [_summary_row(point) for point in vector]
    summary = {
        "label": label,
        "admitted_shapes": sum(entry["admitted"] for entry in entries),
        "rejected_shapes": sum(not entry["admitted"] for entry in entries),
        "mapping_checks": len(matmul_admitted),
        "layout_probes": len(matmul_admitted) + len(vector_admitted),
        "correctness_cases": len(matmul_cases) + len(vector_cases),
        "matmul_max_bound_ratio": max(
            [case["max_bound_ratio"] for case in matmul_cases]
            + [point["accuracy"]["max_bound_ratio"] for point in matmul]
        ),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(shape["samples_ns"]) for point in points for shape in point["shapes"]
        ),
        "matmul": matmul_rows,
        "vector": vector_rows,
        "best_matmul_gflops_per_second": max(
            row["best_median_gflops_per_second"] for row in matmul_rows
        ),
        "best_vector_gb_per_second": max(
            row["best_median_gb_per_second"] for row in vector_rows
        ),
    }
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "admitted_shapes": summary["admitted_shapes"],
        "rejected_shapes": summary["rejected_shapes"],
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "best_matmul_gflops_per_second": summary["best_matmul_gflops_per_second"],
        "best_vector_gb_per_second": summary["best_vector_gb_per_second"],
    }


def verify_report(
    report: object, *, config: MetalLaunchSweepConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``MetalLaunchSweepVerificationError``.
    """

    if config is not None and not isinstance(config, MetalLaunchSweepConfig):
        raise TypeError("config must be a MetalLaunchSweepConfig")
    try:
        return _verify(report, config)
    except MetalLaunchSweepVerificationError:
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
        raise MetalLaunchSweepVerificationError(
            f"report failed verification: {error}"
        ) from error
