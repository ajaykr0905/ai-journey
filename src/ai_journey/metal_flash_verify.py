"""Independent verification of sealed Day 81 reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
fact that the seed and the stored digests fix, with its own code: the
inputs of each case and point, the gapped layouts, the float64 attention
reference by its declared algorithm, the bound of every value, each launch,
and each pipeline verdict from the recorded limits. An output whose values
all equal the rounded reference must hash to that reference plus its
sentinel gap rows and must record its errors exactly. Any other output
records a digest of another buffer, a count of 1 to n differing values, and
errors within its bounds: its largest error must lie from its bound ratio
times the smallest bound to that ratio times the largest bound. The PyTorch
checks meet the same rule against the sanity bound that the verifier
recomputes.

Besides the label and rule texts of the run, the verifier uses three
modules of the run that measure nothing: the config parser checks the
bounds, the kernel registry gives each kernel's text, group size and
threadgroup bytes, and the declared plan gives the hypotheses that the
report must seal, byte for byte. Its exp is the Day 79 verifier's exp64, so
a report verifies with the same bits on any IEEE host. It derives each
statistic, rate, model count, ratio, verdict, ceiling fraction, held
hypothesis, construction byte count, memory difference, run total and the
summary from the raw records. Every section must match its rebuild as
canonical JSON.

The CPU cannot recompute what only the GPU run saw. ``GPU_ONLY`` in the
report module names those fields: the raw samples, each output that differs
from the rounded reference, ``identical_repeats``, the probe results, the
pipeline limits, every allocator reading, the PyTorch checks, the warnings,
the switches and the load averages. The verifier checks their form and
their bounds only. A matching digest shows internal consistency; it cannot
prove that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import math
import re
from hashlib import sha256

import numpy

from .kernel_metal_report import (
    INJECTED_BACKEND,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .kernel_metal_runtime import BACKEND
from .matmul_tiled_report import SYNCHRONIZATION
from .matmul_tiled_verify import (
    MatmulTiledVerificationError,
    _digest,
    _exact,
    _limits,
    _load,
    _switches,
    _timing,
    _warnings,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER
from .metal_flash_benchmark import ARITHMETIC, TRAFFIC_MODEL
from .metal_flash_ceiling import COMPUTE_DEFINITION, FlashCeiling
from .metal_flash_config import COMPULSORY, MetalFlashConfig, config_from_mapping
from .metal_flash_correctness import RULES
from .metal_flash_memory import CONSTRUCTION, COUNTERS_CANNOT_SHOW, COUNTERS_MEASURE
from .metal_flash_plan import PLAN, TORCH_CALL, plan_mapping, plan_sha256
from .metal_flash_reference import BOUND, DISTRIBUTIONS, REFERENCE, VENDOR_BOUND
from .metal_flash_report import (
    CHECK_RULE,
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    verification_scope,
)
from .metal_flash_source import KERNELS, MetalFlashError, library_sha256
from .metal_flash_speedup import FRACTION_RULE, HYPOTHESIS_RULE, RATIO_RULE
from .metal_fused_ceiling import SOURCE as CEILING_SOURCE
from .metal_fused_verify import MetalFusedVerificationError
from .metal_fused_verify import _ceiling as _bandwidth
from .metal_fused_verify import _environment
from .metal_fused_verify import _exp
from .metal_fused_verify import _probe as _exp_probe

_NS = 1_000_000_000
_MAX_CALLS = 4096
_D = 64
_N0, _F1, _SDPA = "n0", "f1", "torch_sdpa"
_SIDES = (_N0, _F1, _SDPA)
_VARIANTS = (_N0, _F1)
_MODES = ("full", "causal")
_GROUPS = {"scores": 256, "softmax": 256, "pv": 64, "f1": 128}
_GAP = 3
_GAP_INPUT = 2.0**100
_SENTINEL = 0x7FC00000
_UNIT = 2.0**-24
_REF_UNIT = 2.0**-53
_EXP_ERROR = 4 * 2.0**-23
_FLUSH = 2.0**-126
_FLOOR = 2.0**-125
_VENDOR_UNIT = 2.0**-14
_VENDOR_FLOOR = 2.0**-100
_LOG_MAX = 88.72283905206835
_CHUNK_BYTES = 1 << 25
# The rounding slack of err / bound and of ratio * bound in float64.
_SLACK = 1e-12
_MAX_BYTES = 1 << 40
_COUNTERS = ("current_bytes", "driver_bytes")
_READINGS = ("before", "after_allocation", "after_call", "peak_during_call")
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_WIDE = "pipeline_simd_width_is_not_32"
_BYTES = "static_threadgroup_bytes_differ_from_the_text"
_SHA256 = re.compile(r"[0-9a-f]{64}")
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
    "plan",
    "environment",
    "torch_mps_switches",
    "host_load",
    "warnings",
    "shaders",
    "planned_work",
    "ceiling",
    "compute_ceiling",
    "exp_probe",
    "correctness",
    "benchmark",
    "memory_method",
    "speedup",
    "verification",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


class MetalFlashVerificationError(MetalFlashError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetalFlashVerificationError(message)


def _hash(array: numpy.ndarray, code: str) -> str:
    return sha256(numpy.ascontiguousarray(array).astype(code).tobytes()).hexdigest()


def _gamma(count: int, unit: float = _UNIT) -> float:
    product = count * unit
    return product / (1 - product)


def _product(base: float, count: int) -> float:
    value = 1.0
    for _ in range(count):
        value *= base
    return value


def _number(value: object, low: float = 0.0, high: float = math.inf) -> bool:
    return type(value) is float and math.isfinite(value) and low <= value <= high


def _count(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


class _Layout:
    """The verifier's own sequence layout."""

    def __init__(self, lengths: list[int], gap: int) -> None:
        self.lengths = lengths
        self.gap = gap
        steps = numpy.asarray(lengths, dtype=numpy.int64) + gap
        self.starts = numpy.concatenate(([0], numpy.cumsum(steps)[:-1]))
        self.tokens = sum(lengths)
        self.rows = self.tokens + gap * len(lengths)
        self.blocks = sum(-(-length // 32) for length in lengths)
        self.scores = sum(length * length for length in lengths)
        self.valid = numpy.concatenate(
            [start + numpy.arange(n) for start, n in zip(self.starts, lengths)]
        )

    def pairs(self, mode: str) -> int:
        if mode == "causal":
            return sum(n * (n + 1) // 2 for n in self.lengths)
        return self.scores

    def tile_loads(self, mode: str) -> int:
        total = 0
        for n in self.lengths:
            blocks = -(-n // 32)
            if mode == "causal":
                total += sum(-(-min(n, 32 * (b + 1)) // 32) for b in range(blocks))
            else:
                total += blocks * -(-n // 32)
        return total

    def mapping(self) -> dict[str, object]:
        lengths = numpy.asarray(self.lengths, dtype=numpy.uint32)
        return {
            "sequences": len(self.lengths),
            "tokens": self.tokens,
            "shortest": min(self.lengths),
            "longest": max(self.lengths),
            "gap": self.gap,
            "buffer_rows": self.rows,
            "score_elements": self.scores,
            "query_groups": self.tokens,
            "block_groups": self.blocks,
            "dense": len(set(self.lengths)) == 1,
            "lengths_sha256": _hash(lengths, "<u4"),
        }

    def launches(self, variant: str, mode: str) -> list[dict[str, object]]:
        suffix = "_causal" if mode == "causal" else ""
        if variant == _N0:
            steps = [
                (f"flash_n0_{step}{suffix}", self.tokens, _GROUPS[step])
                for step in ("scores", "softmax", "pv")
            ]
        else:
            steps = [(f"flash_f1{suffix}", self.blocks, _GROUPS["f1"])]
        return [
            {
                "kernel": name,
                "extent": [groups * size],
                "group_size": [size],
                "groups": [groups],
                "threads": [groups * size],
                "threads_per_group": size,
                "launched_threads": groups * size,
                "active_threads": groups * size,
                "padding_threads": 0,
            }
            for name, groups, size in steps
        ]


def _draw(seed: int, stream: int, index: int, layout: _Layout, name: str):
    generator = numpy.random.default_rng([seed, stream, index])
    count = layout.tokens * _D
    if name == "wide_scores":
        drawn = [
            (5.0 * generator.standard_normal(count)).astype(numpy.float32)
            for _ in range(2)
        ]
        drawn.append(generator.standard_normal(count, dtype=numpy.float32))
    else:
        drawn = [
            generator.standard_normal(count, dtype=numpy.float32) for _ in range(2)
        ]
        if name == "offset_values":
            drawn.append(
                (1000.0 + generator.standard_normal(count)).astype(numpy.float32)
            )
        else:
            drawn.append(generator.standard_normal(count, dtype=numpy.float32))
    if layout.gap == 0:
        return [values.reshape(-1, _D) for values in drawn]
    buffers = []
    for values in drawn:
        buffer = numpy.full((layout.rows, _D), _GAP_INPUT, dtype=numpy.float32)
        buffer[layout.valid] = values.reshape(-1, _D)
        buffers.append(buffer)
    return buffers


def _norms(rows: numpy.ndarray) -> numpy.ndarray:
    total = numpy.zeros(rows.shape[0], dtype=numpy.float64)
    for column in range(_D):
        total = total + rows[:, column] * rows[:, column]
    return numpy.sqrt(total)


def _attention(q, k, v, queries: numpy.ndarray, mode: str):
    """The declared float64 reference and bound of some query rows."""

    n = k.shape[0]
    tiles = -(-n // 32)
    q64, k64, v64 = (array.astype(numpy.float64) for array in (q, k, v))
    norms = _norms(k64)
    size = numpy.abs(v64)
    if mode == "causal":
        norm_limit = numpy.maximum.accumulate(norms)[queries]
        value_limit = numpy.maximum.accumulate(size, axis=0)[queries]
    else:
        norm_limit = numpy.full(queries.size, norms.max())
        value_limit = numpy.repeat(size.max(axis=0)[None], queries.size, 0)
    wide = numpy.concatenate((v64, size, numpy.ones((n, 1))), axis=1)
    step = max(1, _CHUNK_BYTES // (8 * n * wide.shape[1]))
    parts, tops = [], []
    for first in range(0, queries.size, step):
        chosen = queries[first : first + step]
        local = q64[chosen]
        scores = numpy.zeros((chosen.size, n), dtype=numpy.float64)
        for column in range(_D):
            scores = scores + local[:, column, None] * k64[None, :, column]
        scores = scores * 0.125
        seen = numpy.ones(scores.shape, dtype=bool)
        if mode == "causal":
            seen = numpy.arange(n)[None, :] <= chosen[:, None]
        top = numpy.where(seen, scores, -numpy.inf).max(axis=1)
        low = numpy.where(seen, scores, numpy.inf).min(axis=1)
        weights = numpy.where(
            seen, _exp(numpy.where(seen, scores - top[:, None], 0.0)), 0.0
        )
        sums = (weights[:, :, None] * wide[None, :, :]).sum(axis=1)
        parts.append((sums[:, :_D] / sums[:, -1:], sums[:, _D:-1] / sums[:, -1:]))
        tops.append((top, top - low))
    output = numpy.concatenate([part[0] for part in parts])
    mean = numpy.concatenate([part[1] for part in parts])
    spread = numpy.concatenate([top[1] for top in tops])
    products = 0.125 * _norms(q64[queries]) * norm_limit
    g = _gamma(_D + 2) * products
    r = _gamma(_D + 2, _REF_UNIT) * products
    shift = 2.0 * (g + r) + 2.0 * _UNIT * (spread + 2.0 * g)
    growth = _product(1 + _EXP_ERROR, tiles + 1) * _product(1.0 + _UNIT, 2 * tiles + 2)
    a = _exp(shift) * growth - 1.0
    e = (1.0 + a) * (1.0 + _gamma(n + 2 * tiles + 64)) - 1.0
    _require(bool(numpy.all(e < 0.5)), "the attention bound is too weak to use")
    relative = 2.0 * e / (1.0 - e) * (1.0 + _UNIT) + _UNIT
    relative = relative + 4.0 * _gamma(n + 70, _REF_UNIT)
    tail = 4.0 * (n + 2 * tiles + 2) * _FLUSH * value_limit + _FLOOR
    bound = mean * relative[:, None] + tail
    vendor = _VENDOR_UNIT * mean + _VENDOR_FLOOR
    top = float(numpy.concatenate([top[0] for top in tops]).max())
    return output, bound, vendor, top


def _queries(n: int, count: int | None) -> numpy.ndarray:
    if count is None or count >= n:
        return numpy.arange(n, dtype=numpy.int64)
    return numpy.unique(numpy.rint(numpy.linspace(0, n - 1, count))).astype(numpy.int64)


def _reference(layout: _Layout, buffers, mode: str, count: int | None):
    q, k, v = buffers
    rows, outputs, bounds, vendors, tops = [], [], [], [], []
    for start, n in zip(layout.starts, layout.lengths):
        span = slice(int(start), int(start) + n)
        queries = _queries(n, count)
        output, bound, vendor, top = _attention(
            q[span], k[span], v[span], queries, mode
        )
        rows.append(int(start) + queries)
        outputs.append(output)
        bounds.append(bound)
        vendors.append(vendor)
        tops.append(top)
    return (
        numpy.concatenate(rows),
        numpy.concatenate(outputs),
        numpy.concatenate(bounds),
        numpy.concatenate(vendors),
        max(tops),
    )


def _digests(buffers) -> dict[str, str]:
    return {
        f"{name}_sha256": _hash(buffer.reshape(-1), "<f4")
        for name, buffer in zip("qkv", buffers)
    }


def _admission(name: str, limits: dict) -> dict[str, object]:
    kernel = KERNELS[name]
    reason = None
    if limits["max_threads_per_threadgroup"] < kernel.group_size:
        reason = _FEWER
    elif limits["thread_execution_width"] != 32:
        reason = _WIDE
    elif limits["static_threadgroup_memory_bytes"] != kernel.threadgroup_bytes:
        reason = _BYTES
    return {
        "kernel": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": kernel.group_size,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "thread_execution_width": limits["thread_execution_width"],
        "declared_threadgroup_bytes": kernel.threadgroup_bytes,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
    }


def _shaders(section: object) -> None:
    kernels = section.get("kernels") if type(section) is dict else None
    _require(
        type(kernels) is list and len(kernels) == len(KERNELS),
        "the shader section must hold every kernel",
    )
    rebuilt, admissions = [], []
    for entry, (name, kernel) in zip(kernels, KERNELS.items()):
        _require(type(entry) is dict, "a kernel record must be a JSON object")
        limits = _limits(entry.get("limits"))
        rebuilt.append({**kernel.to_mapping(), "limits": limits})
        admissions.append(_admission(name, limits))
    expected = {
        "library_sha256": library_sha256(),
        "kernels": rebuilt,
        "admissions": admissions,
    }
    _require(_exact(section, expected), "shader section does not recompute")
    _require(
        all(entry["admitted"] for entry in admissions),
        "a report needs every pipeline admitted",
    )


def _error_fits(error: float, ratio: float, bound: numpy.ndarray) -> bool:
    """Whether a largest error agrees with its bound ratio and the bounds.

    Each error is at most ``ratio`` times its own bound, and the value at the
    largest ratio has exactly that error, so the largest error lies from
    ``ratio * min(bound)`` to ``ratio * max(bound)``.
    """

    low = ratio * float(bound.min()) * (1 - _SLACK)
    high = ratio * float(bound.max()) * (1 + _SLACK)
    return low <= error <= high


def _errors(entry: object, keys: set[str], size: int, bound, label: str) -> None:
    _require(
        type(entry) is dict
        and set(entry) == keys
        and _number(entry["max_abs_error"])
        and _number(entry["max_bound_ratio"], 0.0, 1.0)
        and _count(entry["differing_from_rounded_reference"], 0, size)
        and _error_fits(entry["max_abs_error"], entry["max_bound_ratio"], bound),
        f"{label} breaks its bound or its record form",
    )


def _rounded_errors(entry: dict, output, bound, label: str) -> None:
    error = numpy.abs(output.astype(numpy.float32).astype(numpy.float64) - output)
    _require(
        _exact(entry["max_abs_error"], float(error.max()))
        and _exact(entry["max_bound_ratio"], float(numpy.max(error / bound))),
        f"{label} claims the rounded reference but does not rebuild it",
    )


def _output(entry: object, layout: _Layout, rows, output, bound, label: str) -> None:
    keys = {
        "launches",
        "buffer_sha256",
        "identical_repeats",
        "max_abs_error",
        "max_bound_ratio",
        "differing_from_rounded_reference",
    }
    _errors(entry, keys, output.size, bound, label)
    _require(
        isinstance(entry["buffer_sha256"], str)
        and _SHA256.fullmatch(entry["buffer_sha256"]) is not None
        and type(entry["identical_repeats"]) is bool,
        f"{label} must hold a buffer digest and the repeat flag",
    )
    buffer = numpy.full((layout.rows, _D), _SENTINEL, dtype=numpy.uint32)
    buffer = buffer.view(numpy.float32)
    buffer[rows] = output.astype(numpy.float32)
    matches = _hash(buffer.reshape(-1), "<f4") == entry["buffer_sha256"]
    if entry["differing_from_rounded_reference"] == 0:
        _require(matches, f"{label} claims the rounded reference but hashes otherwise")
        _rounded_errors(entry, output, bound, label)
    else:
        _require(not matches, f"{label} counts differences in the rounded reference")


def _cases(section: object, config: MetalFlashConfig) -> list[dict[str, object]]:
    cases = config.correctness.cases
    _require(
        type(section) is list and len(section) == len(cases),
        "correctness cases do not cover the config",
    )
    rebuilt = []
    for index, (entry, case) in enumerate(zip(section, cases)):
        label = f"case {index}"
        _require(
            type(entry) is dict and type(entry.get("modes")) is dict,
            f"{label} must be a JSON object with its modes",
        )
        layout = _Layout(list(case.lengths), _GAP)
        buffers = _draw(config.seed, 0, index, layout, case.inputs)
        modes = {}
        for mode in _MODES:
            recorded = entry["modes"].get(mode)
            _require(type(recorded) is dict, f"{label} must record the {mode} mode")
            outputs = recorded.get("outputs")
            _require(type(outputs) is dict, f"{label} {mode} must record its outputs")
            rows, output, bound, _, top = _reference(layout, buffers, mode, None)
            for variant in _VARIANTS:
                _output(
                    outputs.get(variant),
                    layout,
                    rows,
                    output,
                    bound,
                    f"{label} {mode} {variant}",
                )
            modes[mode] = {
                "reference_sha256": _hash(output, "<f8"),
                "bound_sha256": _hash(bound, "<f8"),
                "max_score": top,
                "unscaled_exp_overflows": not top < _LOG_MAX,
                "outputs": {
                    variant: {
                        **outputs[variant],
                        "launches": layout.launches(variant, mode),
                    }
                    for variant in _VARIANTS
                },
            }
        expected = {
            "index": index,
            "case": case.to_mapping(),
            "layout": layout.mapping(),
            "repeats": config.correctness.repeats,
            **_digests(buffers),
            "modes": modes,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
        rebuilt.append(expected)
    return rebuilt


def _modeled(side: str, layout: _Layout, mode: str) -> int | None:
    if side == _N0:
        return 16 * _D * layout.tokens + 16 * layout.pairs(mode)
    if side == _F1:
        return 8 * _D * layout.tokens + 2 * 32 * _D * 4 * layout.tile_loads(mode)
    return None


def _side(
    entry: object, name: str, calls: int, compulsory: int, modeled, flops, config
):
    repeats = config.benchmark.repeats
    samples = entry.get("samples_ns") if type(entry) is dict else None
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{name} samples_ns must hold {repeats} positive integers",
    )
    timing = _timing(samples)
    moved = compulsory * calls
    work = flops * calls
    return {
        "implementation": name,
        "calls_per_sample": calls,
        "warmup": config.benchmark.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": timing,
        "median_ns_per_call": timing["median_ns"] / calls,
        "iqr_ns_per_call": timing["iqr_ns"] / calls,
        "compulsory_bytes_per_call": compulsory,
        "modeled_bytes_per_call": modeled,
        "effective_gb_per_second": moved / timing["median_ns"],
        "iqr_effective_gb_per_second": [
            moved / timing["p75_ns"],
            moved / timing["p25_ns"],
        ],
        "modeled_gb_per_second": None
        if modeled is None
        else modeled * calls / timing["median_ns"],
        "flops_per_call": flops,
        "gflops_per_second": work / timing["median_ns"],
        "iqr_gflops_per_second": [work / timing["p75_ns"], work / timing["p25_ns"]],
    }


def _check(entry: object, name: str, output, bound) -> dict[str, object]:
    keys = {"max_abs_error", "max_bound_ratio", "differing_from_rounded_reference"}
    _errors(entry, keys, output.size, bound, f"the check of {name}")
    if entry["differing_from_rounded_reference"] == 0:
        _rounded_errors(entry, output, bound, f"the check of {name}")
    return entry


def _construction(side: str, layout: _Layout) -> dict[str, int | None]:
    tensors = 16 * _D * layout.rows
    count = len(layout.lengths)
    if side == _N0:
        index = 4 * (3 * count + 2 * layout.tokens)
        return {"tensors": tensors, "scratch": 4 * layout.scores, "index": index}
    if side == _F1:
        index = 4 * (2 * count + 2 * layout.blocks)
        return {"tensors": tensors, "scratch": 0, "index": index}
    return {"tensors": tensors, "scratch": None, "index": 0}


def _memory(section: object, layout: _Layout) -> dict[str, object]:
    _require(
        type(section) is dict and set(section) == set(_SIDES),
        "memory must cover the three sides in order",
    )
    rebuilt = {}
    for side in _SIDES:
        entry = section[side]
        readings = entry.get("readings") if type(entry) is dict else None
        _require(
            type(readings) is dict
            and set(readings) == {*_READINGS, "polls"}
            and _count(readings["polls"], 1, _MAX_BYTES)
            and all(
                type(readings[name]) is dict
                and set(readings[name]) == set(_COUNTERS)
                and all(
                    _count(readings[name][counter], 0, _MAX_BYTES)
                    for counter in _COUNTERS
                )
                for name in _READINGS
            ),
            f"the memory readings of {side} do not match the form",
        )
        built = _construction(side, layout)
        rebuilt[side] = {
            "readings": readings,
            "construction": CONSTRUCTION[side],
            "construction_bytes": built,
            "known_construction_bytes": sum(
                value for value in built.values() if value is not None
            ),
            "allocated_current_bytes": readings["after_allocation"]["current_bytes"]
            - readings["before"]["current_bytes"],
            "peak_current_over_allocation_bytes": readings["peak_during_call"][
                "current_bytes"
            ]
            - readings["after_allocation"]["current_bytes"],
            "peak_driver_over_before_bytes": readings["peak_during_call"][
                "driver_bytes"
            ]
            - readings["before"]["driver_bytes"],
        }
    return rebuilt


def _points(section: object, config: MetalFlashConfig) -> list[dict[str, object]]:
    timing = config.benchmark
    points = timing.points()
    _require(
        type(section) is list and len(section) == len(points),
        "benchmark points do not cover the shapes and modes",
    )
    rebuilt = []
    for entry, point in zip(section, points):
        label = f"point {point.index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        shape, mode = point.shape, point.mode
        layout = _Layout([shape.length] * shape.sequences, 0)
        buffers = _draw(config.seed, 1, point.index, layout, timing.inputs)
        rows, output, bound, vendor, _ = _reference(
            layout, buffers, mode, timing.check_rows
        )
        compulsory = 16 * _D * layout.tokens
        calls = max(1, -(-timing.target_bytes_per_sample // compulsory))
        _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
        flops = 4 * _D * layout.pairs(mode)
        recorded, checks = entry.get("sides"), entry.get("checks")
        _require(
            type(recorded) is dict and type(checks) is dict,
            f"{label} sides and checks must be objects",
        )
        expected = {
            "index": point.index,
            "sequences": shape.sequences,
            "length": shape.length,
            "mode": mode,
            "layout": layout.mapping(),
            "launches": {
                variant: layout.launches(variant, mode) for variant in _VARIANTS
            },
            **_digests(buffers),
            "check_rows": timing.check_rows,
            "reference_sha256": _hash(output, "<f8"),
            "pairs": layout.pairs(mode),
            "compulsory_bytes_per_call": compulsory,
            "side_order": list(_SIDES),
            "sides": {
                side: _side(
                    recorded.get(side),
                    side,
                    calls,
                    compulsory,
                    _modeled(side, layout, mode),
                    flops,
                    config,
                )
                for side in _SIDES
            },
            "checks": {
                side: _check(
                    checks.get(side), side, output, vendor if side == _SDPA else bound
                )
                for side in _SIDES
            },
            "memory": _memory(entry.get("memory"), layout),
        }
        _require(
            _exact(entry, expected),
            f"{label} does not recompute from its seed, samples and readings",
        )
        rebuilt.append(expected)
    return rebuilt


def _ratio(first: dict, second: dict, minimum: float) -> dict[str, object]:
    one, two = first["timing"], second["timing"]
    ratio = two["median_ns"] / one["median_ns"]
    ahead, behind = one["p75_ns"] < two["p25_ns"], two["p75_ns"] < one["p25_ns"]
    verdict = "no_clear_difference"
    if ratio >= minimum and ahead:
        verdict = "faster"
    elif 1 / ratio >= minimum and behind:
        verdict = "slower"
    return {
        "first": first["implementation"],
        "second": second["implementation"],
        "ratio": ratio,
        "iqr_separated": ahead or behind,
        "verdict": verdict,
    }


def _rate(entry: dict, bandwidth: float, fma: float) -> dict[str, object]:
    modeled = entry["modeled_gb_per_second"]
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": list(entry["iqr_effective_gb_per_second"]),
        "modeled_gb_per_second": modeled,
        "gflops_per_second": entry["gflops_per_second"],
        "iqr_gflops_per_second": list(entry["iqr_gflops_per_second"]),
        "median_ns_per_call": entry["median_ns_per_call"],
        "effective_fraction_of_bandwidth": entry["effective_gb_per_second"] / bandwidth,
        "modeled_fraction_of_bandwidth": None
        if modeled is None
        else modeled / bandwidth,
        "fraction_of_fma": entry["gflops_per_second"] / fma,
    }


def _speedup(points, bandwidth: float, fma: float, minimum: float, version: str):
    rows = []
    for point in points:
        sides = point["sides"]
        steps = []
        for step in PLAN:
            verdict = _ratio(sides[step.first], sides[step.base], minimum)
            steps.append(
                {
                    "step": step.step,
                    **verdict,
                    "predicted": list(step.predicted),
                    "held": verdict["verdict"] in step.predicted,
                }
            )
        rows.append(
            {
                "index": point["index"],
                "sequences": point["sequences"],
                "length": point["length"],
                "mode": point["mode"],
                "rates": {side: _rate(sides[side], bandwidth, fma) for side in _SIDES},
                "steps": steps,
            }
        )
    return {
        "vendor": f"{TORCH_CALL}, PyTorch {version}",
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "min_faster_ratio": minimum,
        "bandwidth_gb_per_second": bandwidth,
        "fma_gflops_per_second": fma,
        "points": rows,
        "hypotheses": [
            {
                "step": step.step,
                "predicted": list(step.predicted),
                "held_at": [row["index"] for row in rows if row["steps"][i]["held"]],
                "missed_at": [
                    row["index"] for row in rows if not row["steps"][i]["held"]
                ],
            }
            for i, step in enumerate(PLAN)
        ],
    }


def _planned(config: MetalFlashConfig) -> dict[str, int]:
    timing = config.benchmark
    samples = timing.warmup + timing.repeats
    calls = moved = 0
    points = timing.points()
    for point in points:
        work = 16 * _D * point.shape.sequences * point.shape.length
        batch = max(1, -(-timing.target_bytes_per_sample // work))
        calls += 3 * samples * batch
        moved += 3 * samples * batch * work
    cases = config.correctness.cases
    return {
        "correctness_cases": len(cases),
        "correctness_pairs": sum(sum(n * n for n in case.lengths) for case in cases),
        "correctness_launches": len(cases) * 4 * config.correctness.repeats,
        "benchmark_points": len(points),
        "sides_per_point": 3,
        "timed_samples": 3 * len(points) * timing.repeats,
        "calls": calls,
        "compulsory_bytes": moved,
    }


def _summary(label, cases, probe, points, gains, captured, load) -> dict:
    outputs = [
        (variant, case["modes"][mode]["outputs"][variant])
        for case in cases
        for mode in _MODES
        for variant in _VARIANTS
    ]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "max_bound_ratio": {
            variant: max(e["max_bound_ratio"] for name, e in outputs if name == variant)
            for variant in _VARIANTS
        },
        "identical_repeats": {
            variant: all(
                e["identical_repeats"] for name, e in outputs if name == variant
            )
            for variant in _VARIANTS
        },
        "unscaled_exp_overflow_cases": [
            case["index"]
            for case in cases
            if any(case["modes"][mode]["unscaled_exp_overflows"] for mode in _MODES)
        ],
        "probe_max_ulps": probe["max_ulps"],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "index": row["index"],
                "sequences": row["sequences"],
                "length": row["length"],
                "mode": row["mode"],
                "median_ms_per_call": {
                    side: row["rates"][side]["median_ns_per_call"] / 1e6
                    for side in _SIDES
                },
                "gflops_per_second": {
                    side: row["rates"][side]["gflops_per_second"] for side in _SIDES
                },
                "verdicts": {step["step"]: step["verdict"] for step in row["steps"]},
                "peak_driver_over_before_bytes": {
                    side: point["memory"][side]["peak_driver_over_before_bytes"]
                    for side in _SIDES
                },
            }
            for row, point in zip(gains["points"], points)
        ],
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
            }
            for record in gains["hypotheses"]
        },
        "bandwidth_gb_per_second": gains["bandwidth_gb_per_second"],
        "fma_gflops_per_second": gains["fma_gflops_per_second"],
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


def _compute_ceiling(section: object, bandwidth_report: str) -> float:
    _require(
        type(section) is dict
        and set(section)
        == {"source", "definition", "report_sha256", "fma_gflops_per_second"}
        and section["source"] == CEILING_SOURCE
        and section["definition"] == COMPUTE_DEFINITION
        and section["report_sha256"] == bandwidth_report
        and _number(section["fma_gflops_per_second"], 1e-300),
        "the FMA ceiling must come from the report of the bandwidth ceiling",
    )
    return section["fma_gflops_per_second"]


def _verify(report: object, supplied, ceiling) -> dict[str, object]:
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
        and report["config_sha256"] == config.fingerprint(),
        "config digest does not match the embedded config",
    )
    if supplied is not None:
        _require(
            supplied.fingerprint() == config.fingerprint(),
            "report was not made from the supplied config",
        )
    _require(
        _exact(report["plan"], {**plan_mapping(), "sha256": plan_sha256()}),
        "the report does not seal the declared plan and its hypotheses",
    )
    _environment(report["environment"], measured)
    bandwidth = _bandwidth(report["ceiling"], measured, report["environment"])
    fma = _compute_ceiling(
        report["compute_ceiling"], report["ceiling"]["report_sha256"]
    )
    if ceiling is not None:
        _require(
            _exact(report["ceiling"], ceiling.bandwidth.to_mapping())
            and _exact(report["compute_ceiling"], ceiling.compute_mapping()),
            "report was not made with the supplied ceiling report",
        )
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    _shaders(report["shaders"])
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    probe = report["exp_probe"]
    _exp_probe(probe)
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config)
    expected = {
        "reference": REFERENCE,
        "bound": BOUND,
        "distributions": dict(DISTRIBUTIONS),
        "gap": _GAP,
        "gap_input": _GAP_INPUT,
        "sentinel_float32_bits": _SENTINEL,
        "rules": RULES,
        "cases": cases,
    }
    _require(_exact(correctness, expected), "correctness section does not recompute")
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config)
    expected = {
        "device": "mps",
        "inputs": config.benchmark.inputs,
        "synchronization": SYNCHRONIZATION,
        "sides": list(_SIDES),
        "torch_call": TORCH_CALL,
        "compulsory": COMPULSORY,
        "arithmetic": ARITHMETIC,
        "traffic_model": dict(TRAFFIC_MODEL),
        "vendor_bound": VENDOR_BOUND,
        "check_rule": CHECK_RULE,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    _require(
        _exact(
            report["memory_method"],
            {
                "measures": COUNTERS_MEASURE,
                "cannot_show": COUNTERS_CANNOT_SHOW,
                "construction": dict(CONSTRUCTION),
            },
        ),
        "the memory method was changed",
    )
    version = report["environment"]["torch_version"]
    gains = _speedup(
        points, bandwidth, fma, config.comparison.min_faster_ratio, version
    )
    _require(
        _exact(report["speedup"], gains),
        "the speedup rows or the hypotheses do not recompute from the samples",
    )
    _require(
        _exact(report["verification"], verification_scope()),
        "the verification scope was changed",
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
        label, cases, probe, points, gains, captured, report["host_load"]
    )
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[REPORT_DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "plan_sha256": report["plan"]["sha256"],
        "ceiling_report_sha256": report["ceiling"]["report_sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "identical_repeats": summary["identical_repeats"],
        "hypotheses": summary["hypotheses"],
        "points": summary["points"],
        "bandwidth_gb_per_second": summary["bandwidth_gb_per_second"],
        "fma_gflops_per_second": summary["fma_gflops_per_second"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object,
    *,
    config: MetalFlashConfig | None = None,
    ceiling: FlashCeiling | None = None,
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally against its inputs.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 or Day 79 check, is reported as
    ``MetalFlashVerificationError``.
    """

    if config is not None and not isinstance(config, MetalFlashConfig):
        raise TypeError("config must be a MetalFlashConfig")
    if ceiling is not None and not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    try:
        return _verify(report, config, ceiling)
    except MetalFlashVerificationError:
        raise
    except (
        MatmulTiledVerificationError,
        MetalFusedVerificationError,
        MetalFlashError,
    ) as error:
        raise MetalFlashVerificationError(str(error)) from error
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
        StopIteration,
    ) as error:
        raise MetalFlashVerificationError(
            f"report failed verification: {error}"
        ) from error
