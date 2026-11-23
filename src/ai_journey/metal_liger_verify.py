"""Independent verification of sealed Day 80 reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
fact that the seed and the stored digests fix, with its own code: the
inputs of each case and point, the gapped layouts, the float64 reference of
each family by its declared algorithm, the bound of every value, each
launch, and each pipeline verdict from the recorded limits. An output whose
values all equal the rounded reference must hash to that reference plus its
sentinel gaps and must record its errors exactly. Any other output records
a digest of another buffer, a count of 1 to n differing values, and errors
within its bounds: its largest error must lie from its bound ratio times the
smallest bound to that ratio times the largest bound. The PyTorch checks
meet the same rule against the sanity bound that the verifier recomputes.

Besides the label and rule texts of the run, the verifier uses three
modules of the run that measure nothing: the config parser checks the
bounds, the kernel registry gives each kernel's text and threadgroup bytes,
and the declared plan gives the hypotheses that the report must seal, byte
for byte. Its exp is the Day 79 verifier's exp64, and its log is its own
copy of the declared log64, so a report verifies with the same bits on any
IEEE host. It derives each statistic, rate, model byte count, ratio,
verdict, ceiling fraction, held hypothesis, run total and summary from the
raw samples. Every section must match its rebuild as canonical JSON.

The CPU cannot recompute what only the GPU run saw. ``GPU_ONLY`` in the
report module names those fields: the raw samples, each output that differs
from the rounded reference, ``identical_repeats``, the probe results, the
pipeline limits, the PyTorch checks, the warnings, the switches and the load
averages. The verifier checks their form and their bounds only. A matching
digest shows internal consistency; it cannot prove that a timing was
observed or that a GPU ran.
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
from .metal_fused_ceiling import BandwidthCeiling
from .metal_fused_verify import MetalFusedVerificationError
from .metal_fused_verify import _ceiling as _ceiling_rate
from .metal_fused_verify import _environment
from .metal_fused_verify import _exp
from .metal_fused_verify import _probe as _exp_probe
from .metal_liger_benchmark import TRAFFIC_MODEL, UNWRITTEN_SOFTMAX
from .metal_liger_config import COMPULSORY, MetalLigerConfig, config_from_mapping
from .metal_liger_correctness import RULES
from .metal_liger_plan import PLAN, TORCH_SEQUENCES, plan_mapping, plan_sha256
from .metal_liger_reference import (
    BOUNDS,
    DISTRIBUTIONS,
    PROBE_ARGUMENTS,
    PROBE_RULES,
    REFERENCE,
    VENDOR_BOUNDS,
)
from .metal_liger_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    verification_scope,
)
from .metal_liger_source import KERNELS, MetalLigerError, library_sha256
from .metal_liger_speedup import FRACTION_RULE, HYPOTHESIS_RULE, RATIO_RULE

_NS = 1_000_000_000
_MAX_CALLS = 4096
_RMS, _SWI, _CE = "rmsnorm", "swiglu", "cross_entropy"
_FAMILIES = (_RMS, _SWI, _CE)
_CODES = {_RMS: 0, _SWI: 1, _CE: 2}
_KERNEL = {_RMS: "liger_rmsnorm", _SWI: "liger_swiglu", _CE: "liger_cross_entropy"}
_TORCH = {_RMS: "torch_rmsnorm", _SWI: "torch_swiglu", _CE: "torch_cross_entropy"}
_GAP = 32
_GAP_INPUT = 2.0**100
_SENTINEL = 0x7FC00000
_UNIT = 2.0**-24
_REF_UNIT = 2.0**-53
_EXP_ERROR = 4 * 2.0**-23
_RSQRT_ERROR = 2 * 2.0**-23
_LOG_ERROR = 4 * 2.0**-23
_FLUSH = 2.0**-126
_FLOOR = 2.0**-125
_TAIL = 2.0**-120
_VENDOR_UNIT = 2.0**-16
_VENDOR_FLOOR = 2.0**-100
# The rounding slack of err / bound and of ratio * bound in float64.
_SLACK = 1e-12
_EPS32 = float(numpy.float32(1.0e-6))
_CHUNK = 1 << 22
_PROBE_COUNT = 1 << 16
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_WIDE = "pipeline_simd_width_is_not_32"
_BYTES = "static_threadgroup_bytes_differ_from_the_text"
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_SQRT_HALF = 0.70710678118654757274
_ATANH = tuple(1.0 / (2 * order + 1) for order in range(14))
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
    "math_probes",
    "correctness",
    "benchmark",
    "speedup",
    "verification",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


class MetalLigerVerificationError(MetalLigerError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetalLigerVerificationError(message)


def _hash(array: numpy.ndarray, code: str) -> str:
    return sha256(numpy.ascontiguousarray(array).astype(code).tobytes()).hexdigest()


def _gamma(count: int, unit: float = _UNIT) -> float:
    product = count * unit
    return product / (1 - product)


def _growth(count: int) -> float:
    value = 1.0
    for _ in range(count):
        value *= 1 + _EXP_ERROR
    return value


def _log(values: numpy.ndarray) -> numpy.ndarray:
    """The declared log64: frexp to [sqrt(1/2), sqrt(2)), then 2 atanh(f)."""

    mantissa, exponent = numpy.frexp(numpy.asarray(values, dtype=numpy.float64))
    low = mantissa < _SQRT_HALF
    mantissa = numpy.where(low, mantissa * 2.0, mantissa)
    exponent = numpy.where(low, exponent - 1, exponent).astype(numpy.float64)
    f = (mantissa - 1.0) / (mantissa + 1.0)
    square = f * f
    series = numpy.full_like(f, _ATANH[-1])
    for term in reversed(_ATANH[:-1]):
        series = series * square + term
    return exponent * _LN2_HI + (exponent * _LN2_LO + 2.0 * f * series)


def _group(family: str, longest: int) -> int:
    size = 32
    while size * 16 < longest and size < 1024:
        size *= 2
    return size


class _Layout:
    """The verifier's own row layout: lengths, gap and the slots of each row."""

    def __init__(self, lengths: list[int], gap: int) -> None:
        self.lengths = numpy.asarray(lengths, dtype=numpy.int64)
        self.gap = gap
        self.rows = len(lengths)
        self.elements = int(self.lengths.sum())
        self.longest = int(self.lengths.max())
        self.packed = numpy.concatenate(([0], numpy.cumsum(self.lengths)[:-1]))
        steps = self.lengths + gap
        self.starts = numpy.concatenate(([0], numpy.cumsum(steps)[:-1]))
        self.slots = self.elements + gap * self.rows
        self.valid = numpy.arange(self.elements, dtype=numpy.int64) + numpy.repeat(
            self.starts - self.packed, self.lengths
        )

    def outputs(self, family: str) -> int:
        return self.rows + self.gap if family == _CE else self.slots

    def output_valid(self, family: str) -> numpy.ndarray:
        if family == _CE:
            return numpy.arange(self.rows, dtype=numpy.int64)
        return self.valid

    def mapping(self, family: str) -> dict[str, object]:
        return {
            "rows": self.rows,
            "elements": self.elements,
            "shortest": int(self.lengths.min()),
            "longest": self.longest,
            "group_size": _group(family, self.longest),
            "gap": self.gap,
            "buffer_elements": self.slots,
            "output_elements": self.outputs(family),
            "dense": bool(numpy.all(self.lengths == self.lengths[0])),
            "lengths_sha256": _hash(self.lengths, "<u4"),
        }

    def launch(self, family: str) -> dict[str, object]:
        size = _group(family, self.longest)
        threads = self.rows * size
        return {
            "extent": [threads],
            "threads": [threads],
            "group_size": [size],
            "groups": [self.rows],
            "threads_per_group": size,
            "launched_threads": threads,
            "active_threads": threads,
            "padding_threads": 0,
        }

    def chunks(self):
        first, total = 0, 0
        for row, length in enumerate(self.lengths.tolist()):
            if row > first and total + length > _CHUNK:
                yield first, row
                first, total = row, 0
            total += length
        yield first, self.rows


def _draw(generator, count: int, name: str) -> numpy.ndarray:
    if name == "normal":
        return generator.standard_normal(count, dtype=numpy.float32)
    if name == "uniform_80":
        return generator.uniform(-80.0, 80.0, count).astype(numpy.float32)
    offset = 1000.0 if name == "offset_plus_1000" else -1000.0
    return (offset + generator.standard_normal(count)).astype(numpy.float32)


def _gapped(values: numpy.ndarray, layout: _Layout) -> numpy.ndarray:
    if layout.gap == 0:
        return values
    buffer = numpy.full(layout.slots, _GAP_INPUT, dtype=numpy.float32)
    buffer[layout.valid] = values
    return buffer


def _inputs(seed: int, family: str, stream: int, index: int, layout, name: str):
    generator = numpy.random.default_rng([seed, _CODES[family], stream, index])
    main = _gapped(_draw(generator, layout.elements, name), layout)
    if family == _RMS:
        second = generator.standard_normal(layout.longest, dtype=numpy.float32)
    elif family == _SWI:
        values = generator.standard_normal(layout.elements, dtype=numpy.float32)
        second = _gapped(values, layout)
    else:
        second = generator.integers(0, layout.lengths, dtype=numpy.int64).astype(
            numpy.uint32
        )
    return main, second


def _digests(family: str, main, second) -> dict[str, str]:
    code = "<u4" if family == _CE else "<f4"
    return {"main_sha256": _hash(main, "<f4"), "second_sha256": _hash(second, code)}


def _rmsnorm(main, second, layout: _Layout):
    x = main[layout.valid].astype(numpy.float64)
    lengths, starts = layout.lengths, layout.packed
    columns = numpy.arange(layout.elements, dtype=numpy.int64) - numpy.repeat(
        starts, lengths
    )
    w = second.astype(numpy.float64)[columns]
    squares = numpy.add.reduceat(x * x, starts)
    inverse = 1.0 / numpy.sqrt(squares / lengths + _EPS32)
    output = x * numpy.repeat(inverse, lengths) * w
    size = _group(_RMS, layout.longest)
    rel = []
    for length in layout.lengths.tolist():
        roundings = -(-length // size) + 31 + (size // 32 - 1)
        g = _gamma(roundings + 2)
        c = 1.0 / math.sqrt(1.0 - g) - 1.0
        rel.append((1 + c) * (1 + _RSQRT_ERROR) * (1 + _UNIT) * (1 + _UNIT) - 1)
    ref = numpy.array([_gamma(length + 8, _REF_UNIT) for length in lengths])
    rel_each = numpy.repeat(numpy.array(rel), lengths)
    ref_each = numpy.repeat(ref, lengths)
    return output, (rel_each + ref_each) / (1 - ref_each) * numpy.abs(output) + _FLOOR


def _swiglu(main, second, layout: _Layout):
    a = main[layout.valid].astype(numpy.float64)
    b = second[layout.valid].astype(numpy.float64)
    output = numpy.empty(layout.elements, dtype=numpy.float64)
    for first, last in layout.chunks():
        low = int(layout.packed[first])
        high = layout.elements if last == layout.rows else int(layout.packed[last])
        part = a[low:high]
        negative = part < 0
        e = _exp(numpy.where(negative, part, -part))
        silu = numpy.where(negative, part * e / (1.0 + e), part / (1.0 + e))
        output[low:high] = silu * b[low:high]
    e = (1 + _EXP_ERROR) * (1 + _UNIT) - 1
    rel = (1 + e / (1 - e)) * (1 + _UNIT) * (1 + _UNIT) - 1
    ref = 16 * _REF_UNIT
    bound = (rel + ref) / (1 - ref) * numpy.abs(output)
    return output, bound + _TAIL * numpy.abs(b) + _FLOOR


def _cross_entropy(main, second, layout: _Layout):
    targets = second.astype(numpy.int64)
    size = _group(_CE, layout.longest)
    losses = numpy.empty(layout.rows, dtype=numpy.float64)
    bounds = numpy.empty(layout.rows, dtype=numpy.float64)
    for first, last in layout.chunks():
        low = int(layout.packed[first])
        high = layout.elements if last == layout.rows else int(layout.packed[last])
        x = main[layout.valid[low:high]].astype(numpy.float64)
        local = layout.packed[first:last] - low
        lengths = layout.lengths[first:last]
        top = numpy.maximum.reduceat(x, local)
        bottom = numpy.minimum.reduceat(x, local)
        sums = numpy.add.reduceat(_exp(x - numpy.repeat(top, lengths)), local)
        log_sum = _log(sums)
        margin = top - x[local + targets[first:last]]
        loss = margin + log_sum
        losses[first:last] = loss
        for offset, length in enumerate(lengths.tolist()):
            k = -(-length // size)
            calls = k + 2
            roundings = k + 1 + 31 + 1 + (size // 32 - 1)
            spread = float(top[offset] - bottom[offset])
            shift = float(_exp(numpy.array([calls * _UNIT * spread]))[0])
            rho = shift * _growth(calls) * (1 + _gamma(roundings)) - 1
            rho += (length + 2 * size) * _FLUSH
            _require(rho < 0.5, "a cross-entropy sum bound is too weak")
            t = rho / (1 - rho)
            inner = _UNIT * float(margin[offset]) + t
            inner += _LOG_ERROR * (float(log_sum[offset]) + t)
            value = float(loss[offset])
            ref = _gamma(length + 8, _REF_UNIT) * (1 + 2 * value)
            bounds[first + offset] = inner + _UNIT * (value + inner) + ref + _FLOOR
    return losses, bounds


def _reference(family: str, main, second, layout: _Layout):
    if family == _RMS:
        return _rmsnorm(main, second, layout)
    if family == _SWI:
        return _swiglu(main, second, layout)
    return _cross_entropy(main, second, layout)


def _number(value: object, low: float = 0.0, high: float = math.inf) -> bool:
    return type(value) is float and math.isfinite(value) and low <= value <= high


def _count(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _admission(name: str, limits: dict, required: int) -> dict[str, object]:
    declared = KERNELS[name].threadgroup_bytes
    need = required if KERNELS[name].family is not None else 256
    reason = None
    if limits["max_threads_per_threadgroup"] < need:
        reason = _FEWER
    elif limits["thread_execution_width"] != 32:
        reason = _WIDE
    elif limits["static_threadgroup_memory_bytes"] != declared:
        reason = _BYTES
    return {
        "kernel": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": need,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "thread_execution_width": limits["thread_execution_width"],
        "declared_threadgroup_bytes": declared,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
    }


def _shaders(section: object, required: int) -> None:
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
        admissions.append(_admission(name, limits, required))
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


def _probes(section: object) -> dict[str, dict[str, object]]:
    _require(
        type(section) is dict and set(section) == {"exp", "rsqrt", "log"},
        "math_probes must hold exp, rsqrt and log",
    )
    _exp_probe(section["exp"])
    for function, assumed in (("rsqrt", _RSQRT_ERROR), ("log", _LOG_ERROR)):
        entry = section[function]
        _require(type(entry) is dict, f"the {function} probe must be an object")
        expected = {
            "function": function,
            "count": _PROBE_COUNT,
            "arguments": PROBE_ARGUMENTS[function],
            "rule": PROBE_RULES[function],
            "assumed_error": assumed,
            "results_sha256": entry.get("results_sha256"),
            "max_relative_error": entry.get("max_relative_error"),
            "max_ulps": entry.get("max_ulps"),
            "exact_zero": True if function == "log" else None,
            "holds": True,
        }
        _require(
            _exact(entry, expected)
            and isinstance(entry["results_sha256"], str)
            and _SHA256.fullmatch(entry["results_sha256"]) is not None
            and _number(entry["max_relative_error"], 0.0, assumed)
            and _number(entry["max_ulps"]),
            f"the {function} probe does not meet its declared rule",
        )
    return section


def _vendor_bound(family: str, output: numpy.ndarray) -> numpy.ndarray:
    if family == _CE:
        return _VENDOR_UNIT * (output + 1.0)
    return _VENDOR_UNIT * numpy.abs(output) + _VENDOR_FLOOR


def _error_fits(error: float, ratio: float, bound: numpy.ndarray) -> bool:
    """Whether a largest error agrees with its bound ratio and the bounds.

    Each error is at most ``ratio`` times its own bound, and the value at the
    largest ratio has exactly that error, so the largest error lies from
    ``ratio * min(bound)`` to ``ratio * max(bound)``.
    """

    low = ratio * float(bound.min()) * (1 - _SLACK)
    high = ratio * float(bound.max()) * (1 + _SLACK)
    return low <= error <= high


def _output_record(entry, family, output, bound, layout, label) -> None:
    keys = {
        "buffer_sha256",
        "identical_repeats",
        "max_abs_error",
        "max_bound_ratio",
        "differing_from_rounded_reference",
    }
    size = output.size
    _require(
        type(entry) is dict
        and set(entry) == keys
        and isinstance(entry["buffer_sha256"], str)
        and _SHA256.fullmatch(entry["buffer_sha256"]) is not None
        and type(entry["identical_repeats"]) is bool
        and _number(entry["max_abs_error"])
        and _number(entry["max_bound_ratio"], 0.0, 1.0)
        and _count(entry["differing_from_rounded_reference"], 0, size)
        and _error_fits(entry["max_abs_error"], entry["max_bound_ratio"], bound),
        f"{label} output breaks its bound or its record form",
    )
    rounded = output.astype(numpy.float32)
    buffer = numpy.full(layout.outputs(family), _SENTINEL, dtype=numpy.uint32)
    buffer = buffer.view(numpy.float32)
    buffer[layout.output_valid(family)] = rounded
    matches = _hash(buffer, "<f4") == entry["buffer_sha256"]
    if entry["differing_from_rounded_reference"] == 0:
        error = numpy.abs(rounded.astype(numpy.float64) - output)
        _require(
            matches
            and _exact(entry["max_abs_error"], float(error.max()))
            and _exact(entry["max_bound_ratio"], float(numpy.max(error / bound))),
            f"{label} claims the rounded reference but does not rebuild it",
        )
    else:
        _require(not matches, f"{label} counts differences in the rounded reference")


def _cases(section: object, config: MetalLigerConfig) -> list[dict[str, object]]:
    cases = config.correctness.cases
    total = sum(len(group) for group in cases.values())
    _require(
        type(section) is list and len(section) == total,
        "correctness cases do not cover the config",
    )
    rebuilt = []
    entries = iter(section)
    for family in _FAMILIES:
        for index, case in enumerate(cases[family]):
            entry = next(entries)
            label = f"{family} case {index}"
            _require(type(entry) is dict, f"{label} must be a JSON object")
            layout = _Layout(list(case.lengths), _GAP)
            main, second = _inputs(config.seed, family, 0, index, layout, case.inputs)
            output, bound = _reference(family, main, second, layout)
            _output_record(entry.get("output"), family, output, bound, layout, label)
            naive = None
            if family == _CE:
                x = main[layout.valid]
                with numpy.errstate(all="ignore"):
                    sums = numpy.add.reduceat(
                        numpy.exp(x), layout.packed, dtype=numpy.float32
                    )
                    loss = numpy.log(sums) - x[layout.packed + second.astype(int)]
                naive = bool(numpy.all(numpy.isfinite(loss)))
            expected = {
                "family": family,
                "index": index,
                "case": case.to_mapping(),
                "layout": layout.mapping(family),
                "launch": layout.launch(family),
                "repeats": config.correctness.repeats,
                **_digests(family, main, second),
                "reference_sha256": _hash(output, "<f8"),
                "bound_sha256": _hash(bound, "<f8"),
                "naive_float32_finite": naive,
                "output": entry.get("output"),
            }
            _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
            rebuilt.append(expected)
    return rebuilt


def _compulsory(family: str, layout: _Layout) -> int:
    if family == _RMS:
        return 8 * layout.elements + 4 * layout.longest
    if family == _SWI:
        return 12 * layout.elements
    return 4 * layout.elements + 8 * layout.rows


def _modeled(name: str, family: str, layout: _Layout) -> int:
    n, r = layout.elements, layout.rows
    if name == _KERNEL[family]:
        return 4 * n + 16 * r if family == _CE else 12 * n + 8 * r
    return {_RMS: 40 * n + 20 * r, _SWI: 20 * n, _CE: 8 * n + 16 * r}[family]


def _side(entry: object, name: str, calls: int, compulsory: int, modeled, config):
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
        "modeled_gb_per_second": modeled * calls / timing["median_ns"],
    }


def _check(entry: object, name: str, size: int, output, bound) -> dict[str, object]:
    _require(
        type(entry) is dict
        and set(entry)
        == {"max_abs_error", "max_bound_ratio", "differing_from_rounded_reference"}
        and _number(entry["max_abs_error"])
        and _number(entry["max_bound_ratio"], 0.0, 1.0)
        and _count(entry["differing_from_rounded_reference"], 0, size)
        and _error_fits(entry["max_abs_error"], entry["max_bound_ratio"], bound),
        f"the check of {name} breaks its bound or its record form",
    )
    if entry["differing_from_rounded_reference"] == 0:
        error = numpy.abs(output.astype(numpy.float32).astype(numpy.float64) - output)
        _require(
            _exact(entry["max_abs_error"], float(error.max()))
            and _exact(entry["max_bound_ratio"], float(numpy.max(error / bound))),
            f"the check of {name} claims the rounded reference but does not match it",
        )
    return entry


def _points(section: object, config: MetalLigerConfig) -> list[dict[str, object]]:
    shapes = config.benchmark.shapes
    total = sum(len(group) for group in shapes.values())
    _require(
        type(section) is list and len(section) == total,
        "benchmark points do not cover the shapes",
    )
    rebuilt = []
    entries = iter(section)
    for family in _FAMILIES:
        for index, shape in enumerate(shapes[family]):
            entry = next(entries)
            label = f"{family} point {index}"
            _require(type(entry) is dict, f"{label} must be a JSON object")
            layout = _Layout([shape.cols] * shape.rows, 0)
            main, second = _inputs(
                config.seed, family, 1, index, layout, config.benchmark.inputs
            )
            output, bound = _reference(family, main, second, layout)
            compulsory = _compulsory(family, layout)
            calls = max(1, -(-config.benchmark.target_bytes_per_sample // compulsory))
            _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
            recorded, checks = entry.get("sides"), entry.get("checks")
            _require(
                type(recorded) is dict and type(checks) is dict,
                f"{label} sides and checks must be objects",
            )
            names = [_KERNEL[family], _TORCH[family]]
            size = output.size
            expected = {
                "family": family,
                "index": index,
                "shape": shape.to_mapping(),
                "layout": layout.mapping(family),
                "launch": layout.launch(family),
                **_digests(family, main, second),
                "reference_sha256": _hash(output, "<f8"),
                "compulsory_bytes_per_call": compulsory,
                "unwritten_softmax_bytes": 4 * layout.elements
                if family == _CE
                else None,
                "side_order": names,
                "sides": {
                    name: _side(
                        recorded.get(name),
                        name,
                        calls,
                        compulsory,
                        _modeled(name, family, layout),
                        config,
                    )
                    for name in names
                },
                "checks": {
                    names[0]: _check(
                        checks.get(names[0]), names[0], size, output, bound
                    ),
                    names[1]: _check(
                        checks.get(names[1]),
                        names[1],
                        size,
                        output,
                        _vendor_bound(family, output),
                    ),
                },
            }
            _require(
                _exact(entry, expected),
                f"{label} does not recompute from its seed and samples",
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


def _rate(entry: dict, ceiling: float) -> dict[str, object]:
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": list(entry["iqr_effective_gb_per_second"]),
        "modeled_gb_per_second": entry["modeled_gb_per_second"],
        "effective_fraction_of_ceiling": entry["effective_gb_per_second"] / ceiling,
        "modeled_fraction_of_ceiling": entry["modeled_gb_per_second"] / ceiling,
    }


def _speedup(points: list[dict], ceiling: float, minimum: float, version: str):
    predicted = {step.family: list(step.predicted) for step in PLAN}
    rows = []
    for point in points:
        family, sides = point["family"], point["sides"]
        kernel, torch = sides[_KERNEL[family]], sides[_TORCH[family]]
        to_torch = _ratio(kernel, torch, minimum)
        rows.append(
            {
                "family": family,
                "index": point["index"],
                "shape": dict(point["shape"]),
                "kernel": _rate(kernel, ceiling),
                "torch": _rate(torch, ceiling),
                "to_torch": to_torch,
                "predicted": predicted[family],
                "held": to_torch["verdict"] in predicted[family],
            }
        )
    return {
        "vendors": {
            family: f"{TORCH_SEQUENCES[family]}, PyTorch {version}"
            for family in _FAMILIES
        },
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "min_faster_ratio": minimum,
        "ceiling_gb_per_second": ceiling,
        "points": rows,
        "hypotheses": [
            {
                "family": family,
                "predicted": predicted[family],
                "held_at": [
                    row["index"]
                    for row in rows
                    if row["family"] == family and row["held"]
                ],
                "missed_at": [
                    row["index"]
                    for row in rows
                    if row["family"] == family and not row["held"]
                ],
            }
            for family in _FAMILIES
        ],
    }


def _planned(config: MetalLigerConfig) -> dict[str, int]:
    timing = config.benchmark
    samples = timing.warmup + timing.repeats
    calls = moved = points = 0
    for family in _FAMILIES:
        for shape in timing.shapes[family]:
            work = _compulsory(family, _Layout([shape.cols] * shape.rows, 0))
            batch = max(1, -(-timing.target_bytes_per_sample // work))
            calls += 2 * samples * batch
            moved += 2 * samples * batch * work
            points += 1
    cases = [case for group in config.correctness.cases.values() for case in group]
    return {
        "correctness_cases": len(cases),
        "correctness_elements": sum(sum(case.lengths) for case in cases),
        "correctness_launches": len(cases) * config.correctness.repeats,
        "benchmark_points": points,
        "sides_per_point": 2,
        "timed_samples": 2 * points * timing.repeats,
        "calls": calls,
        "compulsory_bytes": moved,
    }


def _summary(label, cases, probes, points, gains, captured, load) -> dict:
    groups = {
        family: [c for c in cases if c["family"] == family] for family in _FAMILIES
    }
    return {
        "label": label,
        "correctness_cases": len(cases),
        "max_bound_ratio": {
            family: max(case["output"]["max_bound_ratio"] for case in group)
            for family, group in groups.items()
        },
        "identical_repeats": {
            family: all(case["output"]["identical_repeats"] for case in group)
            for family, group in groups.items()
        },
        "naive_float32_failed_cases": [
            case["index"]
            for case in groups[_CE]
            if case["naive_float32_finite"] is False
        ],
        "probe_max_ulps": {name: probe["max_ulps"] for name, probe in probes.items()},
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "family": row["family"],
                "index": row["index"],
                "shape": dict(row["shape"]),
                "kernel_gb_per_second": row["kernel"]["effective_gb_per_second"],
                "torch_gb_per_second": row["torch"]["effective_gb_per_second"],
                "ratio": row["to_torch"]["ratio"],
                "verdict": row["to_torch"]["verdict"],
                "held": row["held"],
            }
            for row in gains["points"]
        ],
        "hypotheses": {
            record["family"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
            }
            for record in gains["hypotheses"]
        },
        "ceiling_gb_per_second": gains["ceiling_gb_per_second"],
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


def _torch_checks(points: list[dict]) -> None:
    for point in points:
        check = point["checks"][_TORCH[point["family"]]]
        _require(
            check["max_bound_ratio"] <= 1.0,
            "a PyTorch side exceeds its sanity bound",
        )


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
    ceiling_rate = _ceiling_rate(report["ceiling"], measured, report["environment"])
    if ceiling is not None:
        _require(
            _exact(report["ceiling"], ceiling.to_mapping()),
            "report was not made with the supplied ceiling report",
        )
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    _shaders(report["shaders"], config.required_threads())
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    probes = _probes(report["math_probes"])
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config)
    expected = {
        "reference": dict(REFERENCE),
        "bounds": dict(BOUNDS),
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
    _torch_checks(points)
    expected = {
        "device": "mps",
        "inputs": config.benchmark.inputs,
        "synchronization": SYNCHRONIZATION,
        "kernels": dict(_KERNEL),
        "torch_sequences": dict(TORCH_SEQUENCES),
        "compulsory": dict(COMPULSORY),
        "traffic_model": dict(TRAFFIC_MODEL),
        "unwritten_softmax": UNWRITTEN_SOFTMAX,
        "vendor_bounds": dict(VENDOR_BOUNDS),
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    version = report["environment"]["torch_version"]
    gains = _speedup(points, ceiling_rate, config.comparison.min_faster_ratio, version)
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
        label, cases, probes, points, gains, captured, report["host_load"]
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
        "hypotheses": summary["hypotheses"],
        "points": summary["points"],
        "ceiling_gb_per_second": summary["ceiling_gb_per_second"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object,
    *,
    config: MetalLigerConfig | None = None,
    ceiling: BandwidthCeiling | None = None,
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally against its inputs.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 or Day 79 check, is reported as
    ``MetalLigerVerificationError``.
    """

    if config is not None and not isinstance(config, MetalLigerConfig):
        raise TypeError("config must be a MetalLigerConfig")
    if ceiling is not None and not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    try:
        return _verify(report, config, ceiling)
    except MetalLigerVerificationError:
        raise
    except (MatmulTiledVerificationError, MetalFusedVerificationError) as error:
        raise MetalLigerVerificationError(str(error)) from error
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
        raise MetalLigerVerificationError(
            f"report failed verification: {error}"
        ) from error
