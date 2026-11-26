"""Independent verification of sealed Day 82 reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
fact that the seed and the stored digests fix, with its own code: the
float32 draws of each case and size, their half and bfloat roundings, the
float64 reference by its declared k order, the scale ``|A| @ |B|``, the
row and column sums, the bound of every output in each input type, each
launch and each pipeline verdict from the recorded limits. Every recorded
error must agree with the rebuilt bounds and scale: its largest error lies
from its bound ratio times the smallest bound to that ratio times the
largest bound, and its largest share of the scale lies between the largest
error over the largest scale and the ratio times the largest bound over
scale. A claim of equal bits must agree with the digests and the errors.

Besides the label and rule texts of the run, the verifier uses three
modules of the run that measure nothing: the config parser checks the
bounds, the kernel registry gives each kernel's text, group size and
threadgroup bytes, and the declared plan gives the hypotheses that the
report must seal, byte for byte. It derives each statistic, rate,
intensity, roofline, fraction, ratio, verdict, held hypothesis, run total
and the summary from the raw records. Every section must match its rebuild
as canonical JSON.

The CPU cannot recompute what only the GPU run saw. ``GPU_ONLY`` in the
report module names those fields: the raw samples, each kernel output
digest and its errors, ``identical_repeats`` and the equal-bits flags, the
Metal language version and the support records, the pipeline limits, the
PyTorch checks, the warnings, the switches and the load averages. The
verifier checks their form, their bounds and their agreement only. A
matching digest shows internal consistency; it cannot prove that a timing
was observed or that a GPU ran.
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
from .metal_flash_ceiling import FlashCeiling
from .metal_flash_verify import MetalFlashVerificationError
from .metal_flash_verify import _compute_ceiling as _flash_compute_ceiling
from .metal_fused_verify import MetalFusedVerificationError
from .metal_fused_verify import _ceiling as _bandwidth
from .metal_fused_verify import _environment
from .metal_tc_benchmark import ARITHMETIC, COMPULSORY
from .metal_tc_ceiling import ROOFLINE_RULE
from .metal_tc_config import MetalTcConfig, config_from_mapping
from .metal_tc_correctness import RULES
from .metal_tc_plan import PLAN, TORCH_CALL, plan_mapping, plan_sha256
from .metal_tc_reference import (
    BOUND,
    DISTRIBUTION,
    REFERENCE,
    RELATIVE_ERROR,
    VENDOR_BOUND,
)
from .metal_tc_report import (
    CHECK_RULE,
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    verification_scope,
)
from .metal_tc_runtime import SUPPORT_REASONS, SUPPORT_RULE
from .metal_tc_source import KERNELS, MetalTcError, library_sha256
from .metal_tc_speedup import FRACTION_RULE, HYPOTHESIS_RULE, RATIO_RULE

_NS = 1_000_000_000
_MAX_CALLS = 4096
_TILE, _GROUP, _TAIL = 32, 128, 64
_GUARD = _SENTINEL = 0x7FC00000
_F32, _F16, _BF16 = "float32", "float16", "bfloat16"
_TYPES = (_F32, _F16, _BF16)
_S1, _M1, _M2, _M3 = "s1", "m1", "m2", "m3"
_VARIANTS = (_S1, _M1, _M2, _M3)
_T32, _T16, _TBF = "torch_float32", "torch_float16", "torch_bfloat16"
_SIDES = (*_VARIANTS, _T32, _T16, _TBF)
_OPTIONAL = (_M3, _TBF)
_KIND = {
    _S1: _F32,
    _M1: _F32,
    _M2: _F16,
    _M3: _BF16,
    _T32: _F32,
    _T16: _F16,
    _TBF: _BF16,
}
_KERNEL = {
    _S1: "tc_s1_float",
    _M1: "tc_m1_float",
    _M2: "tc_m2_half",
    _M3: "tc_m3_bfloat",
}
_BYTES = {_F32: 4, _F16: 2, _BF16: 2}
_UNIT = {_F32: 0.0, _F16: 2.0**-11, _BF16: 2.0**-8}
_FLOOR = {_F32: 2.0**-126, _F16: 2.0**-14, _BF16: 2.0**-126}
_VENDOR = {_F32: 2.0**-12, _F16: 2.0**-6, _BF16: 2.0**-4}
_VENDOR_FLOOR = 2.0**-100
_ACCUMULATE = 2.0**-24
_REF = 2.0**-53
_FLUSH = 2.0**-126
_COMPILE_FAILED = "the_bfloat_library_did_not_compile"
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_WIDE = "pipeline_simd_width_is_not_32"
_OTHER_BYTES = "static_threadgroup_bytes_differ_from_the_text"
# The rounding slack of err / bound and of ratio * bound in float64.
_SLACK = 1e-12
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ERRORS = {"max_abs_error", "max_bound_ratio", "max_relative_error"}
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
    "support",
    "planned_work",
    "ceiling",
    "compute_ceiling",
    "correctness",
    "benchmark",
    "speedup",
    "verification",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


class MetalTcVerificationError(MetalTcError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetalTcVerificationError(message)


def _hash(array: numpy.ndarray, code: str) -> str:
    return sha256(numpy.ascontiguousarray(array).astype(code).tobytes()).hexdigest()


def _gamma(count: int, unit: float) -> float:
    product = count * unit
    return product / (1 - product)


def _number(value: object, low: float = 0.0, high: float = math.inf) -> bool:
    return type(value) is float and math.isfinite(value) and low <= value <= high


def _bfloat(values: numpy.ndarray) -> numpy.ndarray:
    """Round float32 to bfloat by its own integer rule: nearest, ties to even."""

    bits = values.view(numpy.uint32)
    upper = (bits >> 16).astype(numpy.uint32)
    lower = bits & 0xFFFF
    up = (lower > 0x8000) | ((lower == 0x8000) & (upper & 1 == 1))
    return ((upper + up.astype(numpy.uint32)) << 16).astype(numpy.uint32)


class _Operands:
    """The verifier's own draws and roundings of one case or size."""

    def __init__(self, seed: int, stream: int, index: int, m: int, n: int, k: int):
        generator = numpy.random.default_rng([seed, stream, index])
        self.a = generator.standard_normal((m, k), dtype=numpy.float32)
        self.b = generator.standard_normal((k, n), dtype=numpy.float32)
        self.m, self.n, self.k = m, n, k

    def digests(self) -> dict[str, dict[str, str]]:
        result = {}
        for kind in _TYPES:
            pair = {}
            for name, values in (("a", self.a), ("b", self.b)):
                if kind == _F32:
                    digest = _hash(values, "<f4")
                elif kind == _F16:
                    digest = _hash(values.astype(numpy.float16), "<f2")
                else:
                    digest = _hash((_bfloat(values) >> 16).astype(numpy.uint16), "<u2")
                pair[f"{name}_sha256"] = digest
            result[kind] = pair
        return result

    def reference(self, count: int | None):
        if count is None or count >= self.m:
            rows = numpy.arange(self.m, dtype=numpy.int64)
        else:
            rows = numpy.unique(numpy.rint(numpy.linspace(0, self.m - 1, count)))
            rows = rows.astype(numpy.int64)
        left = self.a[rows].astype(numpy.float64)
        right = self.b.astype(numpy.float64)
        shape = (rows.size, self.n)
        output = numpy.zeros(shape, dtype=numpy.float64)
        scale = numpy.zeros(shape, dtype=numpy.float64)
        product = numpy.empty(shape, dtype=numpy.float64)
        row_sums = numpy.zeros(rows.size, dtype=numpy.float64)
        column_sums = numpy.zeros(self.n, dtype=numpy.float64)
        for step in range(self.k):
            numpy.multiply.outer(left[:, step], right[step, :], out=product)
            output += product
            numpy.abs(product, out=product)
            scale += product
        magnitude_a, magnitude_b = numpy.abs(left), numpy.abs(right)
        for step in range(self.k):
            row_sums += magnitude_a[:, step]
            column_sums += magnitude_b[step, :]
        _require(bool(numpy.all(scale > 0)), "a checked output has no scale")
        return output, scale, row_sums[:, None] + column_sums[None, :]


def _bound(kind: str, k: int, scale: numpy.ndarray, sums: numpy.ndarray):
    u, e = _UNIT[kind], _FLOOR[kind]
    g, r = _gamma(k, _ACCUMULATE), _gamma(k, _REF)
    factor = g * (1.0 + u) * (1.0 + u) + 2.0 * u + u * u + r
    term = e * (1.0 + u) * (1.0 + g)
    constant = k * e * e * (1.0 + g) + 2.0 * (k + 1) * _FLUSH
    return factor * scale + term * sums + constant


def _launch(m: int, n: int) -> dict[str, object]:
    groups = -(-m // _TILE) * -(-n // _TILE)
    threads = groups * _GROUP
    return {
        "extent": [threads],
        "group_size": [_GROUP],
        "groups": [groups],
        "threads": [threads],
        "threads_per_group": _GROUP,
        "launched_threads": threads,
        "active_threads": threads,
        "padding_threads": 0,
    }


def _errors(entry: object, keys: set[str], bound, scale, label: str) -> dict:
    """Check that the recorded errors agree with the rebuilt bounds and scale."""

    _require(
        type(entry) is dict
        and set(entry) == keys
        and _number(entry.get("max_abs_error"))
        and _number(entry.get("max_bound_ratio"), 0.0, 1.0)
        and _number(entry.get("max_relative_error")),
        f"{label} breaks its bound or its record form",
    )
    error, ratio = entry["max_abs_error"], entry["max_bound_ratio"]
    share = entry["max_relative_error"]
    low = ratio * float(bound.min()) * (1 - _SLACK)
    high = ratio * float(bound.max()) * (1 + _SLACK)
    least = error / float(scale.max()) * (1 - _SLACK)
    most = ratio * float(numpy.max(bound / scale)) * (1 + _SLACK)
    _require(
        low <= error <= high and least <= share <= most,
        f"{label} records errors that its bounds and scale cannot give",
    )
    return entry


def _support(section: object) -> dict[str, bool]:
    sides = section.get("sides") if type(section) is dict else None
    version = section.get("metal_language_version") if type(section) is dict else None
    _require(
        type(section) is dict
        and set(section) == {"rule", "metal_language_version", "sides"}
        and section["rule"] == SUPPORT_RULE
        and type(version) is int
        and 0 < version < 10_000
        and type(sides) is dict
        and set(sides) == set(_OPTIONAL),
        "the support record must hold its rule, the Metal version and both sides",
    )
    result = {}
    for name, record in sides.items():
        _require(
            type(record) is dict
            and set(record) == {"supported", "reason"}
            and type(record["supported"]) is bool
            and (record["reason"] is None) is record["supported"]
            and (record["reason"] is None or record["reason"] in SUPPORT_REASONS),
            f"the support of {name} must be a flag with one known reason",
        )
        result[name] = record["supported"]
    return result


def _admission(name: str, limits: dict) -> dict[str, object]:
    kernel = KERNELS[name]
    reason = None
    if limits["max_threads_per_threadgroup"] < kernel.group_size:
        reason = _FEWER
    elif limits["thread_execution_width"] != 32:
        reason = _WIDE
    elif limits["static_threadgroup_memory_bytes"] != kernel.threadgroup_bytes:
        reason = _OTHER_BYTES
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


def _shaders(section: object, support: dict) -> None:
    kernels = section.get("kernels") if type(section) is dict else None
    _require(
        type(kernels) is list and len(kernels) == len(KERNELS),
        "the shader section must hold every kernel",
    )
    refused = support["sides"][_M3]["reason"] == _COMPILE_FAILED
    rebuilt, admissions = [], []
    for entry, (name, kernel) in zip(kernels, KERNELS.items(), strict=True):
        _require(type(entry) is dict, "a kernel record must be a JSON object")
        if name == _KERNEL[_M3] and refused:
            _require(entry.get("limits") is None, "an uncompiled kernel has no limits")
            limits = None
        else:
            limits = _limits(entry.get("limits"))
            admissions.append(_admission(name, limits))
        rebuilt.append({**kernel.to_mapping(), "limits": limits})
    expected = {
        "libraries": {name: library_sha256(name) for name in ("main", "bfloat")},
        "kernels": rebuilt,
        "admissions": admissions,
    }
    _require(_exact(section, expected), "shader section does not recompute")
    _require(
        all(entry["admitted"] for entry in admissions),
        "a report needs every compiled pipeline admitted",
    )


def _ran(supported: dict[str, bool], sides: tuple[str, ...]) -> list[str]:
    return [side for side in sides if supported.get(side, True)]


def _same(flag: object, label: str) -> bool:
    _require(type(flag) is bool, f"{label} must be a flag")
    return flag


def _cases(section: object, config: MetalTcConfig, ran: list[str]) -> list[dict]:
    cases = config.correctness.cases
    _require(
        type(section) is list and len(section) == len(cases),
        "correctness cases do not cover the config",
    )
    rebuilt = []
    keys = {"buffer_sha256", "identical_repeats", *_ERRORS}
    for index, (entry, case) in enumerate(zip(section, cases, strict=True)):
        label = f"case {index}"
        _require(
            type(entry) is dict and type(entry.get("outputs")) is dict,
            f"{label} must be a JSON object with its outputs",
        )
        outputs = entry["outputs"]
        _require(set(outputs) == set(ran), f"{label} must cover the supported kernels")
        values = _Operands(config.seed, 0, index, case.m, case.n, case.k)
        output, scale, sums = values.reference(None)
        bounds = {kind: _bound(kind, case.k, scale, sums) for kind in _TYPES}
        for variant in ran:
            record = outputs[variant]
            _errors(record, keys, bounds[_KIND[variant]], scale, f"{label} {variant}")
            _require(
                isinstance(record["buffer_sha256"], str)
                and _SHA256.fullmatch(record["buffer_sha256"]) is not None
                and type(record["identical_repeats"]) is bool,
                f"{label} {variant} must hold a buffer digest and the repeat flag",
            )
        same = _same(entry.get("m1_same_bits_as_s1"), f"{label} m1_same_bits_as_s1")
        twins = all(
            outputs[_M1][key] == outputs[_S1][key]
            for key in ("buffer_sha256", *_ERRORS)
        )
        _require(
            same == (outputs[_M1]["buffer_sha256"] == outputs[_S1]["buffer_sha256"])
            and (not same or twins),
            f"{label} claims equal bits that its digests or errors deny",
        )
        expected = {
            "index": index,
            "case": case.to_mapping(),
            "repeats": config.correctness.repeats,
            "launch": _launch(case.m, case.n),
            "inputs": values.digests(),
            "reference_sha256": _hash(output, "<f8"),
            "magnitude_sha256": _hash(scale, "<f8"),
            "bound_sha256": {kind: _hash(bounds[kind], "<f8") for kind in _TYPES},
            "outputs": outputs,
            "m1_same_bits_as_s1": same,
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
        rebuilt.append(expected)
    return rebuilt


def _compulsory(side: str, size: int) -> int:
    element = _BYTES[_KIND[side]]
    output = 4 if side in _VARIANTS else element
    return (2 * element + output) * size * size


def _side(entry: object, name: str, calls: int, size: int, config) -> dict:
    repeats = config.benchmark.repeats
    samples = entry.get("samples_ns") if type(entry) is dict else None
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{name} samples_ns must hold {repeats} positive integers",
    )
    timing = _timing(samples)
    flops = 2 * size * size * size
    compulsory = _compulsory(name, size)
    work, moved = flops * calls, compulsory * calls
    return {
        "implementation": name,
        "calls_per_sample": calls,
        "warmup": config.benchmark.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": timing,
        "median_ns_per_call": timing["median_ns"] / calls,
        "iqr_ns_per_call": timing["iqr_ns"] / calls,
        "flops_per_call": flops,
        "compulsory_bytes_per_call": compulsory,
        "tflops_per_second": work / timing["median_ns"] / 1000,
        "iqr_tflops_per_second": [
            work / timing["p75_ns"] / 1000,
            work / timing["p25_ns"] / 1000,
        ],
        "effective_gb_per_second": moved / timing["median_ns"],
        "intensity_flops_per_byte": flops / compulsory,
    }


def _points(section: object, config: MetalTcConfig, ran: list[str]) -> list[dict]:
    timing = config.benchmark
    points = timing.points()
    _require(
        type(section) is list and len(section) == len(points),
        "benchmark points do not cover the sizes",
    )
    rebuilt = []
    for entry, point in zip(section, points, strict=True):
        label = f"point {point.index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        size = point.size
        values = _Operands(config.seed, 1, point.index, size, size, size)
        output, scale, sums = values.reference(timing.check_rows)
        calls = -(-timing.target_flops_per_sample // (2 * size**3))
        _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
        recorded, checks = entry.get("sides"), entry.get("checks")
        _require(
            type(recorded) is dict and type(checks) is dict and set(checks) == set(ran),
            f"{label} sides and checks must cover the supported sides",
        )
        rebuilt_checks = {}
        for side in ran:
            kind = _KIND[side]
            bound = (
                _bound(kind, size, scale, sums)
                if side in _VARIANTS
                else _VENDOR[kind] * scale + _VENDOR_FLOOR
            )
            rebuilt_checks[side] = _errors(
                checks[side], _ERRORS, bound, scale, f"{label} check of {side}"
            )
        same = _same(entry.get("m1_same_bits_as_s1"), f"{label} m1 flag")
        vendor = _same(entry.get("torch_float32_same_bits_as_s1"), f"{label} flag")
        _require(
            (not same or checks[_M1] == checks[_S1])
            and (
                not vendor
                or all(
                    checks[_T32][key] == checks[_S1][key]
                    for key in ("max_abs_error", "max_relative_error")
                )
            ),
            f"{label} claims equal bits that its errors deny",
        )
        expected = {
            "index": point.index,
            "size": size,
            "launch": _launch(size, size),
            "inputs": values.digests(),
            "check_rows": timing.check_rows,
            "reference_sha256": _hash(output, "<f8"),
            "magnitude_sha256": _hash(scale, "<f8"),
            "side_order": list(_SIDES),
            "supported": list(ran),
            "sides": {
                side: _side(recorded.get(side), side, calls, size, config)
                for side in ran
            },
            "checks": rebuilt_checks,
            "m1_same_bits_as_s1": same,
            "torch_float32_same_bits_as_s1": vendor,
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


def _rate(entry: dict, bandwidth: float, fma: float) -> dict[str, object]:
    gflops = entry["tflops_per_second"] * 1000
    intensity = entry["intensity_flops_per_byte"]
    roof = min(fma, intensity * bandwidth)
    return {
        "tflops_per_second": entry["tflops_per_second"],
        "iqr_tflops_per_second": list(entry["iqr_tflops_per_second"]),
        "median_ns_per_call": entry["median_ns_per_call"],
        "intensity_flops_per_byte": intensity,
        "roofline_gflops_per_second": roof,
        "fraction_of_fma": gflops / fma,
        "fraction_of_roofline": gflops / roof,
        "compute_bound": intensity * bandwidth >= fma,
    }


def _speedup(points, bandwidth: float, fma: float, minimum: float, version: str):
    rows = []
    for point in points:
        sides = point["sides"]
        steps = []
        for step in PLAN:
            if step.first not in sides or step.base not in sides:
                steps.append(
                    {
                        "step": step.step,
                        "ran": False,
                        "predicted": list(step.predicted),
                        "held": None,
                    }
                )
                continue
            verdict = _ratio(sides[step.first], sides[step.base], minimum)
            steps.append(
                {
                    "step": step.step,
                    "ran": True,
                    **verdict,
                    "predicted": list(step.predicted),
                    "held": verdict["verdict"] in step.predicted,
                }
            )
        rows.append(
            {
                "index": point["index"],
                "size": point["size"],
                "rates": {
                    side: _rate(sides[side], bandwidth, fma)
                    for side in _SIDES
                    if side in sides
                },
                "steps": steps,
            }
        )
    return {
        "vendor": f"{TORCH_CALL}, PyTorch {version}",
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "roofline_rule": ROOFLINE_RULE,
        "min_faster_ratio": minimum,
        "bandwidth_gb_per_second": bandwidth,
        "fma_gflops_per_second": fma,
        "ridge_flops_per_byte": fma / bandwidth,
        "points": rows,
        "hypotheses": [
            {
                "step": step.step,
                "predicted": list(step.predicted),
                "held_at": [r["index"] for r in rows if r["steps"][i]["held"] is True],
                "missed_at": [
                    r["index"] for r in rows if r["steps"][i]["held"] is False
                ],
                "not_run_at": [
                    r["index"] for r in rows if r["steps"][i]["held"] is None
                ],
            }
            for i, step in enumerate(PLAN)
        ],
    }


def _planned(config: MetalTcConfig) -> dict[str, int]:
    timing = config.benchmark
    samples = timing.warmup + timing.repeats
    calls = flops = 0
    for size in timing.sizes:
        work = 2 * size**3
        batch = -(-timing.target_flops_per_sample // work)
        calls += len(_SIDES) * samples * batch
        flops += len(_SIDES) * samples * batch * work
    cases = config.correctness.cases
    return {
        "correctness_cases": len(cases),
        "correctness_products": sum(case.m * case.n * case.k for case in cases),
        "correctness_launches": len(cases) * 4 * config.correctness.repeats,
        "benchmark_points": len(timing.sizes),
        "sides_per_point": len(_SIDES),
        "timed_samples": len(_SIDES) * len(timing.sizes) * timing.repeats,
        "calls": calls,
        "flops": flops,
    }


def _summary(label, cases, support, points, gains, captured, load) -> dict:
    ran = [variant for variant in _VARIANTS if variant in cases[0]["outputs"]]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "metal_language_version": support["metal_language_version"],
        "supported": {
            name: record["supported"] for name, record in support["sides"].items()
        },
        "max_bound_ratio": {
            v: max(case["outputs"][v]["max_bound_ratio"] for case in cases) for v in ran
        },
        "max_relative_error": {
            v: max(case["outputs"][v]["max_relative_error"] for case in cases)
            for v in ran
        },
        "identical_repeats": {
            v: all(case["outputs"][v]["identical_repeats"] for case in cases)
            for v in ran
        },
        "m1_same_bits_as_s1_cases": sum(case["m1_same_bits_as_s1"] for case in cases),
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "index": row["index"],
                "size": row["size"],
                "tflops_per_second": {
                    side: entry["tflops_per_second"]
                    for side, entry in row["rates"].items()
                },
                "fraction_of_fma": {
                    side: entry["fraction_of_fma"]
                    for side, entry in row["rates"].items()
                },
                "verdicts": {
                    step["step"]: step["verdict"] if step["ran"] else None
                    for step in row["steps"]
                },
                "m1_same_bits_as_s1": point["m1_same_bits_as_s1"],
                "torch_float32_same_bits_as_s1": point["torch_float32_same_bits_as_s1"],
            }
            for row, point in zip(gains["points"], points, strict=True)
        ],
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "not_run": len(record["not_run_at"]),
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
    fma = _flash_compute_ceiling(
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
    supported = _support(report["support"])
    _shaders(report["shaders"], report["support"])
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, _ran(supported, _VARIANTS))
    expected = {
        "distribution": DISTRIBUTION,
        "reference": REFERENCE,
        "bound": BOUND,
        "relative_error": RELATIVE_ERROR,
        "tail": _TAIL,
        "guard_float32_bits": _GUARD,
        "sentinel_float32_bits": _SENTINEL,
        "rules": RULES,
        "cases": cases,
    }
    _require(_exact(correctness, expected), "correctness section does not recompute")
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config, _ran(supported, _SIDES))
    expected = {
        "device": "mps",
        "synchronization": SYNCHRONIZATION,
        "sides": list(_SIDES),
        "torch_call": TORCH_CALL,
        "arithmetic": ARITHMETIC,
        "compulsory": COMPULSORY,
        "vendor_bound": VENDOR_BOUND,
        "check_rule": CHECK_RULE,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
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
        label, cases, report["support"], points, gains, captured, report["host_load"]
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
        "metal_language_version": summary["metal_language_version"],
        "supported": summary["supported"],
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "max_relative_error": summary["max_relative_error"],
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
    config: MetalTcConfig | None = None,
    ceiling: FlashCeiling | None = None,
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally against its inputs.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76, Day 79 or Day 81 check, is reported as
    ``MetalTcVerificationError``.
    """

    if config is not None and not isinstance(config, MetalTcConfig):
        raise TypeError("config must be a MetalTcConfig")
    if ceiling is not None and not isinstance(ceiling, FlashCeiling):
        raise TypeError("ceiling must be a FlashCeiling")
    try:
        return _verify(report, config, ceiling)
    except MetalTcVerificationError:
        raise
    except (
        MatmulTiledVerificationError,
        MetalFusedVerificationError,
        MetalFlashVerificationError,
        MetalTcError,
    ) as error:
        raise MetalTcVerificationError(str(error)) from error
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
        raise MetalTcVerificationError(
            f"report failed verification: {error}"
        ) from error
