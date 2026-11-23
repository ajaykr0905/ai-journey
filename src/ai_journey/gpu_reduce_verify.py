"""Independent verification of sealed reduction reports, on the CPU alone.

A device result cannot be observed again, so verification rebuilds the
check of each recorded result from the seed with its own code, not the code
of the run:

- it draws each input again with NumPy's generator and hashes it;
- it sums the input exactly by another method than the run: ``frexp``
  splits each value into an integer significand and an exponent, Python
  integers join the exponents, and ``Fraction`` holds the result;
- it counts each order's depth D again from the kernel's declared mapping
  and tree, and recomputes ``gamma_D(2^-24) * sum |x|`` in exact rationals;
- it requires each recorded result to be a finite float32 value within that
  bound, and equal to the exact sum when every order is exact.

A report records the result of the first run of each kernel on each case,
and the latest timed result of each side at each size. The later runs and
the repeat leave only a flag that says whether they kept those bits, so the
verifier checks the type of that flag and cannot rebuild it.

The timings cannot be observed again either, so they are checked for
structure: sample counts, positive integers, and the statistics, rates,
batch sizes, ratios, verdicts, held hypotheses, run totals and summary that
follow from them. Each pipeline verdict is derived again from
the recorded limits, and each launch from n. Besides the label and rule
texts of the run, the verifier uses three modules that measure nothing: the
config parser checks the bounds, the kernel registry gives each kernel's
text and launch shape, and the declared plan gives the hypotheses that the
report must seal, byte for byte. Every section must match its rebuild as
canonical JSON, so ``1``, ``1.0`` and ``true`` stay distinct. Torch is never
imported.

With a gpu-ceilings report, already accepted by the Day 74 verifier, the
ceiling section must name that report exactly. Only a
``time.perf_counter_ns`` report from ``torch.mps.compile_shader`` verifies
as a measurement, and it must hold a measured ceiling of its own
environment; any ``injected`` backend or timer verifies only as synthetic.
A matching digest shows internal consistency; it cannot prove that a timing
was observed or that a GPU ran.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy

from .gpu_ceiling_report import MEASUREMENT_LABEL as CEILING_MEASURED
from .gpu_ceiling_report import SYNTHETIC_LABEL as CEILING_SYNTHETIC
from .gpu_reduce_benchmark import DECLARED_TOLERANCE, GB_PER_SECOND, ORDER_BOUND
from .gpu_reduce_bound import BOUND_SOURCE, DEPTH_RULE, EXACT_RULE, VENDOR_RULE
from .gpu_reduce_ceiling import DEFINITION, FRACTION_RULE, SOURCE, BandwidthCeiling
from .gpu_reduce_config import GpuReduceConfig, config_from_mapping
from .gpu_reduce_plan import PLAN, SIZE_CLASSES, plan_mapping, plan_sha256
from .gpu_reduce_report import (
    CORRECTNESS_RULES,
    MEASUREMENT_BOUNDARY,
    REFERENCE,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
    UNIT_ROUNDOFF,
)
from .gpu_reduce_runtime import TORCH_EQUIVALENT
from .gpu_reduce_source import KERNELS, GpuReduceError, library_source
from .gpu_reduce_speedup import BEST_RULE, HYPOTHESIS_RULE, RATIO_RULE
from .gpu_reduce_speedup import FRACTION_RULE as SPEEDUP_FRACTION_RULE
from .kernel_metal_report import (
    ENVIRONMENT_FIELDS,
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
    _calls,
    _digest,
    _exact,
    _hash,
    _launch,
    _limits,
    _load,
    _switches,
    _text_digest,
    _timing,
    _warnings,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_SENTINEL = 0x7FC00000
_CASE_STREAM, _TIMING_STREAM = 78, 79
_STEPS = ("R0", "R1", "R2", "R3", "R4", "R5")
_KERNELS = tuple(name for name in KERNELS if KERNELS[name].step is not None)
_COMBINE = "reduce_combine"
_TORCH = "torch_sum"
_SIDES = (*_KERNELS, _TORCH)
_VENDOR_DEPTH = 4096
_CHUNK = 1 << 22
_SCALE = 1 << 149
_STREAMS = ("stream_copy", "stream_read_reduce")
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_WIDTH = "pipeline_simd_width_is_not_32"
_BYTES = "static_threadgroup_bytes_differ_from_the_kernel"
_VERDICTS = ("faster", "slower", "no_clear_difference")
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
    "ceiling",
    "planned_work",
    "correctness",
    "benchmark",
    "speedup",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GpuReduceVerificationError(GpuReduceError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuReduceVerificationError(message)


def _ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _draw(seed: int, index: int, pattern: str, n: int) -> numpy.ndarray:
    """Draw one case input again from the seed, as the lab defines it."""

    generator = numpy.random.default_rng([seed, _CASE_STREAM, index])
    if pattern == "normal":
        return generator.standard_normal(n, dtype=numpy.float32)
    if pattern == "positive":
        return generator.random(n, dtype=numpy.float32)
    if pattern == "mixed":
        signs = generator.choice(numpy.array([-1.0, 1.0]), size=n)
        mantissas = 1.0 + generator.random(n, dtype=numpy.float32)
        exponents = generator.integers(-24, 25, size=n)
        values = signs * numpy.ldexp(mantissas.astype(numpy.float64), exponents)
        return values.astype(numpy.float32)
    if pattern == "ones":
        return numpy.ones(n, dtype=numpy.float32)
    _require(pattern == "halves", f"unknown pattern {pattern!r}")
    return (generator.integers(-8, 9, size=n) * 0.5).astype(numpy.float32)


class _Sums:
    """Exact running sums by frexp: value = integer * 2^(exponent - 24)."""

    def __init__(self) -> None:
        self.total = 0
        self.magnitude = 0
        self.grid: int | None = None

    def add(self, chunk: numpy.ndarray) -> None:
        mantissa, exponent = numpy.frexp(chunk.astype(numpy.float64))
        integer = numpy.ldexp(mantissa, 24).astype(numpy.int64)
        keep = integer != 0
        integer, exponent = integer[keep], exponent[keep]
        if not integer.size:
            return
        low_exponent = int(exponent.min()) - 24
        _require(low_exponent >= -149, "an input value is below the float32 range")
        offsets = exponent - 24 - low_exponent
        for weights, target in ((integer, "total"), (numpy.abs(integer), "magnitude")):
            buckets = numpy.bincount(offsets, weights=weights.astype(numpy.float64))
            joined = sum(int(value) << shift for shift, value in enumerate(buckets))
            # Scale to units of 2^-149, the step of every float32.
            setattr(
                self, target, getattr(self, target) + (joined << (low_exponent + 149))
            )
        magnitude = numpy.abs(integer)
        lowest = magnitude & -magnitude
        powers = numpy.frexp(lowest.astype(numpy.float64))[1] - 1
        grid = int((powers + exponent - 24).min())
        self.grid = grid if self.grid is None else min(self.grid, grid)

    def mapping(self) -> dict[str, object]:
        _require(
            self.grid is None or self.grid >= -126,
            "an input value is finer than 2^-126, so the bound would not hold",
        )
        magnitude = Fraction(self.magnitude, _SCALE)
        exact = self.grid is None or magnitude <= Fraction(2) ** (24 + self.grid)
        return {
            "reference": float(Fraction(self.total, _SCALE)),
            "sum_abs": float(magnitude),
            "grid_exponent": self.grid,
            "exact_in_any_order": exact,
        }

    def exact(self) -> tuple[Fraction, Fraction]:
        return Fraction(self.total, _SCALE), Fraction(self.magnitude, _SCALE)


def _sums(values: numpy.ndarray) -> _Sums:
    sums = _Sums()
    for start in range(0, values.size, _CHUNK):
        sums.add(values[start : start + _CHUNK])
    return sums


def _shape(name: str, n: int) -> tuple[int, int, int]:
    """Return the pass-1 extent, the partial count and the depth D at n."""

    kernel = KERNELS[name]
    if kernel.mapping == "one_element_per_thread":
        extent, chain = n, 0
    elif kernel.mapping == "two_elements_256_apart":
        extent, chain = _ceil(n, 2), 1
    else:
        _require(kernel.mapping == "grid_stride_loop", f"{name} has no known mapping")
        extent = kernel.fixed_groups * kernel.group_size
        chain = _ceil(n, extent) - 1
    groups = _ceil(extent, kernel.group_size)
    tree = 31 + 7 if kernel.tree == "simd_sum_then_simd_sum" else 8
    combine = _ceil(groups, 1024) - 1 + 10
    return extent, groups, chain + tree + combine


def _gamma(depth: int) -> Fraction:
    return Fraction(depth, (1 << 24) - depth)


def _check(
    entry: dict[str, object],
    depth: int,
    total: Fraction,
    magnitude: Fraction,
    exact_case: bool,
    label: str,
) -> dict[str, object]:
    """Rebuild one result's check from its recorded output and the exact sums."""

    output = entry.get("output")
    _require(
        type(output) is float
        and math.isfinite(output)
        and float(numpy.float32(output)) == output,
        f"{label} must record a finite float32 result",
    )
    error = abs(Fraction(output) - total)
    bound = _gamma(depth) * magnitude
    _require(error <= bound, f"{label} lies outside the bound of its order")
    _require(not exact_case or error == 0, f"{label} misses an exact sum")
    ratio = 0.0 if bound == 0 else float(error / bound)
    return {
        "output": output,
        "depth": depth,
        "error": float(error),
        "bound": float(bound),
        "bound_ratio": ratio,
        "exact_result": error == 0,
    }


def _admission(name: str, limits: dict[str, int]) -> dict[str, object]:
    kernel = KERNELS[name]
    reason = None
    if limits["max_threads_per_threadgroup"] < kernel.group_size:
        reason = _FEWER
    elif (
        kernel.simd_lanes is not None
        and limits["thread_execution_width"] != kernel.simd_lanes
    ):
        reason = _WIDTH
    elif limits["static_threadgroup_memory_bytes"] != kernel.threadgroup_bytes:
        reason = _BYTES
    return {
        "name": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": kernel.group_size,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "required_simd_width": kernel.simd_lanes,
        "simd_width": limits["thread_execution_width"],
        "declared_threadgroup_bytes": kernel.threadgroup_bytes,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
    }


def _shaders(section: object) -> dict[str, dict[str, int]]:
    entries = section.get("kernels") if type(section) is dict else None
    _require(
        type(entries) is list and len(entries) == len(KERNELS),
        "shaders must list every registered kernel",
    )
    limits, rebuilt = {}, []
    for name, entry in zip(KERNELS, entries):
        limits[name] = _limits(entry.get("limits") if type(entry) is dict else None)
        admission = _admission(name, limits[name])
        _require(admission["admitted"], f"{name} was refused by its pipeline")
        rebuilt.append(
            {
                **KERNELS[name].to_mapping(),
                "source_sha256": _text_digest(KERNELS[name].source),
                "limits": limits[name],
                "admission": admission,
            }
        )
    expected = {"library_sha256": _text_digest(library_source()), "kernels": rebuilt}
    _require(_exact(section, expected), "shader section does not recompute")
    return limits


def _environment(environment: object, measured: bool) -> dict[str, str]:
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
    return environment


def _triple(values: object) -> bool:
    return (
        type(values) is list
        and len(values) == 3
        and all(
            type(value) is float and math.isfinite(value) and value >= 0.0
            for value in values
        )
    )


def _ceiling(
    section: object,
    environment: dict[str, str],
    measured: bool,
    supplied: BandwidthCeiling | None,
) -> dict[str, object]:
    fields = {
        "source",
        "definition",
        "report_sha256",
        "config_sha256",
        "measured",
        "label",
        "gb_per_second",
        "kernel",
        "buffer_bytes",
        "load_average_before",
        "load_average_after",
        "environment",
        "same_environment",
        "fraction_rule",
    }
    _require(
        type(section) is dict and set(section) == fields,
        "ceiling fields do not match the schema",
    )
    rate = section["gb_per_second"]
    other = section["environment"]
    _require(
        section["source"] == SOURCE
        and section["definition"] == DEFINITION
        and section["fraction_rule"] == FRACTION_RULE
        and all(
            isinstance(section[key], str) and _SHA256.fullmatch(section[key])
            for key in ("report_sha256", "config_sha256")
        )
        and type(section["measured"]) is bool
        and section["label"]
        == (CEILING_MEASURED if section["measured"] else CEILING_SYNTHETIC)
        and type(rate) is float
        and math.isfinite(rate)
        and rate > 0
        and section["kernel"] in _STREAMS
        and type(section["buffer_bytes"]) is int
        and section["buffer_bytes"] > 0
        and _triple(section["load_average_before"])
        and _triple(section["load_average_after"])
        and type(other) is dict
        and set(other) == set(ENVIRONMENT_FIELDS),
        "the ceiling section is not a verified gpu-ceilings record",
    )
    same = all(other[field] == environment[field] for field in ENVIRONMENT_FIELDS)
    _require(section["same_environment"] is same, "same_environment does not hold")
    _require(
        not measured or (section["measured"] and same),
        "a measured run needs a measured ceiling of its own environment",
    )
    if supplied is not None:
        expected = {
            **supplied.to_mapping(),
            "same_environment": same,
            "fraction_rule": FRACTION_RULE,
        }
        _require(
            _exact(section, expected),
            "report was not made with the supplied gpu-ceilings report",
        )
    return section


def _planned(config: GpuReduceConfig) -> dict[str, int]:
    sweep = config.benchmark
    samples = 7 * (sweep.warmup + sweep.repeats)
    calls = moved = 0
    for n in sweep.sizes:
        batch = _calls(4 * n, sweep.target_bytes_per_sample)
        calls += samples * batch
        moved += samples * batch * 4 * n
    cases = config.correctness.cases
    return {
        "correctness_cases": len(cases),
        "correctness_elements": sum(case.n for case in cases),
        "correctness_launches": len(cases) * 6 * config.correctness.runs * 2,
        "benchmark_points": len(sweep.sizes),
        "sides_per_point": 7,
        "timed_samples": 7 * len(sweep.sizes) * sweep.repeats,
        "calls": calls,
        "bytes_read": moved,
    }


def _bits(value: float) -> bytes:
    return numpy.float32(value).tobytes()


def _same_order(outputs: dict[str, float]) -> list[dict[str, object]]:
    return [
        {
            "kernel": step.kernel,
            "same_order_as": step.same_order_as,
            "same_bits": _bits(outputs[step.kernel])
            == _bits(outputs[step.same_order_as]),
        }
        for step in PLAN
        if step.same_order_as is not None
    ]


def _cases(
    section: object, config: GpuReduceConfig, limits: dict
) -> list[dict[str, object]]:
    cases = config.correctness.cases
    _require(
        type(section) is list and len(section) == len(cases),
        "correctness cases do not cover the config",
    )
    rebuilt = []
    for index, (entry, case) in enumerate(zip(section, cases)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        values = _draw(config.seed, index, case.pattern, case.n)
        sums = _sums(values)
        facts = sums.mapping()
        total, magnitude = sums.exact()
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(_KERNELS),
            f"{label} must record one output per step kernel",
        )
        records, results = [], {}
        for name, output in zip(_KERNELS, outputs):
            _require(type(output) is dict, f"{label} {name} must be an object")
            extent, groups, depth = _shape(name, case.n)
            same = output.get("same_bits_every_run")
            _require(type(same) is bool, f"{label} {name} must record its run bits")
            check = _check(
                output,
                depth,
                total,
                magnitude,
                facts["exact_in_any_order"],
                f"{label} {name}",
            )
            results[name] = check["output"]
            records.append(
                {
                    "kernel": name,
                    "launch": _launch(
                        [extent], [KERNELS[name].group_size], limits[name]
                    ),
                    "partials": groups,
                    "partials_guard": 64,
                    "output_guard": 31,
                    "runs": config.correctness.runs,
                    "same_bits_every_run": same,
                    **check,
                }
            )
        expected = {
            "index": index,
            "pattern": case.pattern,
            "n": case.n,
            "input_sha256": _hash(values),
            "input_guard": 1024,
            **facts,
            "outputs": records,
            "same_order": _same_order(results),
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
        rebuilt.append(expected)
    return rebuilt


def _side(
    entry: object, name: str, calls: int, per_call: int, config: GpuReduceConfig
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
        "iqr_gb_per_second": [moved / timing["p75_ns"], moved / timing["p25_ns"]],
    }


def _points(
    section: object, config: GpuReduceConfig, limits: dict
) -> list[dict[str, object]]:
    sizes = config.benchmark.sizes
    _require(
        type(section) is list and len(section) == len(sizes),
        "benchmark points do not cover the sweep",
    )
    generator = numpy.random.default_rng([config.seed, _TIMING_STREAM, 0])
    values = generator.standard_normal(max(sizes), dtype=numpy.float32)
    order = sorted(set(sizes))
    running, start, prefix = _Sums(), 0, {}
    for n in order:
        for first in range(start, n, _CHUNK):
            running.add(values[first : min(first + _CHUNK, n)])
        start = n
        prefix[n] = (running.mapping(), running.exact())
    rebuilt = []
    for index, (entry, n) in enumerate(zip(section, sizes)):
        label = f"point {index} (n={n})"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        facts, (total, magnitude) = prefix[n]
        calls = _calls(4 * n, config.benchmark.target_bytes_per_sample)
        recorded = entry.get("sides")
        checks = entry.get("checks")
        _require(
            type(recorded) is dict and type(checks) is dict,
            f"{label} sides and checks must be objects",
        )
        launches, rebuilt_checks = {}, {}
        for name in _SIDES:
            check = checks.get(name)
            _require(type(check) is dict, f"{label} must check {name}")
            same = check.get("same_bits_on_repeat")
            _require(type(same) is bool, f"{label} {name} must record its repeat")
            if name == _TORCH:
                depth, rule = _VENDOR_DEPTH, DECLARED_TOLERANCE
            else:
                extent, _, depth = _shape(name, n)
                rule = ORDER_BOUND
                launches[name] = _launch(
                    [extent], [KERNELS[name].group_size], limits[name]
                )
            result = _check(
                check, depth, total, magnitude, facts["exact_in_any_order"], label
            )
            rebuilt_checks[name] = {
                **result,
                "rule": rule,
                "same_bits_on_repeat": same,
            }
        expected = {
            "index": index,
            "n": n,
            "bytes_per_call": 4 * n,
            "side_order": list(_SIDES),
            "launches": launches,
            "sides": {
                name: _side(recorded.get(name), name, calls, 4 * n, config)
                for name in _SIDES
            },
            **facts,
            "checks": rebuilt_checks,
        }
        _require(
            _exact(entry, expected),
            f"{label} does not recompute from its samples and the seed",
        )
        rebuilt.append(expected)
    return rebuilt


def _ratio(first: dict, second: dict, minimum: float) -> dict[str, object]:
    one, two = first["timing"], second["timing"]
    ratio = first["median_gb_per_second"] / second["median_gb_per_second"]
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


def _class(n: int) -> str:
    for name, first, last in SIZE_CLASSES:
        if first <= n <= last:
            return name
    raise GpuReduceVerificationError(f"{n} is outside every size class")


def _rate(entry: dict) -> dict[str, object]:
    return {
        "median_gb_per_second": entry["median_gb_per_second"],
        "iqr_gb_per_second": entry["iqr_gb_per_second"],
    }


def _speedup(
    points: list[dict], minimum: float, torch_version: str, ceiling: float
) -> dict[str, object]:
    entries = []
    for point in points:
        sides, n = point["sides"], point["n"]
        vendor = sides[_TORCH]["median_gb_per_second"]
        rows = []
        for step in PLAN:
            entry = sides[step.kernel]
            to_base = _ratio(entry, sides[step.base], minimum)
            predicted = list(dict(step.predictions)[_class(n)])
            rows.append(
                {
                    "step": step.step,
                    "kernel": step.kernel,
                    "base": step.base,
                    "rate": _rate(entry),
                    "fraction_of_ceiling": entry["median_gb_per_second"] / ceiling,
                    "fraction_of_torch_sum": entry["median_gb_per_second"] / vendor,
                    "to_base": to_base,
                    "to_r0": None
                    if step.kernel == _KERNELS[0]
                    else _ratio(entry, sides[_KERNELS[0]], minimum),
                    "predicted": predicted,
                    "held": to_base["verdict"] in predicted,
                }
            )
        best = _KERNELS[0]
        for name in _KERNELS[1:]:
            if (
                sides[name]["median_gb_per_second"]
                > sides[best]["median_gb_per_second"]
            ):
                best = name
        entries.append(
            {
                "index": point["index"],
                "n": n,
                "size_class": _class(n),
                "torch_sum": {
                    **_rate(sides[_TORCH]),
                    "fraction_of_ceiling": vendor / ceiling,
                },
                "rows": rows,
                "best": {
                    "kernel": best,
                    **_rate(sides[best]),
                    "fraction_of_ceiling": sides[best]["median_gb_per_second"]
                    / ceiling,
                    "fraction_of_torch_sum": sides[best]["median_gb_per_second"]
                    / vendor,
                    "to_r0": None
                    if best == _KERNELS[0]
                    else _ratio(sides[best], sides[_KERNELS[0]], minimum),
                },
            }
        )
    hypotheses = []
    for position, step in enumerate(PLAN):
        held = [(entry["index"], entry["rows"][position]["held"]) for entry in entries]
        hypotheses.append(
            {
                "step": step.step,
                "kernel": step.kernel,
                "base": step.base,
                "held_at": [index for index, value in held if value],
                "missed_at": [index for index, value in held if not value],
            }
        )
    return {
        "reference_kernel": _KERNELS[0],
        "vendor": f"vendor GPU sum: torch.sum on MPS, PyTorch {torch_version}",
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": SPEEDUP_FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": minimum,
        "ceiling_gb_per_second": ceiling,
        "points": entries,
        "hypotheses": hypotheses,
    }


def _summary(
    label: str,
    cases: list[dict],
    points: list[dict],
    gains: dict,
    captured: dict,
    load: dict,
    ceiling: dict,
    fraction_measured: bool,
) -> dict[str, object]:
    order = []
    for step in PLAN:
        if step.same_order_as is None:
            continue
        mine, theirs = _KERNELS.index(step.kernel), _KERNELS.index(step.same_order_as)
        order.append(
            {
                "kernel": step.kernel,
                "same_order_as": step.same_order_as,
                "cases_with_other_bits": [
                    case["index"]
                    for case in cases
                    if _bits(case["outputs"][mine]["output"])
                    != _bits(case["outputs"][theirs]["output"])
                ],
                "points_with_other_bits": [
                    point["index"]
                    for point in points
                    if _bits(point["checks"][step.kernel]["output"])
                    != _bits(point["checks"][step.same_order_as]["output"])
                ],
            }
        )
    rows = []
    for entry in gains["points"]:
        best = entry["best"]
        rows.append(
            {
                "n": entry["n"],
                "best_kernel": best["kernel"],
                "best_gb_per_second": best["median_gb_per_second"],
                "best_fraction_of_ceiling": best["fraction_of_ceiling"],
                "best_over_r0": None
                if best["to_r0"] is None
                else best["to_r0"]["ratio"],
                "torch_sum_gb_per_second": entry["torch_sum"]["median_gb_per_second"],
                "torch_sum_fraction_of_ceiling": entry["torch_sum"][
                    "fraction_of_ceiling"
                ],
            }
        )
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": list(_KERNELS),
        "exact_cases": [case["index"] for case in cases if case["exact_in_any_order"]],
        "max_bound_ratio": {
            name: max(case["outputs"][position]["bound_ratio"] for case in cases)
            for position, name in enumerate(_KERNELS)
        },
        "cases_with_changed_bits": {
            name: [
                case["index"]
                for case in cases
                if not case["outputs"][position]["same_bits_every_run"]
            ]
            for position, name in enumerate(_KERNELS)
        },
        "points_with_changed_bits": {
            name: [
                point["index"]
                for point in points
                if not point["checks"][name]["same_bits_on_repeat"]
            ]
            for name in _SIDES
        },
        "same_order": order,
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": rows,
        "hypotheses": {
            record["step"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
            }
            for record in gains["hypotheses"]
        },
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
        "ceiling_gb_per_second": ceiling["gb_per_second"],
        "ceiling_report_sha256": ceiling["report_sha256"],
        "fraction_measured": fraction_measured,
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


def _verify(
    report: object,
    supplied: GpuReduceConfig | None,
    ceiling: BandwidthCeiling | None,
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
    environment = _environment(report["environment"], measured)
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    limits = _shaders(report["shaders"])
    section = _ceiling(report["ceiling"], environment, measured, ceiling)
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, limits)
    expected = {
        "reference": REFERENCE,
        "unit_roundoff": UNIT_ROUNDOFF,
        "bound_source": BOUND_SOURCE,
        "depth_rule": DEPTH_RULE,
        "exact_rule": EXACT_RULE,
        "sentinel_float32_bits": _SENTINEL,
        "rules": CORRECTNESS_RULES,
        "combine_launch": _launch([1024], [1024], limits[_COMBINE]),
        "cases": cases,
    }
    _require(_exact(correctness, expected), "correctness section does not recompute")
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config, limits)
    expected = {
        "device": "mps",
        "gb_per_second": GB_PER_SECOND,
        "torch_equivalent": TORCH_EQUIVALENT,
        "vendor_rule": VENDOR_RULE,
        "synchronization": SYNCHRONIZATION,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    gains = _speedup(
        points,
        config.comparison.min_faster_ratio,
        environment["torch_version"],
        section["gb_per_second"],
    )
    _require(
        _exact(report["speedup"], gains),
        "the speedup rows or the hypotheses do not recompute from the samples",
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
    fraction_measured = measured and section["measured"] and section["same_environment"]
    summary = _summary(
        label,
        cases,
        points,
        gains,
        captured,
        report["host_load"],
        section,
        fraction_measured,
    )
    _require(_exact(report["summary"], summary), "summary does not reconcile")
    return {
        "kind": REPORT_KIND,
        "report_sha256": report[REPORT_DIGEST_FIELD],
        "config_sha256": report["config_sha256"],
        "plan_sha256": report["plan"]["sha256"],
        "measured": measured,
        "scope": scope,
        "label": label,
        "ceiling_checked": ceiling is not None,
        **{key: summary[key] for key in summary if key != "label"},
    }


def verify_report(
    report: object,
    *,
    config: GpuReduceConfig | None = None,
    ceiling: BandwidthCeiling | None = None,
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally bound to its inputs.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 check, is reported as
    ``GpuReduceVerificationError``.
    """

    if config is not None and not isinstance(config, GpuReduceConfig):
        raise TypeError("config must be a GpuReduceConfig")
    if ceiling is not None and not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    try:
        return _verify(report, config, ceiling)
    except GpuReduceVerificationError:
        raise
    except MatmulTiledVerificationError as error:
        raise GpuReduceVerificationError(str(error)) from error
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise GpuReduceVerificationError(
            f"report failed verification: {error}"
        ) from error
