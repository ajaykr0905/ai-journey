"""Independent verification of sealed fused softmax reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
fact that the seed and the stored digests fix, with its own code: the input
rows of each case and point, the gapped and ragged layouts, the float64
reference by its declared algorithm, the bound of every element and row sum
for each step's order, the softmax with no max subtraction, the probe
arguments, each launch, and each pipeline verdict from the recorded limits.
An output whose bits all equal the rounded reference must hash to that
reference plus its sentinel gaps, and must record its errors exactly. Any
other output records a digest of another buffer, a count of 1 to n
differing elements, and errors within its bounds.

Besides the label and rule texts of the run, the verifier uses three
modules of the run that measure nothing: the config parser checks the
bounds, the kernel registry gives each kernel's text and threadgroup bytes,
and the declared plan gives the hypotheses that the report must seal, byte
for byte. It derives each statistic, rate, model byte count, ratio,
verdict, fraction of the ceiling, held hypothesis, run total and summary
from the raw samples. Every section must match its rebuild as canonical
JSON, so ``1``, ``1.0`` and ``true`` stay distinct. Timings cannot be
observed again, so they are checked for structure only, and torch is never
imported.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement, and only with a measured ceiling from the same
environment. Any ``injected`` backend or timer verifies only as synthetic.
A matching digest shows internal consistency; it cannot prove that a timing
was observed or that a GPU ran.
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
from .metal_fused_benchmark import COMPULSORY, TRAFFIC_MODEL, VENDOR_BOUND
from .metal_fused_ceiling import DEFINITION as CEILING_DEFINITION
from .metal_fused_ceiling import SOURCE as CEILING_SOURCE
from .metal_fused_ceiling import BandwidthCeiling
from .metal_fused_config import MetalFusedConfig, config_from_mapping
from .metal_fused_correctness import RULES
from .metal_fused_plan import PLAN, plan_mapping, plan_sha256
from .metal_fused_reference import (
    BOUND,
    DISTRIBUTIONS,
    EXP_PROBE,
    EXP_PROBE_RULE,
    ORDER_TERMS,
    REFERENCE,
    ROW_SUM_BOUND,
)
from .metal_fused_report import (
    MEASUREMENT_BOUNDARY,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
)
from .metal_fused_runtime import TORCH_EQUIVALENT
from .metal_fused_source import KERNELS, VARIANTS, MetalFusedError, fused_library
from .metal_fused_speedup import BEST_RULE, FRACTION_RULE, HYPOTHESIS_RULE, RATIO_RULE

_NS = 1_000_000_000
_MAX_CALLS = 4096
_STEPS = tuple(VARIANTS)
_U0, _F1, _F2, _F3 = _STEPS
_TORCH = "torch_softmax"
_SLOTS = 16
_WIDTH = 32
_GAP = 32
_GAP_INPUT = 2.0**100
_SENTINEL = 0x7FC00000
_UNIT = 2.0**-24
_EXP_ULPS = 4
_EXP_ERROR = _EXP_ULPS * 2.0**-23
_FLUSH = 2.0**-126
_FLOOR = 2.0**-125
_REF_UNIT = 2.0**-53
_VENDOR_UNIT = 2.0**-16
_PROBE_COUNT = 1 << 18
_PROBE_EXTREMES = (
    -3.4028234663852886e38,
    -1.0e30,
    -1000.0,
    -200.0,
    -150.0,
    -105.0,
    -87.5,
    -0.0,
)
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_WIDE = "pipeline_simd_width_is_not_32"
_BYTES = "static_threadgroup_bytes_differ_from_the_text"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
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
    "refusals",
    "planned_work",
    "ceiling",
    "exp_probe",
    "correctness",
    "benchmark",
    "speedup",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}


class MetalFusedVerificationError(MetalFusedError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetalFusedVerificationError(message)


def _hash(array: numpy.ndarray, code: str) -> str:
    return sha256(numpy.ascontiguousarray(array).astype(code).tobytes()).hexdigest()


def _gamma(count: int, unit: float) -> float:
    product = count * unit
    return product / (1 - product)


def _group(longest: int) -> int:
    size = _WIDTH
    while size * _SLOTS < longest:
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
        self.size = _group(self.longest)
        self.buffer = self.elements + gap * self.rows
        steps = self.lengths + gap
        self.starts = numpy.concatenate(([0], numpy.cumsum(steps)[:-1]))
        self.packed = numpy.concatenate(([0], numpy.cumsum(self.lengths)[:-1]))
        offsets = numpy.repeat(self.starts - self.packed, self.lengths)
        self.valid = numpy.arange(self.elements, dtype=numpy.int64) + offsets

    def mapping(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "elements": self.elements,
            "shortest": int(self.lengths.min()),
            "longest": self.longest,
            "group_size": self.size,
            "gap": self.gap,
            "buffer_elements": self.buffer,
            "dense": bool(numpy.all(self.lengths == self.lengths[0])),
            "lengths_sha256": _hash(self.lengths, "<i4"),
        }

    def launch(self) -> dict[str, object]:
        threads = self.rows * self.size
        return {
            "extent": [threads],
            "group_size": [self.size],
            "groups": [self.rows],
            "threads": [threads],
            "threads_per_group": self.size,
            "launched_threads": threads,
            "active_threads": threads,
            "padding_threads": 0,
        }


def _draw(seed: int, stream: int, index: int, count: int, name: str):
    generator = numpy.random.default_rng([seed, stream, index])
    if name == "normal":
        return generator.standard_normal(count, dtype=numpy.float32)
    if name == "uniform_80":
        return generator.uniform(-80.0, 80.0, count).astype(numpy.float32)
    offset = {"offset_plus_1000": 1000.0, "offset_minus_1000": -1000.0}[name]
    return (offset + generator.standard_normal(count)).astype(numpy.float32)


def _rows(seed: int, stream: int, index: int, layout: _Layout, name: str):
    values = _draw(seed, stream, index, layout.elements, name)
    if layout.gap == 0:
        return values
    buffer = numpy.full(layout.buffer, _GAP_INPUT, dtype=numpy.float32)
    buffer[layout.valid] = values
    return buffer


def _reference(buffer: numpy.ndarray, layout: _Layout):
    """The declared float64 reference: exp(x - max) over its row sum by reduceat."""

    values = buffer[layout.valid].astype(numpy.float64)
    top = numpy.maximum.reduceat(values, layout.packed)
    distance = numpy.repeat(top, layout.lengths) - values
    terms = numpy.exp(-distance)
    totals = numpy.add.reduceat(terms, layout.packed)
    return terms / numpy.repeat(totals, layout.lengths), distance


def _terms(step: str, length: int, size: int) -> tuple[int, int]:
    used = -(-length // size)
    partials = size // _WIDTH - 1
    if step in (_U0, _F1):
        return (used - 1) + int(math.log2(size)), 1
    if step == _F2:
        return (used - 1) + (_WIDTH - 1) + partials, 1
    return (used - 1) + 1 + (_WIDTH - 1) + 1 + partials, used + 2


def _bounds(step: str, output, distance, layout: _Layout):
    """The declared bound of each element and row sum, for one step's order."""

    lengths = layout.lengths
    terms = [_terms(step, int(length), layout.size) for length in lengths]
    additions = numpy.repeat([float(a) for a, _ in terms], lengths)
    exponentials = numpy.repeat([float(n) for _, n in terms], lengths)
    ref = numpy.array([_gamma(int(n) + 8, _REF_UNIT) for n in lengths])
    ref_each = numpy.repeat(ref, lengths)
    shift = numpy.exp(distance * _UNIT)
    alpha = shift * (1 + _EXP_ERROR)
    gamma = additions * _UNIT / (1 - additions * _UNIT)
    beta = shift * (1 + _EXP_ERROR) ** exponentials * (1 + gamma)
    weighted = numpy.add.reduceat(output * (beta - 1), layout.packed)
    rho = weighted / (1 - ref) + (lengths + 2 * layout.size) * _FLUSH
    rel = alpha * (1 + _UNIT) / (1 - numpy.repeat(rho, lengths)) - 1
    elements = (rel + ref_each) / (1 - ref_each) * output + _FLOOR
    summation = numpy.array([_gamma(int(n), _REF_UNIT) for n in lengths])
    row_sums = numpy.add.reduceat(elements, layout.packed) + ref + 2 * summation
    return elements, row_sums


def _naive_finite(buffer: numpy.ndarray, layout: _Layout) -> bool:
    values = buffer[layout.valid]
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore"):
        terms = numpy.exp(values)
        totals = numpy.add.reduceat(terms, layout.packed, dtype=numpy.float32)
        output = terms / numpy.repeat(totals, layout.lengths)
    return bool(numpy.all(numpy.isfinite(output)))


def _number(value: object, low: float = 0.0, high: float = math.inf) -> bool:
    return type(value) is float and math.isfinite(value) and low <= value <= high


def _count(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _admission(step: str, limits: dict, required: int) -> dict[str, object]:
    reason = None
    for name in VARIANTS[step].kernels:
        entry = limits[name]
        if entry["max_threads_per_threadgroup"] < required:
            reason = _FEWER
        elif entry["thread_execution_width"] != _WIDTH:
            reason = _WIDE
        elif (
            entry["static_threadgroup_memory_bytes"] != KERNELS[name].threadgroup_bytes
        ):
            reason = _BYTES
        if reason is not None:
            break
    return {
        "variant": step,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": required,
        "kernels": [
            {
                "name": name,
                "allowed_threads": limits[name]["max_threads_per_threadgroup"],
                "thread_execution_width": limits[name]["thread_execution_width"],
                "declared_threadgroup_bytes": KERNELS[name].threadgroup_bytes,
                "static_threadgroup_bytes": limits[name][
                    "static_threadgroup_memory_bytes"
                ],
            }
            for name in VARIANTS[step].kernels
        ],
    }


def _shaders(section: object, required: int) -> list[dict[str, object]]:
    entries = section.get("kernels") if type(section) is dict else None
    _require(
        type(entries) is list and len(entries) == len(KERNELS),
        "shaders must list every registered kernel",
    )
    limits = {
        name: _limits(entry.get("limits") if type(entry) is dict else None)
        for name, entry in zip(KERNELS, entries)
    }
    admissions = [_admission(step, limits, required) for step in _STEPS]
    expected = {
        "library_sha256": sha256(fused_library().encode("ascii")).hexdigest(),
        "kernels": [
            {
                "name": name,
                "buffers": list(kernel.buffers),
                "threadgroup_bytes": kernel.threadgroup_bytes,
                "simd_functions": kernel.simd_functions,
                "source_sha256": sha256(kernel.source.encode("ascii")).hexdigest(),
                "limits": limits[name],
            }
            for name, kernel in KERNELS.items()
        ],
        "steps": [variant.to_mapping() for variant in VARIANTS.values()],
        "admissions": admissions,
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return admissions


def _probe(section: object) -> dict[str, object]:
    count = _PROBE_COUNT - len(_PROBE_EXTREMES)
    grid = numpy.linspace(-104.0, 0.0, count).astype(numpy.float32)
    arguments = numpy.concatenate(
        (grid, numpy.asarray(_PROBE_EXTREMES, dtype=numpy.float32))
    )
    exact = numpy.exp(arguments.astype(numpy.float64))
    normal = int(numpy.count_nonzero(exact >= _FLUSH))
    _require(type(section) is dict, "the exp probe must be a JSON object")
    tiny = _PROBE_COUNT - normal
    _require(
        type(section.get("results_sha256")) is str
        and _SHA256.fullmatch(section["results_sha256"]) is not None
        and _number(section.get("max_relative_error"), 0.0, _EXP_ERROR)
        and _number(section.get("max_ulps"), 0.0, 2.0 * _EXP_ULPS)
        and _count(section.get("tiny_nonzero"), 0, tiny)
        and _number(section.get("tiny_max_abs_error"), 0.0, _FLUSH),
        "the exp probe does not meet the assumption of the bounds",
    )
    expected = {
        "arguments": EXP_PROBE,
        "count": _PROBE_COUNT,
        "arguments_sha256": _hash(arguments, "<f4"),
        "rule": EXP_PROBE_RULE,
        "exp_ulps": _EXP_ULPS,
        "exp_error": _EXP_ERROR,
        "results_sha256": section["results_sha256"],
        "normal_results": normal,
        "max_relative_error": section["max_relative_error"],
        "max_ulps": section["max_ulps"],
        "tiny_results": tiny,
        "tiny_nonzero": section["tiny_nonzero"],
        "tiny_max_abs_error": section["tiny_max_abs_error"],
        "holds": True,
    }
    _require(_exact(section, expected), "the exp probe section does not recompute")
    return section


def _output(entry: object, step: str, rebuilt: dict, label: str) -> dict:
    _require(type(entry) is dict, f"{label} output must be a JSON object")
    elements = rebuilt["layout"].elements
    differing = entry.get("differing_from_rounded_reference")
    _require(
        entry.get("kernel") == step
        and type(entry.get("buffer_sha256")) is str
        and _SHA256.fullmatch(entry["buffer_sha256"]) is not None
        and type(entry.get("identical_repeats")) is bool
        and _number(entry.get("max_abs_error"))
        and _number(entry.get("max_bound_ratio"), 0.0, 1.0)
        and _number(entry.get("max_row_sum_error"))
        and _number(entry.get("max_row_sum_ratio"), 0.0, 1.0)
        and _count(differing, 0, elements),
        f"{label}: {step} breaks a correctness rule or its record form",
    )
    elements_bound, rows_bound = rebuilt["bounds"][step]
    if differing == 0:
        exact = rebuilt["exact"]
        error = numpy.abs(exact["packed"].astype(numpy.float64) - rebuilt["output"])
        sums = numpy.abs(
            numpy.add.reduceat(
                exact["packed"].astype(numpy.float64), rebuilt["layout"].packed
            )
            - 1.0
        )
        _require(
            entry["buffer_sha256"] == exact["sha256"]
            and entry["max_abs_error"] == float(error.max())
            and entry["max_bound_ratio"] == float(numpy.max(error / elements_bound))
            and entry["max_row_sum_error"] == float(sums.max())
            and entry["max_row_sum_ratio"] == float(numpy.max(sums / rows_bound)),
            f"{label}: {step} claims the rounded reference bits but does not hold them",
        )
    else:
        _require(
            entry["buffer_sha256"] != rebuilt["exact"]["sha256"],
            f"{label}: {step} claims other bits but records the rounded reference",
        )
    return {
        **entry,
        "bound_sha256": _hash(elements_bound, "<f8"),
        "row_sum_bound_sha256": _hash(rows_bound, "<f8"),
    }


def _cases(
    section: object, config: MetalFusedConfig, names: list[str]
) -> list[dict[str, object]]:
    cases = config.correctness.cases
    _require(
        type(section) is list and len(section) == len(cases),
        "correctness cases do not cover the config",
    )
    rebuilt_cases = []
    for index, (entry, case) in enumerate(zip(section, cases)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        layout = _Layout(list(case.lengths), _GAP)
        buffer = _rows(config.seed, 0, index, layout, case.inputs)
        output, distance = _reference(buffer, layout)
        exact = numpy.full(layout.buffer, _SENTINEL, dtype=numpy.uint32).view(
            numpy.float32
        )
        rounded = output.astype(numpy.float32)
        exact[layout.valid] = rounded
        rebuilt = {
            "layout": layout,
            "output": output,
            "bounds": {step: _bounds(step, output, distance, layout) for step in names},
            "exact": {"sha256": _hash(exact, "<f4"), "packed": rounded},
        }
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(names),
            f"{label} must record one output per step that ran",
        )
        checked = [
            _output(item, step, rebuilt, label) for item, step in zip(outputs, names)
        ]
        digests = {item["kernel"]: item["buffer_sha256"] for item in checked}
        expected = {
            "index": index,
            "case": case.to_mapping(),
            "layout": layout.mapping(),
            "launch": layout.launch(),
            "repeats": config.correctness.repeats,
            "input_sha256": _hash(buffer, "<f4"),
            "reference_sha256": _hash(output, "<f8"),
            "naive_float32_finite": _naive_finite(buffer, layout),
            "outputs": checked,
            "same_bits": [
                {
                    "variant": step.variant,
                    "as": step.same_bits_as,
                    "same": digests[step.variant] == digests[step.same_bits_as],
                }
                for step in PLAN
                if step.same_bits_as is not None
                and step.variant in digests
                and step.same_bits_as in digests
            ],
        }
        _require(_exact(entry, expected), f"{label} does not rebuild from the seed")
        rebuilt_cases.append(expected)
    return rebuilt_cases


def _modeled(step: str, layout: _Layout) -> int:
    if step == _U0:
        return 16 * layout.elements + 44 * layout.rows
    return 8 * layout.elements + 8 * layout.rows


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
        "modeled_gb_per_second": None
        if modeled is None
        else modeled * calls / timing["median_ns"],
    }


def _check(entry: object, name: str, elements: int) -> dict[str, object]:
    _require(
        type(entry) is dict
        and set(entry)
        == {"max_abs_error", "max_bound_ratio", "differing_from_rounded_reference"}
        and _number(entry["max_abs_error"])
        and _number(entry["max_bound_ratio"], 0.0, 1.0)
        and _count(entry["differing_from_rounded_reference"], 0, elements),
        f"the check of {name} breaks its bound or its record form",
    )
    return entry


def _points(
    section: object, config: MetalFusedConfig, names: list[str]
) -> list[dict[str, object]]:
    shapes = config.benchmark.shapes
    _require(
        type(section) is list and len(section) == len(shapes),
        "benchmark points do not cover the shapes",
    )
    sides = [*names, _TORCH]
    rebuilt = []
    for index, (entry, shape) in enumerate(zip(section, shapes)):
        label = f"point {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        if shape.cols is not None:
            lengths = [shape.cols] * shape.rows
        else:
            generator = numpy.random.default_rng([config.seed, 2, index])
            drawn = generator.integers(
                1, shape.max_length + 1, size=shape.rows, dtype=numpy.int64
            )
            lengths = [int(value) for value in drawn]
        layout = _Layout(lengths, 0)
        buffer = _rows(config.seed, 1, index, layout, config.benchmark.inputs)
        output, _ = _reference(buffer, layout)
        compulsory = 8 * layout.elements
        calls = max(1, -(-config.benchmark.target_bytes_per_sample // compulsory))
        _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
        recorded = entry.get("sides")
        checks = entry.get("checks")
        _require(
            type(recorded) is dict and type(checks) is dict,
            f"{label} sides and checks must be objects",
        )
        expected = {
            "index": index,
            "shape": shape.to_mapping(),
            "layout": layout.mapping(),
            "launch": layout.launch(),
            "torch_input": {
                "rows": layout.rows,
                "cols": layout.longest,
                "padded": not layout.mapping()["dense"],
            },
            "input_sha256": _hash(buffer, "<f4"),
            "reference_sha256": _hash(output, "<f8"),
            "compulsory_bytes_per_call": compulsory,
            "side_order": sides,
            "sides": {
                name: _side(
                    recorded.get(name),
                    name,
                    calls,
                    compulsory,
                    None if name == _TORCH else _modeled(name, layout),
                    config,
                )
                for name in sides
            },
            "checks": {
                name: _check(checks.get(name), name, layout.elements) for name in sides
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


def _versus(sides: dict, name: str, other: str, minimum: float):
    if name == other or name not in sides or other not in sides:
        return None
    return _ratio(sides[name], sides[other], minimum)


def _rate(entry: dict, ceiling: float) -> dict[str, object]:
    modeled = entry["modeled_gb_per_second"]
    return {
        "effective_gb_per_second": entry["effective_gb_per_second"],
        "iqr_effective_gb_per_second": entry["iqr_effective_gb_per_second"],
        "modeled_gb_per_second": modeled,
        "effective_fraction_of_ceiling": entry["effective_gb_per_second"] / ceiling,
        "modeled_fraction_of_ceiling": None if modeled is None else modeled / ceiling,
    }


def _speedup(points: list[dict], ceiling: float, minimum: float, version: str):
    entries = []
    for point in points:
        sides = point["sides"]
        rows = []
        for step in PLAN:
            name = step.variant
            to_base = _versus(sides, name, step.base, minimum)
            rows.append(
                {
                    "step": step.step,
                    "variant": name,
                    "base": step.base,
                    "ran": name in sides,
                    "rate": _rate(sides[name], ceiling) if name in sides else None,
                    "to_base": to_base,
                    "to_u0": _versus(sides, name, _U0, minimum),
                    "to_torch": _versus(sides, name, _TORCH, minimum),
                    "predicted": list(step.predicted),
                    "held": None
                    if to_base is None
                    else to_base["verdict"] in step.predicted,
                }
            )
        best = None
        for name in _STEPS:
            if name in sides and (
                best is None
                or sides[name]["effective_gb_per_second"]
                > sides[best]["effective_gb_per_second"]
            ):
                best = name
        entries.append(
            {
                "index": point["index"],
                "shape": point["shape"],
                "torch": {
                    "rate": _rate(sides[_TORCH], ceiling),
                    "to_u0": _versus(sides, _TORCH, _U0, minimum),
                },
                "best": None
                if best is None
                else {
                    "variant": best,
                    "to_u0": _versus(sides, best, _U0, minimum),
                    "to_torch": _ratio(sides[best], sides[_TORCH], minimum),
                },
                "rows": rows,
            }
        )
    hypotheses = []
    for step in PLAN:
        held = [
            (entry["index"], row["held"])
            for entry in entries
            for row in entry["rows"]
            if row["variant"] == step.variant
        ]
        hypotheses.append(
            {
                "step": step.step,
                "variant": step.variant,
                "predicted": list(step.predicted),
                "held_at": [index for index, value in held if value is True],
                "missed_at": [index for index, value in held if value is False],
                "no_verdict_at": [index for index, value in held if value is None],
            }
        )
    return {
        "baseline": _U0,
        "vendor": f"vendor GPU softmax: torch.softmax(dim=-1) on MPS, PyTorch {version}",
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "fraction_rule": FRACTION_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": minimum,
        "ceiling_gb_per_second": ceiling,
        "points": entries,
        "hypotheses": hypotheses,
    }


def _planned(config: MetalFusedConfig) -> dict[str, int]:
    timing = config.benchmark
    cases = config.correctness.cases
    samples = timing.warmup + timing.repeats
    calls = launches = moved = 0
    for index, shape in enumerate(timing.shapes):
        if shape.cols is not None:
            elements = shape.rows * shape.cols
        else:
            generator = numpy.random.default_rng([config.seed, 2, index])
            elements = int(
                generator.integers(
                    1, shape.max_length + 1, size=shape.rows, dtype=numpy.int64
                ).sum()
            )
        work = 8 * elements
        batch = max(1, -(-timing.target_bytes_per_sample // work))
        calls += 5 * samples * batch
        launches += 7 * samples * batch
        moved += 5 * samples * batch * work
    return {
        "correctness_cases": len(cases),
        "correctness_elements": sum(sum(case.lengths) for case in cases),
        "correctness_launches": len(cases) * config.correctness.repeats * 6 + 1,
        "exp_probe_arguments": _PROBE_COUNT,
        "benchmark_points": len(timing.shapes),
        "sides_per_point": 5,
        "timed_samples": 5 * len(timing.shapes) * timing.repeats,
        "calls": calls,
        "kernel_launches": launches,
        "compulsory_bytes": moved,
    }


def _summary(label, names, refused, cases, probe, points, gains, captured, load):
    outputs = [output for case in cases for output in case["outputs"]]

    def same(variant: str, flag: bool) -> list[int]:
        return [
            case["index"]
            for case in cases
            for item in case["same_bits"]
            if item["variant"] == variant and item["same"] is flag
        ]

    return {
        "label": label,
        "steps_checked": names,
        "refused_steps": refused,
        "correctness_cases": len(cases),
        "max_bound_ratio": max(output["max_bound_ratio"] for output in outputs),
        "max_row_sum_ratio": max(output["max_row_sum_ratio"] for output in outputs),
        "identical_repeats": {
            name: all(
                output["identical_repeats"]
                for output in outputs
                if output["kernel"] == name
            )
            for name in names
        },
        "same_bits": [
            {
                "variant": step.variant,
                "as": step.same_bits_as,
                "same_at": same(step.variant, True),
                "differ_at": same(step.variant, False),
            }
            for step in PLAN
            if step.same_bits_as is not None
        ],
        "naive_float32_failed_cases": [
            case["index"] for case in cases if not case["naive_float32_finite"]
        ],
        "exp_probe_max_ulps": probe["max_ulps"],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "index": point["index"],
                "shape": point["shape"],
                "best": None if entry["best"] is None else entry["best"]["variant"],
                "effective_gb_per_second": {
                    name: side["effective_gb_per_second"]
                    for name, side in point["sides"].items()
                },
                "to_base": {
                    row["variant"]: None
                    if row["to_base"] is None
                    else row["to_base"]["verdict"]
                    for row in entry["rows"]
                },
            }
            for point, entry in zip(points, gains["points"])
        ],
        "hypotheses": {
            record["variant"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
            }
            for record in gains["hypotheses"]
        },
        "ceiling_gb_per_second": gains["ceiling_gb_per_second"],
        "warnings": captured["count"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
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


def _ceiling(section: object, measured: bool, environment: dict) -> float:
    _require(type(section) is dict, "the ceiling must be a JSON object")
    try:
        ceiling = BandwidthCeiling(
            **{
                key: value
                for key, value in section.items()
                if key not in ("source", "definition")
            }
        )
    except (MetalFusedError, TypeError) as error:
        raise MetalFusedVerificationError(
            f"the ceiling is malformed: {error}"
        ) from error
    _require(
        _exact(section, ceiling.to_mapping())
        and section["source"] == CEILING_SOURCE
        and section["definition"] == CEILING_DEFINITION,
        "the ceiling record does not match its definition",
    )
    _require(
        not measured or (ceiling.measured and ceiling.same_environment(environment)),
        "a measurement needs a measured ceiling from the same environment",
    )
    return ceiling.gb_per_second


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


def _verify(report: object, supplied: MetalFusedConfig | None, ceiling) -> dict:
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
    _require(
        _exact(report["plan"], {**plan_mapping(), "sha256": plan_sha256()}),
        "the report does not seal the declared plan and its hypotheses",
    )
    _environment(report["environment"], measured)
    ceiling_rate = _ceiling(report["ceiling"], measured, report["environment"])
    if ceiling is not None:
        _require(
            _exact(report["ceiling"], ceiling.to_mapping()),
            "report was not made with the supplied ceiling report",
        )
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    admissions = _shaders(report["shaders"], config.largest_group())
    refusals = [entry for entry in admissions if not entry["admitted"]]
    _require(_exact(report["refusals"], refusals), "refusals do not match the limits")
    names = [entry["variant"] for entry in admissions if entry["admitted"]]
    _require(names != [], "a report needs at least one admitted step")
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    probe = _probe(report["exp_probe"])
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, names)
    expected = {
        "reference": REFERENCE,
        "bound": BOUND,
        "order_terms": dict(ORDER_TERMS),
        "row_sum_bound": ROW_SUM_BOUND,
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
    points = _points(benchmark.get("points"), config, names)
    expected = {
        "device": "mps",
        "inputs": config.benchmark.inputs,
        "synchronization": SYNCHRONIZATION,
        "compulsory": COMPULSORY,
        "traffic_model": TRAFFIC_MODEL,
        "torch_equivalent": TORCH_EQUIVALENT,
        "vendor_bound": VENDOR_BOUND,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    version = report["environment"]["torch_version"]
    gains = _speedup(points, ceiling_rate, config.comparison.min_faster_ratio, version)
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
    summary = _summary(
        label,
        names,
        [entry["variant"] for entry in refusals],
        cases,
        probe,
        points,
        gains,
        captured,
        report["host_load"],
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
        "refused_steps": summary["refused_steps"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "hypotheses": summary["hypotheses"],
        "ceiling_gb_per_second": summary["ceiling_gb_per_second"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object,
    *,
    config: MetalFusedConfig | None = None,
    ceiling: BandwidthCeiling | None = None,
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally against its inputs.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 check, is reported as
    ``MetalFusedVerificationError``.
    """

    if config is not None and not isinstance(config, MetalFusedConfig):
        raise TypeError("config must be a MetalFusedConfig")
    if ceiling is not None and not isinstance(ceiling, BandwidthCeiling):
        raise TypeError("ceiling must be a BandwidthCeiling")
    try:
        return _verify(report, config, ceiling)
    except MetalFusedVerificationError:
        raise
    except MatmulTiledVerificationError as error:
        raise MetalFusedVerificationError(str(error)) from error
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise MetalFusedVerificationError(
            f"report failed verification: {error}"
        ) from error
