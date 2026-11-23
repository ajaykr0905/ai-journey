"""Independent verification of sealed scan reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds what the
seed and the stored digests fix, with its own code. It draws each input
again, sums it in float64 in order, and replays S0 and S1 in float32 in
another form than the run: Hillis-Steele as whole shifted rows, Blelloch as
a list of level sums with prefix sums passed down the tree.

- An output that records the exact integer sums must hash to them with its
  sentinel tail in every run, and its error is 0.
- An output whose first run records the replay's bits must hash to them with
  its tail in that run. The verifier computes the error of that run again,
  whatever the later runs gave. The recorded error is the largest of all
  runs: it must equal that error when every run gave the same bits, and it
  must not be smaller when a later run gave other bits.
- Other bits cannot be rebuilt. An output whose first run records them must
  record another digest, a count of 1 to N differing elements and an error
  within its bound. The digest of a later run with other bits is checked for
  its form only.

Besides the label and rule texts of the run, the verifier uses three modules
of the run that measure nothing: the config parser checks the bounds, the
kernel registry gives each kernel's group, memory and declared depth, and
the declared plan gives the hypotheses that the report must seal, byte for
byte. It derives the levels, guard tails, bounds, launches and pipeline
verdicts again, and each statistic, rate, ratio, verdict, held hypothesis,
run total and summary from the raw samples. The timed outputs have no
digest, so each check after the timing is held to its fields and its bound,
and a side that records the S0 bits must record the S0 error. Every section
must match its rebuild as canonical JSON, so ``1``, ``1.0`` and ``true`` stay
distinct. Timings cannot be observed again, so they are checked for
structure only, and torch is never imported.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement; any ``injected`` backend or timer verifies only as
synthetic. A matching digest shows internal consistency; it cannot prove
that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import math
import re

import numpy

from .gpu_ceiling_source import source_sha256 as ceiling_source_sha256
from .gpu_scan_benchmark import CHECK_RULE, TRAFFIC_MODEL
from .gpu_scan_config import GpuScanConfig, config_from_mapping
from .gpu_scan_plan import COPY, PLAN, VENDOR, plan_mapping, plan_sha256
from .gpu_scan_reference import BOUND_RULE, BOUND_SOURCE, ERROR_RULE, PATTERNS
from .gpu_scan_report import (
    BYTES_RULE,
    COPY_TEXT,
    CORRECTNESS_RULES,
    MEASUREMENT_BOUNDARY,
    REFERENCE,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNTHETIC_BOUNDARY,
)
from .gpu_scan_source import ADD_OFFSETS, KERNELS, GpuScanError, add_mapping
from .gpu_scan_source import library_sha256 as scan_library_sha256
from .gpu_scan_speedup import BEST_RULE, COPY_RULE, HYPOTHESIS_RULE, RATIO_RULE
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
    _hash,
    _launch,
    _limits,
    _load,
    _switches,
    _timing,
    _warnings,
)
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_BLOCK = 1024
_ADD_GROUP = 256
_COPY_GROUP = 256
_COPY_PER_THREAD = 4
_MAX_CALLS = 4096
_SENTINEL = 0x7FC00000
_U32 = 2.0**-24
_U64 = 2.0**-53
_S0 = "scan_s0_hillis_steele"
_S1 = "scan_s1_blelloch"
_NORMAL, _INTEGERS = PATTERNS
_VERDICTS = ("faster", "slower", "no_clear_difference")
_FEWER = "pipeline_allows_fewer_threads_than_the_group"
_BYTES = "static_threadgroup_bytes_differ_from_the_kernel"
_WIDTH = "simd_width_is_not_32"
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
    "correctness",
    "benchmark",
    "speedup",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")


class GpuScanVerificationError(GpuScanError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuScanVerificationError(message)


def _levels(length: int) -> list[int]:
    lengths = [length]
    while lengths[-1] > _BLOCK:
        lengths.append(-(-lengths[-1] // _BLOCK))
    return lengths


def _guard(length: int) -> int:
    return -(-length // _BLOCK) * _BLOCK - length + _BLOCK


def _gamma(count: int, unit: float) -> float:
    product = count * unit
    _require(product < 1.0, "a bound needs k u below 1")
    return product / (1.0 - product)


def _bound(name: str, length: int) -> float:
    count = len(_levels(length))
    depth = count * KERNELS[name].block_depth + count - 1
    wide = _gamma(length, _U64)
    return (_gamma(depth, _U32) + wide) / (1.0 - wide)


def _input(seed: int, stream: int, index: int, length: int, pattern: str):
    generator = numpy.random.default_rng([seed, stream, index])
    if pattern == _NORMAL:
        return generator.standard_normal(length, dtype=numpy.float32)
    return generator.integers(-4, 5, size=length).astype(numpy.float32)


def _rows(values: numpy.ndarray) -> numpy.ndarray:
    rows = numpy.zeros((-(-values.size // _BLOCK), _BLOCK), dtype=numpy.float32)
    rows.reshape(-1)[: values.size] = values
    return rows


def _shifted_rows(values: numpy.ndarray) -> numpy.ndarray:
    """Hillis-Steele: each step adds the whole row shifted right by 2^k."""

    rows = _rows(values)
    for power in range(10):
        shift = 1 << power
        rows = numpy.concatenate(
            [rows[:, :shift], rows[:, shift:] + rows[:, :-shift]], axis=1
        )
    return rows


def _tree_rows(values: numpy.ndarray) -> numpy.ndarray:
    """Blelloch: level sums up the tree, then exclusive prefixes down it."""

    sums = [_rows(values)]
    while sums[-1].shape[1] > 1:
        level = sums[-1]
        sums.append(level[:, 0::2] + level[:, 1::2])
    prefix = numpy.zeros_like(sums[-1])
    for level in reversed(sums[:-1]):
        child = numpy.empty((prefix.shape[0], 2 * prefix.shape[1]), numpy.float32)
        child[:, 0::2] = prefix
        child[:, 1::2] = level[:, 0::2] + prefix
        prefix = child
    return numpy.concatenate([prefix[:, 1:], sums[-1]], axis=1)


def _replay(values: numpy.ndarray, block) -> numpy.ndarray:
    rows = block(values)
    out = rows.reshape(-1)[: values.size].copy()
    totals = rows[:, -1].copy()
    if totals.size > 1:
        offsets = _replay(totals, block)
        out[_BLOCK:] = (
            out[_BLOCK:] + numpy.repeat(offsets[:-1], _BLOCK)[: values.size - _BLOCK]
        )
    return out


_REPLAYS = {_S0: _shifted_rows, _S1: _tree_rows}


def _error(output: numpy.ndarray, reference, scale) -> float:
    difference = numpy.abs(output.astype(numpy.float64) - reference)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        ratio = numpy.where(difference == 0.0, 0.0, difference / scale)
    return float(ratio.max())


def _admission(name: str, limits: dict[str, int]) -> dict[str, object]:
    kernel = KERNELS[name]
    reason = None
    if limits["max_threads_per_threadgroup"] < kernel.threads_per_group:
        reason = _FEWER
    elif limits["static_threadgroup_memory_bytes"] != kernel.threadgroup_bytes:
        reason = _BYTES
    elif (
        kernel.simd_width is not None
        and limits["thread_execution_width"] != kernel.simd_width
    ):
        reason = _WIDTH
    return {
        "name": name,
        "admitted": reason is None,
        "reason": reason,
        "required_threads": kernel.threads_per_group,
        "allowed_threads": limits["max_threads_per_threadgroup"],
        "declared_threadgroup_bytes": kernel.threadgroup_bytes,
        "static_threadgroup_bytes": limits["static_threadgroup_memory_bytes"],
        "required_simd_width": kernel.simd_width,
        "simd_width": limits["thread_execution_width"],
    }


def _shaders(section: object) -> tuple[dict, dict]:
    """Check the shader section; return every pipeline's limits and each verdict."""

    _require(
        type(section) is dict
        and set(section) == {"library_sha256", "kernels", "add", "copy"},
        "shaders must hold the library, the kernels, the add and the copy",
    )
    entries = section["kernels"]
    _require(
        type(entries) is list and len(entries) == len(KERNELS),
        "shaders must list every registered kernel",
    )
    limits, admissions, rebuilt = {}, {}, []
    for name, entry in zip(KERNELS, entries):
        limits[name] = _limits(entry.get("limits") if type(entry) is dict else None)
        admissions[name] = _admission(name, limits[name])
        rebuilt.append(
            {
                **KERNELS[name].to_mapping(),
                "limits": limits[name],
                "admission": admissions[name],
            }
        )
    for helper, key in ((ADD_OFFSETS, "add"), (COPY, "copy")):
        part = section[key]
        limits[helper] = _limits(part.get("limits") if type(part) is dict else None)
        _require(
            limits[helper]["max_threads_per_threadgroup"] >= 256,
            f"the {helper} pipeline must take groups of 256",
        )
    expected = {
        "library_sha256": scan_library_sha256(),
        "kernels": rebuilt,
        "add": {**add_mapping(), "limits": limits[ADD_OFFSETS]},
        "copy": {
            "kernel": "stream_copy",
            "text": COPY_TEXT,
            "source_sha256": ceiling_source_sha256(),
            "group_size": _COPY_GROUP,
            "float4_per_thread": _COPY_PER_THREAD,
            "limits": limits[COPY],
        },
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return limits, admissions


def _plan_levels(name: str, length: int, limits: dict) -> list[dict[str, object]]:
    threads = KERNELS[name].threads_per_group
    lengths = _levels(length)
    return [
        {
            "length": count,
            "blocks": -(-count // _BLOCK),
            "scan": _launch([-(-count // _BLOCK) * threads], [threads], limits[name]),
            "add": None
            if level == len(lengths) - 1
            else _launch([count], [_ADD_GROUP], limits[ADD_OFFSETS]),
        }
        for level, count in enumerate(lengths)
    ]


def _digests(entry: dict, runs: int, name: str) -> list[str]:
    digests = entry.get("run_sha256")
    _require(
        type(digests) is list
        and len(digests) == runs
        and all(isinstance(item, str) and _SHA256.fullmatch(item) for item in digests),
        f"{name} must record one SHA-256 digest for each of {runs} runs",
    )
    return digests


def _kernel_output(
    entry: object,
    name: str,
    length: int,
    runs: int,
    expected: dict[str, object],
) -> dict[str, object]:
    """Check one kernel record; rebuildable bits must hash with the tail."""

    _require(type(entry) is dict, f"{name} output must be a JSON object")
    digests = _digests(entry, runs, name)
    tail = numpy.full(_guard(length), _SENTINEL, dtype=numpy.uint32).view(numpy.float32)
    bound = _bound(name, length)
    error = entry.get("max_relative_error")
    differences = entry.get("replay_differences")
    exact = expected["exact"]
    replay = expected["replays"].get(name)
    if exact is not None:
        rebuilt = _hash(exact, tail)
        _require(
            all(item == rebuilt for item in digests),
            f"{name} must hold the exact integer sums with its tail in every run",
        )
        error = 0.0
    if replay is not None:
        rebuilt = _hash(replay, tail)
        _require(
            type(differences) is int and 0 <= differences <= length,
            f"{name} must count the elements that differ from its replay",
        )
        if differences == 0:
            _require(digests[0] == rebuilt, f"{name} claims the replay bits")
            # Run 1 holds the replay bits, so its error is known whatever the
            # later runs gave, and the largest error of all runs covers it.
            first = _error(replay, expected["reference"], expected["scale"])
            if len(set(digests)) == 1:
                error = first
            else:
                _require(
                    type(error) is float and error >= first,
                    f"{name} records less error than its first run, which "
                    "holds the replay bits",
                )
        else:
            _require(
                digests[0] != rebuilt,
                f"{name} records other bits, so it needs another digest",
            )
    _require(
        type(error) is float and math.isfinite(error) and 0.0 <= error <= bound,
        f"{name} needs a finite error within its bound",
    )
    rebuilt_entry = {
        "side": name,
        "buffer_sha256": digests[0],
        "run_sha256": digests,
        "deterministic": len(set(digests)) == 1,
        "guard_elements": _guard(length),
        "max_relative_error": error,
        "bound": bound,
        "bound_ratio": error / bound,
        "replay_bits": None if replay is None else differences == 0,
        "replay_differences": None if replay is None else differences,
        "exact_bits": None if exact is None else True,
    }
    _require(_exact(entry, rebuilt_entry), f"{name} does not rebuild from its case")
    return entry


def _vendor_output(
    entry: object, length: int, runs: int, expected: dict[str, object]
) -> dict[str, object]:
    _require(type(entry) is dict, "the torch.cumsum output must be a JSON object")
    digests = _digests(entry, runs, VENDOR)
    error = entry.get("max_relative_error")
    exact = expected["exact"]
    exact_bits = entry.get("exact_bits")
    if exact is None:
        _require(exact_bits is None, "torch.cumsum has no exact sums on this case")
    else:
        _require(type(exact_bits) is bool, "torch.cumsum must record its exact bits")
        rebuilt = _hash(exact)
        _require(
            exact_bits == all(item == rebuilt for item in digests),
            "the exact bits of torch.cumsum do not match its digests",
        )
        if exact_bits:
            error = 0.0
    _require(
        type(error) is float and math.isfinite(error) and error >= 0.0,
        "torch.cumsum needs a finite measured error",
    )
    rebuilt_entry = {
        "side": VENDOR,
        "buffer_sha256": digests[0],
        "run_sha256": digests,
        "deterministic": len(set(digests)) == 1,
        "guard_elements": None,
        "max_relative_error": error,
        "bound": None,
        "bound_ratio": None,
        "replay_bits": None,
        "replay_differences": None,
        "exact_bits": exact_bits,
    }
    _require(_exact(entry, rebuilt_entry), "torch.cumsum does not rebuild")
    return entry


def _cases(
    section: object, config: GpuScanConfig, limits: dict, names: list[str]
) -> list[dict[str, object]]:
    cases = config.correctness.cases
    runs = config.correctness.runs
    _require(
        type(section) is list and len(section) == len(cases),
        "correctness cases do not cover the config",
    )
    for index, (entry, (length, pattern)) in enumerate(zip(section, cases)):
        label = f"case {index}"
        _require(type(entry) is dict, f"{label} must be a JSON object")
        values = _input(config.seed, 0, index, length, pattern)
        wide = values.astype(numpy.float64)
        reference = numpy.add.accumulate(wide)
        scale = numpy.add.accumulate(numpy.abs(wide))
        exact = (
            numpy.add.accumulate(values.astype(numpy.int64)).astype(numpy.float32)
            if pattern == _INTEGERS
            else None
        )
        replays = {
            name: _replay(values, _REPLAYS[name]) for name in names if name in _REPLAYS
        }
        expected = {
            "exact": exact,
            "replays": replays,
            "reference": reference,
            "scale": scale,
        }
        outputs = entry.get("outputs")
        _require(
            type(outputs) is list and len(outputs) == len(names) + 1,
            f"{label} must record each kernel that ran and torch.cumsum",
        )
        for output, name in zip(outputs, names):
            _kernel_output(output, name, length, runs, expected)
        _vendor_output(outputs[-1], length, runs, expected)
        rebuilt = {
            "index": index,
            "length": length,
            "pattern": pattern,
            "levels": _levels(length),
            "launches": {name: _plan_levels(name, length, limits) for name in names},
            "input_sha256": _hash(values),
            "reference_sha256": _hash(reference),
            "replay_sha256": {name: _hash(replay) for name, replay in replays.items()},
            "exact_sha256": None if exact is None else _hash(exact),
            "outputs": outputs,
        }
        _require(_exact(entry, rebuilt), f"{label} does not rebuild from the seed")
    return section


def _calls(bytes_per_call: int, target: int) -> int:
    calls = max(1, -(-target // bytes_per_call))
    _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
    return calls


def _modeled(length: int) -> int:
    lengths = _levels(length)
    total = 0
    for level, count in enumerate(lengths):
        blocks = -(-count // _BLOCK)
        total += 4 * (2 * count + blocks)
        if level < len(lengths) - 1:
            total += 4 * (2 * (count - min(count, _BLOCK)) + blocks - 1)
    return total


def _side(entry: object, name: str, calls: int, per_call: int, config) -> dict:
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


def _checks(entry: object, names: list[str], length: int) -> dict:
    _require(
        type(entry) is dict and set(entry) == {*names, VENDOR, COPY},
        "a point must check every side after the timing",
    )
    for name in (*names, VENDOR):
        check = entry[name]
        bound = None if name == VENDOR else _bound(name, length)
        _require(
            type(check) is dict
            and set(check)
            == {"max_relative_error", "bound", "bound_ratio", "same_bits_as_s0"},
            f"{name} check fields do not match",
        )
        error = check["max_relative_error"]
        _require(
            type(error) is float
            and math.isfinite(error)
            and error >= 0.0
            and (bound is None or error <= bound),
            f"the timed output of {name} must stay within its bound",
        )
        same = check["same_bits_as_s0"]
        _require(
            (same is None) == (_S0 not in names)
            and (same is None or type(same) is bool)
            and (name != _S0 or same is True),
            f"{name} must compare its bits with S0 when S0 ran",
        )
        # Equal bits give equal errors, so a record cannot claim both.
        if same:
            _require(
                error == entry[_S0]["max_relative_error"],
                f"{name} cannot hold the S0 bits with another error",
            )
        _require(
            _exact(
                check,
                {
                    "max_relative_error": error,
                    "bound": bound,
                    "bound_ratio": None if bound is None else error / bound,
                    "same_bits_as_s0": same,
                },
            ),
            f"{name} check does not recompute",
        )
    _require(
        _exact(entry[COPY], {"same_bits_as_input": True}),
        "the copy must hold its input",
    )
    return entry


def _points(
    section: object, config: GpuScanConfig, limits: dict, names: list[str]
) -> list[dict[str, object]]:
    lengths = config.benchmark.lengths
    _require(
        type(section) is list and len(section) == len(lengths),
        "benchmark points do not cover the sweep",
    )
    sides = [*names, VENDOR, COPY]
    rebuilt = []
    for index, (entry, length) in enumerate(zip(section, lengths)):
        _require(type(entry) is dict, f"point {index} must be a JSON object")
        per_call = 8 * length
        copy_float4 = -(-length // 4)
        copy_per_call = 32 * copy_float4
        calls = _calls(per_call, config.benchmark.target_bytes_per_sample)
        recorded = entry.get("sides")
        _require(type(recorded) is dict, f"point {index} sides must be an object")
        walkers = -(-copy_float4 // _COPY_PER_THREAD)
        expected = {
            "index": index,
            "length": length,
            "levels": _levels(length),
            "dispatches": 2 * len(_levels(length)) - 1,
            "bytes_per_call": per_call,
            "copy_bytes_per_call": copy_per_call,
            "modeled_bytes_per_call": _modeled(length),
            "side_order": sides,
            "launches": {name: _plan_levels(name, length, limits) for name in names},
            "copy_launch": _launch([walkers], [_COPY_GROUP], limits[COPY]),
            "sides": {
                name: _side(
                    recorded.get(name),
                    name,
                    calls,
                    copy_per_call if name == COPY else per_call,
                    config,
                )
                for name in sides
            },
            "checks": _checks(entry.get("checks"), names, length),
        }
        _require(
            _exact(entry, expected),
            f"point {index} at {length} does not recompute from its samples",
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


def _rate(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {
        "median_gb_per_second": entry["median_gb_per_second"],
        "iqr_gb_per_second": entry["iqr_gb_per_second"],
    }


def _pair(entry: dict | None, other: dict | None, minimum: float) -> dict | None:
    return None if entry is None or other is None else _ratio(entry, other, minimum)


def _speedup(points: list[dict], minimum: float, torch_version: str) -> dict:
    entries = []
    for point in points:
        sides = point["sides"]
        vendor, copy = sides[VENDOR], sides[COPY]
        copy_rate = copy["median_gb_per_second"]
        best = None
        for name in KERNELS:
            if name in sides and (
                best is None
                or sides[name]["median_gb_per_second"]
                > sides[best]["median_gb_per_second"]
            ):
                best = name
        rows = []
        for step in PLAN:
            entry, base = sides.get(step.kernel), sides.get(step.base)
            to_base = _pair(entry, base, minimum)
            predicted = dict(step.predictions).get(point["length"])
            modeled = (
                None
                if entry is None
                else entry["calls_per_sample"]
                * point["modeled_bytes_per_call"]
                / entry["timing"]["median_ns"]
            )
            rows.append(
                {
                    "step": step.step,
                    "kernel": step.kernel,
                    "base": step.base,
                    "ran": entry is not None,
                    "rate": _rate(entry),
                    "to_base": to_base,
                    "to_s0": None
                    if step.kernel == _S0
                    else _pair(entry, sides.get(_S0), minimum),
                    "to_vendor": _pair(entry, vendor, minimum),
                    "fraction_of_copy": None
                    if entry is None
                    else entry["median_gb_per_second"] / copy_rate,
                    "modeled_gb_per_second": modeled,
                    "modeled_fraction_of_copy": None
                    if modeled is None
                    else modeled / copy_rate,
                    "predicted": None if predicted is None else list(predicted),
                    "held": None
                    if to_base is None or predicted is None
                    else to_base["verdict"] in predicted,
                }
            )
        entries.append(
            {
                "index": point["index"],
                "length": point["length"],
                "vendor": {
                    **_rate(vendor),
                    "fraction_of_copy": vendor["median_gb_per_second"] / copy_rate,
                },
                "copy": _rate(copy),
                "best": None
                if best is None
                else {
                    "kernel": best,
                    "to_vendor": _ratio(sides[best], vendor, minimum),
                    "fraction_of_copy": sides[best]["median_gb_per_second"] / copy_rate,
                },
                "rows": rows,
            }
        )
    hypotheses = []
    for step in PLAN:
        rows = [
            (entry["index"], row)
            for entry in entries
            for row in entry["rows"]
            if row["kernel"] == step.kernel
        ]
        hypotheses.append(
            {
                "step": step.step,
                "kernel": step.kernel,
                "base": step.base,
                "held_at": [index for index, row in rows if row["held"] is True],
                "missed_at": [index for index, row in rows if row["held"] is False],
                "no_verdict_at": [
                    index for index, row in rows if row["to_base"] is None
                ],
                "no_prediction_at": [
                    index
                    for index, row in rows
                    if row["to_base"] is not None and row["predicted"] is None
                ],
            }
        )
    return {
        "vendor": f"vendor GPU scan: torch.cumsum on MPS, PyTorch {torch_version}",
        "copy": COPY,
        "ratio_rule": RATIO_RULE,
        "hypothesis_rule": HYPOTHESIS_RULE,
        "copy_rule": COPY_RULE,
        "best_rule": BEST_RULE,
        "min_faster_ratio": minimum,
        "points": entries,
        "hypotheses": hypotheses,
    }


def _planned(config: GpuScanConfig) -> dict[str, int]:
    sweep = config.benchmark
    sides = len(KERNELS) + 2
    samples = sides * (sweep.warmup + sweep.repeats)
    batches = [
        _calls(8 * length, sweep.target_bytes_per_sample) for length in sweep.lengths
    ]
    cases = config.correctness.cases
    return {
        "correctness_cases": len(cases),
        "case_elements": sum(length for length, _ in cases),
        "correctness_runs": len(cases) * (len(KERNELS) + 1) * config.correctness.runs,
        "benchmark_points": len(sweep.lengths),
        "sides_per_point": sides,
        "timed_samples": sides * len(sweep.lengths) * sweep.repeats,
        "calls": sum(samples * batch for batch in batches),
        "bytes": sum(
            samples * batch * 8 * length
            for batch, length in zip(batches, sweep.lengths)
        ),
    }


def _summary(
    label: str,
    cases: list[dict[str, object]],
    points: list[dict[str, object]],
    gains: dict[str, object],
    refused: list[str],
    captured: dict[str, object],
    load: dict[str, object],
) -> dict[str, object]:
    sides = [output["side"] for output in cases[0]["outputs"]]
    kernels = sides[:-1]
    outputs = [(case["index"], output) for case in cases for output in case["outputs"]]
    ratios = [
        output["bound_ratio"] for _, output in outputs if output["side"] != VENDOR
    ]
    return {
        "label": label,
        "correctness_cases": len(cases),
        "kernels_checked": kernels,
        "refused_kernels": refused,
        "max_bound_ratio": max(ratios) if ratios else None,
        "cases_with_other_bits": {
            name: [
                index
                for index, output in outputs
                if output["side"] == name and output["replay_bits"] is False
            ]
            for name in kernels
            if name in _REPLAYS
        },
        "nondeterministic_cases": {
            name: [
                index
                for index, output in outputs
                if output["side"] == name and not output["deterministic"]
            ]
            for name in sides
        },
        "vendor_max_relative_error": max(
            output["max_relative_error"]
            for _, output in outputs
            if output["side"] == VENDOR
        ),
        "vendor_missed_exact_at": [
            index
            for index, output in outputs
            if output["side"] == VENDOR and output["exact_bits"] is False
        ],
        "benchmark_points": len(points),
        "timed_samples": sum(
            len(side["samples_ns"])
            for point in points
            for side in point["sides"].values()
        ),
        "points": [
            {
                "length": entry["length"],
                "best": None if entry["best"] is None else entry["best"]["kernel"],
                "best_to_vendor": None
                if entry["best"] is None
                else entry["best"]["to_vendor"]["ratio"],
                "best_fraction_of_copy": None
                if entry["best"] is None
                else entry["best"]["fraction_of_copy"],
                "vendor_gb_per_second": entry["vendor"]["median_gb_per_second"],
                "copy_gb_per_second": entry["copy"]["median_gb_per_second"],
            }
            for entry in gains["points"]
        ],
        "hypotheses": {
            record["kernel"]: {
                "held": len(record["held_at"]),
                "missed": len(record["missed_at"]),
                "no_verdict": len(record["no_verdict_at"]),
                "no_prediction": len(record["no_prediction_at"]),
            }
            for record in gains["hypotheses"]
        },
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


def _verify(report: object, supplied: GpuScanConfig | None) -> dict[str, object]:
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
        "the report does not seal the declared plan and its kernel digests",
    )
    _environment(report["environment"], measured)
    _switches(report["torch_mps_switches"])
    _load(report["host_load"])
    captured = _warnings(report["warnings"])
    limits, admissions = _shaders(report["shaders"])
    refusals = [entry for entry in admissions.values() if not entry["admitted"]]
    _require(_exact(report["refusals"], refusals), "refusals do not match the limits")
    names = [name for name in KERNELS if admissions[name]["admitted"]]
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    cases = _cases(correctness.get("cases"), config, limits, names)
    expected = {
        "reference": REFERENCE,
        "patterns": list(PATTERNS),
        "error_rule": ERROR_RULE,
        "bound_rule": BOUND_RULE,
        "bound_source": BOUND_SOURCE,
        "sentinel_float32_bits": _SENTINEL,
        "rules": CORRECTNESS_RULES,
        "runs": config.correctness.runs,
        "cases": cases,
    }
    _require(
        _exact(correctness, expected),
        "correctness section does not match the lab's reference and bound",
    )
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    points = _points(benchmark.get("points"), config, limits, names)
    expected = {
        "device": "mps",
        "bytes_per_element": 8,
        "bytes_rule": BYTES_RULE,
        "synchronization": SYNCHRONIZATION,
        "traffic_model": TRAFFIC_MODEL,
        "check_rule": CHECK_RULE,
        "points": points,
    }
    _require(_exact(benchmark, expected), "benchmark section does not recompute")
    torch_version = report["environment"]["torch_version"]
    gains = _speedup(points, config.comparison.min_faster_ratio, torch_version)
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
        cases,
        points,
        gains,
        [entry["name"] for entry in refusals],
        captured,
        report["host_load"],
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
        "correctness_cases": summary["correctness_cases"],
        "benchmark_points": summary["benchmark_points"],
        "timed_samples": summary["timed_samples"],
        "refused_kernels": summary["refused_kernels"],
        "max_bound_ratio": summary["max_bound_ratio"],
        "cases_with_other_bits": summary["cases_with_other_bits"],
        "nondeterministic_cases": summary["nondeterministic_cases"],
        "points": summary["points"],
        "hypotheses": summary["hypotheses"],
        "warnings": summary["warnings"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object, *, config: GpuScanConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section and a
    failure in a shared Day 76 check, is reported as
    ``GpuScanVerificationError``.
    """

    if config is not None and not isinstance(config, GpuScanConfig):
        raise TypeError("config must be a GpuScanConfig")
    try:
        return _verify(report, config)
    except GpuScanVerificationError:
        raise
    except MatmulTiledVerificationError as error:
        raise GpuScanVerificationError(str(error)) from error
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise GpuScanVerificationError(
            f"report failed verification: {error}"
        ) from error
