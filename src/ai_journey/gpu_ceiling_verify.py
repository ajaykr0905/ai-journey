"""Independent verification of sealed ceiling reports, on the CPU alone.

A device output cannot be observed again, so verification rebuilds every
correctness fact from the seed with its own code. The FMA chains are rebuilt
in integers: a start ``m * 2**-23`` times ``a = A * 2**-24`` plus
``b = B * 2**-35`` is ``(m * A + B * 2**12) * 2**-47`` exactly, and rounding
that integer to 24 significant bits, half to even, is the fused result. The
run's own reference uses float64 instead, so the two methods check each
other. Copies are rebuilt from the seed, read-reduce partials with a
weighted count, and bank sums with Python integers. Each rebuilt buffer,
with its sentinel tail, must hash to the stored digest.

Timings cannot be observed again either, so they are checked for structure
only: sample counts, positive integers, and the arithmetic of the
statistics, rates, batch sizes, fits, ceilings, ratios, run totals and
summary derived from them. Launch geometry, planned work, the shader source
and its digest, the pipeline limits, the environment tokens, the load
averages and the labels are checked too. Every section must match the
rebuilt one byte for byte as canonical JSON, so ``1``, ``1.0`` and ``true``
stay distinct. No measuring module is called to recompute anything (the
shared config parser only validates bounds), and torch is never imported,
so a host without torch or MPS can verify a report.

Only a ``time.perf_counter_ns`` report from ``torch.mps.compile_shader``
verifies as a measurement; any ``injected`` backend or timer verifies only
as synthetic. A matching digest shows internal consistency; it cannot prove
that a timing was observed or that a GPU ran.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from hashlib import sha256

import numpy

from .gpu_ceiling_config import GpuCeilingConfig, config_from_mapping
from .gpu_ceiling_report import (
    CEILING_DEFINITION,
    DEVICE,
    FMA_RULE,
    INJECTED_BACKEND,
    LOAD_AVERAGE_SOURCE,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_LABEL,
    MEASUREMENT_SCOPE,
    REFERENCE,
    REPORT_DIGEST_FIELD,
    REPORT_KIND,
    SYNCHRONIZATION,
    SYNTHETIC_BOUNDARY,
    SYNTHETIC_LABEL,
    SYNTHETIC_SCOPE,
)
from .gpu_ceiling_source import SHADER_SOURCE, GpuCeilingError
from .kernel_metal_report import ENVIRONMENT_FIELDS
from .kernel_metal_runtime import BACKEND
from .memory_access_benchmark import INJECTED_TIMER, PERCENTILE_METHOD, TIMER

_NS = 1_000_000_000
_MAX_CALLS = 4096
_SENTINEL = 0x7FC00000
_STREAM_GUARD = 32
_LANES = 32
_START_COLUMNS = 32
_MULTIPLIER = 2**24 - 4173
_ADDEND = 12561923
_STEPS_PER_TRIP = 8
_REDUCE_MAX = 1023
_TILE_ROWS = 112
_LOADS = 16
_SHIFTS = 64
_WORD_HASH = 2654435761
_BANK_WIDTH = 4
_BANKS = 32
_FIXED_LIMIT = 0.05
_SLOWDOWN_LIMIT = 1.25
_STREAM_INDEX = {"correctness": 0, "benchmark": 1}
_SEEDS = {"fma": 2, "stream_copy": 3, "stream_read_reduce": 4}
_CHUNK = 1 << 22
_CHAINS = (1, 2, 4, 8, 16, 32)
_ROWS = (32, 33)
_COPY, _REDUCE = "stream_copy", "stream_read_reduce"
_KERNELS = [
    *(
        {
            "name": f"fma_chains_{chains}",
            "family": "fma",
            "buffers": ["out", "start", "coefficients", "n", "trips"],
            "scalar_buffers": [3, 4],
        }
        for chains in _CHAINS
    ),
    *(
        {
            "name": name,
            "family": "bandwidth",
            "buffers": ["out", "in", "n", "threads"],
            "scalar_buffers": [2, 3],
        }
        for name in (_COPY, _REDUCE)
    ),
    *(
        {
            "name": f"bank_rows_{words}",
            "family": "bank",
            "buffers": ["out", "n", "stride", "trips"],
            "scalar_buffers": [1, 2, 3],
        }
        for words in _ROWS
    ),
]
_BANK_MODEL = (
    "CUDA shared memory as the Day 74 memory-access lab models it: 32 banks of "
    "4-byte words, conflict degree = most distinct words one bank serves"
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
    "host_load",
    "shaders",
    "planned_work",
    "correctness",
    "benchmark",
    "ceilings",
    "run",
    "summary",
    REPORT_DIGEST_FIELD,
}
_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GpuCeilingVerificationError(GpuCeilingError):
    """Raised when a sealed report does not survive independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuCeilingVerificationError(message)


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


def _tail(words: int) -> numpy.ndarray:
    return numpy.full(words, _SENTINEL, dtype=numpy.uint32)


def _generator(seed: int, family: str, index: int) -> numpy.random.Generator:
    return numpy.random.default_rng([seed, _SEEDS[family], index])


def _start_significands(seed: int) -> numpy.ndarray:
    generator = _generator(seed, "fma", 0)
    return generator.integers(
        2**23, 2**24, size=(_LANES, _START_COLUMNS), dtype=numpy.int64
    )


def _fused_significands(
    start: numpy.ndarray, checkpoints: list[int]
) -> dict[int, numpy.ndarray]:
    """Step every chain in integers, rounding half to even at 24 bits."""

    values = start.copy()
    tables, done = {}, 0
    half, mask = 1 << 23, (1 << 24) - 1
    for target in sorted(set(checkpoints)):
        for _ in range(target - done):
            exact = values * _MULTIPLIER + (_ADDEND << 12)
            quotient, remainder = exact >> 24, exact & mask
            up = (remainder > half) | ((remainder == half) & (quotient & 1 == 1))
            values = quotient + up
        _require(
            bool(numpy.all((values >= 1 << 23) & (values < 1 << 24))),
            "an FMA chain left the binade [1, 2)",
        )
        done = target
        tables[target] = values.copy()
    return tables


def _float32(significands: numpy.ndarray) -> numpy.ndarray:
    return (significands.astype(numpy.float64) * 2.0**-23).astype(numpy.float32)


def _fma_words(significands: numpy.ndarray, chains: int) -> numpy.ndarray:
    bits = _float32(significands[:, :chains]).view(numpy.uint32)
    words = numpy.zeros(_LANES, dtype=numpy.uint32)
    for column in range(chains):
        words ^= bits[:, column]
    return words


def _spread(words: numpy.ndarray, threads: int) -> numpy.ndarray:
    return numpy.tile(words, -(-threads // _LANES))[:threads]


def _copy_source(seed: int, index: int, count: int) -> numpy.ndarray:
    generator = _generator(seed, _COPY, index)
    return generator.standard_normal(4 * count, dtype=numpy.float32)


def _reduce_source(seed: int, index: int, count: int) -> numpy.ndarray:
    generator = _generator(seed, _REDUCE, index)
    values = generator.integers(0, _REDUCE_MAX + 1, size=4 * count, dtype=numpy.uint16)
    return values.astype(numpy.float32)


def _partials(source: numpy.ndarray, count: int, walkers: int) -> numpy.ndarray:
    """Weight each float4 sum into partial ``e % walkers`` with a count."""

    totals = numpy.zeros(walkers, dtype=numpy.float64)
    for start in range(0, count, _CHUNK):
        stop = min(start + _CHUNK, count)
        sums = source[4 * start : 4 * stop].reshape(-1, 4).sum(axis=1, dtype="f8")
        owners = numpy.arange(start, stop, dtype=numpy.int64) % walkers
        totals += numpy.bincount(owners, weights=sums, minlength=walkers)
    return totals.astype(numpy.float32)


def _first_words(stride: int, row_words: int) -> list[int]:
    return [
        (lane * stride) // 32 * row_words + (lane * stride) % 32
        for lane in range(_LANES)
    ]


def _bank_words(stride: int, row_words: int, trips: int) -> numpy.ndarray:
    """Sum the tile words each lane reads, in Python integers, then wrap."""

    tile = [word * _WORD_HASH % 2**32 for word in range(_TILE_ROWS * row_words)]
    cycles, rest = divmod(trips, _SHIFTS)
    sums = []
    for first in _first_words(stride, row_words):
        per_shift = [
            sum(tile[first + (shift + load) * row_words] for load in range(_LOADS))
            for shift in range(_SHIFTS)
        ]
        sums.append((sum(per_shift) * cycles + sum(per_shift[:rest])) % 2**32)
    return numpy.array(sums, dtype=numpy.uint32)


def _model_degree(stride: int, row_words: int) -> int:
    banks: dict[int, set[int]] = {}
    for word in _first_words(stride, row_words):
        banks.setdefault(word % _BANKS, set()).add(word)
    return max(len(words) for words in banks.values())


def _launch(threads: int, group: int, limits: dict) -> dict[str, object]:
    groups = -(-threads // group)
    width = limits["thread_execution_width"]
    simd = -(-group // width)
    _require(
        group <= limits["max_threads_per_threadgroup"],
        "a threadgroup exceeds its pipeline's recorded limit",
    )
    return {
        "extent": [threads],
        "group_size": [group],
        "groups": [groups],
        "threads": [groups * group],
        "threads_per_group": group,
        "launched_threads": groups * group,
        "active_threads": threads,
        "padding_threads": groups * group - threads,
        "simd_groups_per_threadgroup": simd,
        "idle_lanes_per_threadgroup": simd * width - group,
    }


def _limits(entry: object, name: str) -> dict[str, int]:
    _require(
        type(entry) is dict
        and set(entry)
        == {
            "max_threads_per_threadgroup",
            "thread_execution_width",
            "static_threadgroup_memory_bytes",
        }
        and all(type(value) is int for value in entry.values()),
        f"{name}: pipeline limits must be three integers",
    )
    most = entry["max_threads_per_threadgroup"]
    _require(
        1 <= most <= 1024
        and 1 <= entry["thread_execution_width"] <= most
        and 0 <= entry["static_threadgroup_memory_bytes"] <= 1 << 20,
        f"{name}: pipeline limits are outside what the API can report",
    )
    if name.startswith("bank_rows_"):
        words = int(name.rsplit("_", 1)[1])
        _require(
            entry["static_threadgroup_memory_bytes"] == _TILE_ROWS * words * 4
            and entry["thread_execution_width"] == _LANES,
            f"{name}: a bank pipeline must hold its tile and run 32-lane groups",
        )
    return entry


def _shaders(section: object) -> dict[str, dict[str, int]]:
    _require(type(section) is dict, "shaders must be a JSON object")
    _require(
        _exact(section.get("source"), SHADER_SOURCE),
        "the report was not made with this lab's Metal source",
    )
    kernels = section.get("kernels")
    _require(
        type(kernels) is list and len(kernels) == len(_KERNELS),
        "shaders must list every ceiling kernel",
    )
    limits = {
        spec["name"]: _limits(
            entry.get("limits") if type(entry) is dict else None, spec["name"]
        )
        for spec, entry in zip(_KERNELS, kernels)
    }
    expected = {
        "source": SHADER_SOURCE,
        "source_sha256": sha256(SHADER_SOURCE.encode("ascii")).hexdigest(),
        "kernels": [{**spec, "limits": limits[spec["name"]]} for spec in _KERNELS],
    }
    _require(_exact(section, expected), "shader section does not recompute")
    return limits


def _calls(bytes_per_call: int, target: int) -> int:
    calls = max(1, -(-target // bytes_per_call))
    _require(calls <= _MAX_CALLS, f"a sample exceeds {_MAX_CALLS} calls")
    return calls


def _stream_bytes(kernel: str, count: int, per_thread: int) -> int:
    if kernel == _COPY:
        return 32 * count
    return 16 * count + 4 * -(-count // per_thread)


def _planned(config: GpuCeilingConfig) -> dict[str, int]:
    fma, bandwidth, bank = config.fma, config.bandwidth, config.bank
    samples = config.timing.warmup + config.timing.repeats
    per_thread = {
        _COPY: bandwidth.copy_float4_per_thread,
        _REDUCE: bandwidth.reduce_float4_per_thread,
    }
    streams = [
        _stream_bytes(kernel, size // 16, per_thread[kernel])
        for kernel in (_COPY, _REDUCE)
        for size in bandwidth.buffer_bytes
    ]
    batches = [
        _calls(per_call, bandwidth.target_bytes_per_sample) for per_call in streams
    ]
    fma_points = len(fma.chain_counts) * len(fma.fma_steps)
    bank_points = 2 * len(bank.strides)
    return {
        "correctness_cases": len(fma.chain_counts) + 2 + 2 * len(bank.strides),
        "fma_points": fma_points,
        "stream_points": len(streams),
        "bank_points": bank_points,
        "timed_samples": config.timing.repeats
        * (fma_points + len(streams) + bank_points),
        "calls": samples * (fma_points + bank_points + sum(batches)),
        "flops": samples
        * sum(
            2 * fma.threads * chains * steps
            for chains in fma.chain_counts
            for steps in fma.fma_steps
        ),
        "stream_bytes": samples
        * sum(per_call * batch for per_call, batch in zip(streams, batches)),
        "bank_accesses": samples * bank_points * bank.threads * bank.trips * _LOADS,
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


def _load(section: object) -> dict[str, object]:
    def averages(value: object) -> bool:
        return (
            type(value) is list
            and len(value) == 3
            and all(
                type(item) is float and math.isfinite(item) and item >= 0
                for item in value
            )
        )

    _require(
        type(section) is dict
        and set(section) == {"source", "before", "after"}
        and section["source"] == LOAD_AVERAGE_SOURCE
        and averages(section["before"])
        and averages(section["after"]),
        "host_load must hold two readings of three nonnegative load averages",
    )
    return section


def _fma_cases(
    section: object, config: GpuCeilingConfig, limits: dict, tables: dict
) -> list[dict[str, object]]:
    chains_list = config.fma.chain_counts
    _require(
        type(section) is list and len(section) == len(chains_list),
        "fma cases do not cover the config's chain counts",
    )
    cases = config.correctness
    start = _start_significands(config.seed)
    rebuilt = []
    for entry, chains in zip(section, chains_list):
        name = f"fma_chains_{chains}"
        launch = _launch(cases.threads, config.fma.group_size, limits[name])
        expected = _spread(_fma_words(tables[cases.fma_steps], chains), cases.threads)
        rebuilt.append(
            {
                "kernel": name,
                "chains": chains,
                "fma_steps": cases.fma_steps,
                "threads": cases.threads,
                "launch": launch,
                "start_sha256": _hash(_float32(start)),
                "expected_sha256": _hash(expected),
                "buffer_sha256": _hash(expected, _tail(launch["padding_threads"])),
                "guard_words": launch["padding_threads"],
            }
        )
        _require(_exact(entry, rebuilt[-1]), f"{name} case does not rebuild")
    return rebuilt


def _stream_cases(
    section: object, config: GpuCeilingConfig, limits: dict
) -> list[dict[str, object]]:
    _require(type(section) is list and len(section) == 2, "stream cases are two")
    count = config.correctness.stream_float4
    rebuilt = []
    for entry, kernel in zip(section, (_COPY, _REDUCE)):
        per_thread = (
            config.bandwidth.copy_float4_per_thread
            if kernel == _COPY
            else config.bandwidth.reduce_float4_per_thread
        )
        walkers = -(-count // per_thread)
        launch = _launch(walkers, config.bandwidth.group_size, limits[kernel])
        index = _STREAM_INDEX["correctness"]
        if kernel == _COPY:
            source = _copy_source(config.seed, index, count)
            expected, guard = source, 4 * _STREAM_GUARD
        else:
            source = _reduce_source(config.seed, index, count)
            expected = _partials(source, count, walkers)
            guard = launch["padding_threads"]
        rebuilt.append(
            {
                "kernel": kernel,
                "float4_count": count,
                "float4_per_thread": per_thread,
                "threads": walkers,
                "launch": launch,
                "source_sha256": _hash(source),
                "expected_sha256": _hash(expected),
                "buffer_sha256": _hash(expected, _tail(guard)),
                "guard_words": guard,
            }
        )
        _require(_exact(entry, rebuilt[-1]), f"{kernel} case does not rebuild")
    return rebuilt


def _bank_cases(
    section: object, config: GpuCeilingConfig, limits: dict
) -> list[dict[str, object]]:
    plan = [(words, stride) for words in _ROWS for stride in config.bank.strides]
    _require(
        type(section) is list and len(section) == len(plan),
        "bank cases do not cover both layouts at every stride",
    )
    cases = config.correctness
    rebuilt = []
    for entry, (words, stride) in zip(section, plan):
        name = f"bank_rows_{words}"
        launch = _launch(cases.threads, config.bank.group_size, limits[name])
        expected = _spread(_bank_words(stride, words, cases.bank_trips), cases.threads)
        rebuilt.append(
            {
                "kernel": name,
                "row_words": words,
                "stride": stride,
                "trips": cases.bank_trips,
                "threads": cases.threads,
                "launch": launch,
                "expected_sha256": _hash(expected),
                "buffer_sha256": _hash(expected, _tail(launch["padding_threads"])),
                "guard_words": launch["padding_threads"],
            }
        )
        _require(
            _exact(entry, rebuilt[-1]),
            f"{name} case at stride {stride} does not rebuild",
        )
    return rebuilt


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


def _samples(entry: object, config: GpuCeilingConfig, calls: int, label: str):
    repeats = config.timing.repeats
    samples = entry.get("samples_ns") if type(entry) is dict else None
    _require(
        type(samples) is list
        and len(samples) == repeats
        and all(type(sample) is int and sample > 0 for sample in samples),
        f"{label}: samples_ns must hold {repeats} positive integers",
    )
    timing = _timing(samples)
    return timing, {
        "calls_per_sample": calls,
        "warmup": config.timing.warmup,
        "repeats": repeats,
        "samples_ns": samples,
        "timing": timing,
        "median_ns_per_call": timing["median_ns"] / calls,
        "iqr_ns_per_call": timing["iqr_ns"] / calls,
    }


def _fma_points(
    section: object, config: GpuCeilingConfig, limits: dict, tables: dict
) -> list[dict[str, object]]:
    fma = config.fma
    plan = [(chains, steps) for chains in fma.chain_counts for steps in fma.fma_steps]
    _require(
        type(section) is list and len(section) == len(plan),
        "fma points do not cover the sweep",
    )
    rebuilt = []
    for entry, (chains, steps) in zip(section, plan):
        name = f"fma_chains_{chains}"
        flops = 2 * fma.threads * chains * steps
        timing, mapping = _samples(entry, config, 1, f"{name} at {steps} steps")
        words = _spread(_fma_words(tables[steps], chains), fma.threads)
        rebuilt.append(
            {
                "kernel": name,
                "chains": chains,
                "fma_steps": steps,
                "threads": fma.threads,
                "launch": _launch(fma.threads, fma.group_size, limits[name]),
                "flops_per_call": flops,
                **mapping,
                "median_gflops_per_second": flops / timing["median_ns"],
                "best_gflops_per_second": flops / timing["min_ns"],
                "output_sha256": _hash(words),
            }
        )
        _require(
            _exact(entry, rebuilt[-1]),
            f"{name} point at {steps} steps does not recompute",
        )
    return rebuilt


def _stream_points(
    section: object, config: GpuCeilingConfig, limits: dict
) -> list[dict[str, object]]:
    bandwidth = config.bandwidth
    counts = [size // 16 for size in bandwidth.buffer_bytes]
    plan = [(kernel, count) for kernel in (_COPY, _REDUCE) for count in counts]
    _require(
        type(section) is list and len(section) == len(plan),
        "stream points do not cover the sweep",
    )
    index = _STREAM_INDEX["benchmark"]
    sources = {
        _COPY: _copy_source(config.seed, index, max(counts)),
        _REDUCE: _reduce_source(config.seed, index, max(counts)),
    }
    rebuilt = []
    for entry, (kernel, count) in zip(section, plan):
        per_thread = (
            bandwidth.copy_float4_per_thread
            if kernel == _COPY
            else bandwidth.reduce_float4_per_thread
        )
        walkers = -(-count // per_thread)
        per_call = _stream_bytes(kernel, count, per_thread)
        calls = _calls(per_call, bandwidth.target_bytes_per_sample)
        timing, mapping = _samples(entry, config, calls, f"{kernel} at {count}")
        prefix = sources[kernel][: 4 * count]
        output = prefix if kernel == _COPY else _partials(prefix, count, walkers)
        moved = per_call * calls
        rebuilt.append(
            {
                "kernel": kernel,
                "buffer_bytes": 16 * count,
                "float4_count": count,
                "float4_per_thread": per_thread,
                "threads": walkers,
                "launch": _launch(walkers, bandwidth.group_size, limits[kernel]),
                "bytes_per_call": per_call,
                "bytes_per_sample": moved,
                **mapping,
                "median_gb_per_second": moved / timing["median_ns"],
                "best_gb_per_second": moved / timing["min_ns"],
                "output_sha256": _hash(output),
            }
        )
        _require(
            _exact(entry, rebuilt[-1]),
            f"{kernel} point over {count} float4 does not recompute",
        )
    return rebuilt


def _bank_points(
    section: object, config: GpuCeilingConfig, limits: dict
) -> list[dict[str, object]]:
    bank = config.bank
    plan = [(words, stride) for words in _ROWS for stride in bank.strides]
    _require(
        type(section) is list and len(section) == len(plan),
        "bank points do not cover the sweep",
    )
    rebuilt = []
    accesses = bank.threads * bank.trips * _LOADS
    for entry, (words, stride) in zip(section, plan):
        name = f"bank_rows_{words}"
        timing, mapping = _samples(entry, config, 1, f"{name} at stride {stride}")
        output = _spread(_bank_words(stride, words, bank.trips), bank.threads)
        rebuilt.append(
            {
                "kernel": name,
                "row_words": words,
                "stride": stride,
                "trips": bank.trips,
                "threads": bank.threads,
                "launch": _launch(bank.threads, bank.group_size, limits[name]),
                "accesses_per_call": accesses,
                **mapping,
                "median_ps_per_access": timing["median_ns"] * 1000 / accesses,
                "output_sha256": _hash(output),
            }
        )
        _require(
            _exact(entry, rebuilt[-1]),
            f"{name} point at stride {stride} does not recompute",
        )
    return rebuilt


def _call(point: dict[str, object]) -> Fraction:
    return Fraction(point["timing"]["median_ns"], point["calls_per_sample"])


def _line(series: list[dict[str, object]]) -> tuple[Fraction, Fraction]:
    """Least squares in centered form, exact: slope and intercept."""

    xs = [Fraction(point["flops_per_call"]) for point in series]
    ys = [_call(point) for point in series]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    return slope, mean_y - slope * mean_x


def _first_best(points: list[dict[str, object]], key: str) -> dict[str, object]:
    best = points[0]
    for point in points:
        if point[key] > best[key]:
            best = point
    return best


def _ceilings(
    fma: list[dict[str, object]],
    streams: list[dict[str, object]],
    banks: list[dict[str, object]],
) -> dict[str, object]:
    def series(chains: int) -> list[dict[str, object]]:
        return sorted(
            (point for point in fma if point["chains"] == chains),
            key=lambda point: point["fma_steps"],
        )

    scaling = []
    for chains in sorted({point["chains"] for point in fma}):
        points = series(chains)
        slope, fixed = _line(points)
        scaling.append(
            {
                "chains": chains,
                "points": [
                    {
                        "fma_steps": point["fma_steps"],
                        "flops_per_call": point["flops_per_call"],
                        "median_ns_per_call": point["median_ns_per_call"],
                        "work_ratio_to_previous": None
                        if index == 0
                        else point["flops_per_call"]
                        / points[index - 1]["flops_per_call"],
                        "time_ratio_to_previous": None
                        if index == 0
                        else float(_call(point) / _call(points[index - 1])),
                    }
                    for index, point in enumerate(points)
                ],
                "marginal_gflops_per_second": float(1 / slope) if slope > 0 else None,
                "fixed_ns_per_call": float(fixed),
                "fixed_share_at_largest": float(fixed / _call(points[-1])),
            }
        )
    best = _first_best(fma, "median_gflops_per_second")
    slope, fixed = _line(series(best["chains"]))
    share = fixed / _call(best)
    compute = {
        "gflops_per_second": best["median_gflops_per_second"],
        "best_sample_gflops_per_second": best["best_gflops_per_second"],
        "kernel": best["kernel"],
        "chains": best["chains"],
        "fma_steps": best["fma_steps"],
        "flops_per_call": best["flops_per_call"],
        "fixed_share_of_ceiling_call": float(share),
        "fixed_share_limit": _FIXED_LIMIT,
        "scales_with_work": slope > 0 and share <= Fraction(_FIXED_LIMIT),
    }
    largest = max(point["buffer_bytes"] for point in streams)
    finalists = [point for point in streams if point["buffer_bytes"] == largest]
    top = _first_best(finalists, "median_gb_per_second")
    rates = {point["kernel"]: point["median_gb_per_second"] for point in finalists}
    bandwidth = {
        "gb_per_second": top["median_gb_per_second"],
        "best_sample_gb_per_second": top["best_gb_per_second"],
        "kernel": top["kernel"],
        "buffer_bytes": largest,
        "copy_gb_per_second": rates[_COPY],
        "read_reduce_gb_per_second": rates[_REDUCE],
    }
    return {
        "definition": CEILING_DEFINITION,
        "fma": compute,
        "fma_scaling": scaling,
        "bandwidth": bandwidth,
        "ridge_point_flops_per_byte": compute["gflops_per_second"]
        / bandwidth["gb_per_second"],
        "bank": _banks(banks),
    }


def _banks(points: list[dict[str, object]]) -> dict[str, object]:
    half = len(points) // 2
    unpadded, padded = points[:half], points[half:]
    times_u = [_call(point) for point in unpadded]
    times_p = [_call(point) for point in padded]
    base = times_u[0]
    rows = [
        {
            "stride": low["stride"],
            "unpadded_ps_per_access": low["median_ps_per_access"],
            "padded_ps_per_access": high["median_ps_per_access"],
            "unpadded_relative_to_stride_1": float(time_u / base),
            "padded_relative_to_own_stride_1": float(time_p / times_p[0]),
            "padded_relative_to_unpadded_stride_1": float(time_p / base),
            "padding_speedup": float(time_u / time_p),
            "model_unpadded_conflict_degree": _model_degree(low["stride"], 32),
            "model_padded_conflict_degree": _model_degree(high["stride"], 33),
        }
        for low, high, time_u, time_p in zip(unpadded, padded, times_u, times_p)
    ]
    limit = Fraction(_SLOWDOWN_LIMIT)
    ratios_u = [time / base for time in times_u]
    ratios_p = [time / base for time in times_p]
    rises = all(ratios_u[index] > ratios_u[index - 1] for index in range(1, half))
    speedup = times_u[-1] / times_p[-1]
    clear = rises and ratios_u[-1] >= limit
    if max(ratios_u + ratios_p) < limit:
        finding = "no_clear_conflict_pattern"
    elif clear and max(ratios_p) < limit:
        finding = "stride_slowdown_removed_by_padding"
    elif clear and speedup >= limit:
        finding = "stride_slowdown_reduced_by_padding"
    else:
        finding = "no_single_pattern"
    return {
        "model": _BANK_MODEL,
        "slowdown_limit": _SLOWDOWN_LIMIT,
        "rows": rows,
        "unpadded_rises_at_every_stride": rises,
        "largest_unpadded_ratio": float(max(ratios_u)),
        "largest_padded_ratio": float(max(ratios_p)),
        "padded_within_limit": max(ratios_p) < limit,
        "padding_speedup_at_largest_stride": float(speedup),
        "finding": finding,
    }


def _verify(report: object, supplied: GpuCeilingConfig | None) -> dict[str, object]:
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
    load = _load(report["host_load"])
    limits = _shaders(report["shaders"])
    _require(
        _exact(report["planned_work"], _planned(config)),
        "planned work does not match the config",
    )
    starts = _start_significands(config.seed)
    tables = _fused_significands(
        starts, [config.correctness.fma_steps, *config.fma.fma_steps]
    )
    correctness = report["correctness"]
    _require(type(correctness) is dict, "correctness must be a JSON object")
    expected_correctness = {
        "reference": REFERENCE,
        "fma_rule": FMA_RULE,
        "fma_coefficients": [_MULTIPLIER * 2.0**-24, _ADDEND * 2.0**-35],
        "reduce_max_value": _REDUCE_MAX,
        "sentinel_bits": _SENTINEL,
        "stream_guard_float4": _STREAM_GUARD,
        "fma": _fma_cases(correctness.get("fma"), config, limits, tables),
        "stream": _stream_cases(correctness.get("stream"), config, limits),
        "bank": _bank_cases(correctness.get("bank"), config, limits),
    }
    _require(
        _exact(correctness, expected_correctness),
        "correctness section does not match the lab's references",
    )
    benchmark = report["benchmark"]
    _require(type(benchmark) is dict, "benchmark must be a JSON object")
    fma = _fma_points(benchmark.get("fma"), config, limits, tables)
    streams = _stream_points(benchmark.get("stream"), config, limits)
    banks = _bank_points(benchmark.get("bank"), config, limits)
    _require(
        _exact(
            benchmark,
            {
                "device": DEVICE,
                "synchronization": SYNCHRONIZATION,
                "fma": fma,
                "stream": streams,
                "bank": banks,
            },
        ),
        "benchmark section does not recompute",
    )
    ceilings = _ceilings(fma, streams, banks)
    _require(
        _exact(report["ceilings"], ceilings),
        "ceilings do not recompute from the samples",
    )
    points = [*fma, *streams, *banks]
    timed = sum(sum(point["samples_ns"]) for point in points)
    run = report["run"]
    _require(
        type(run) is dict
        and set(run) == {"max_runtime_ns", "elapsed_ns", "timed_ns"}
        and _exact(run["max_runtime_ns"], config.timing.max_runtime_seconds * _NS)
        and _exact(run["timed_ns"], timed)
        and type(run["elapsed_ns"]) is int
        and timed <= run["elapsed_ns"] <= run["max_runtime_ns"],
        "run time does not cover its samples within the budget",
    )
    compute, bandwidth, bank = ceilings["fma"], ceilings["bandwidth"], ceilings["bank"]
    summary = {
        "label": label,
        "correctness_cases": len(expected_correctness["fma"])
        + len(expected_correctness["stream"])
        + len(expected_correctness["bank"]),
        "fma_points": len(fma),
        "stream_points": len(streams),
        "bank_points": len(banks),
        "timed_samples": config.timing.repeats * len(points),
        "fma_ceiling_gflops_per_second": compute["gflops_per_second"],
        "fma_ceiling_chains": compute["chains"],
        "fma_ceiling_fma_steps": compute["fma_steps"],
        "fma_fixed_share_of_ceiling_call": compute["fixed_share_of_ceiling_call"],
        "fma_scales_with_work": compute["scales_with_work"],
        "bandwidth_ceiling_gb_per_second": bandwidth["gb_per_second"],
        "bandwidth_ceiling_kernel": bandwidth["kernel"],
        "bandwidth_ceiling_buffer_bytes": bandwidth["buffer_bytes"],
        "ridge_point_flops_per_byte": ceilings["ridge_point_flops_per_byte"],
        "bank_finding": bank["finding"],
        "bank_largest_unpadded_ratio": bank["largest_unpadded_ratio"],
        "bank_largest_padded_ratio": bank["largest_padded_ratio"],
        "load_average_before": load["before"],
        "load_average_after": load["after"],
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
        "fma_points": summary["fma_points"],
        "stream_points": summary["stream_points"],
        "bank_points": summary["bank_points"],
        "timed_samples": summary["timed_samples"],
        "fma_ceiling_gflops_per_second": summary["fma_ceiling_gflops_per_second"],
        "fma_scales_with_work": summary["fma_scales_with_work"],
        "bandwidth_ceiling_gb_per_second": summary["bandwidth_ceiling_gb_per_second"],
        "ridge_point_flops_per_byte": summary["ridge_point_flops_per_byte"],
        "bank_finding": summary["bank_finding"],
        "load_average_before": summary["load_average_before"],
        "load_average_after": summary["load_average_after"],
    }


def verify_report(
    report: object, *, config: GpuCeilingConfig | None = None
) -> dict[str, object]:
    """Verify a sealed report by recomputation, optionally binding it to a config.

    Every failure, including malformed structure deep inside a section, is
    reported as ``GpuCeilingVerificationError``.
    """

    if config is not None and not isinstance(config, GpuCeilingConfig):
        raise TypeError("config must be a GpuCeilingConfig")
    try:
        return _verify(report, config)
    except GpuCeilingVerificationError:
        raise
    except (
        GpuCeilingError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as error:
        raise GpuCeilingVerificationError(
            f"report failed verification: {error}"
        ) from error
