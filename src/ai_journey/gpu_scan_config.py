"""Strict, bounded controls for one Day 78 scan correctness and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases with the runs of each side, the timing sweep with the Day 73 byte
target, and the smallest ratio of median rates that calls one side faster.
A run times every kernel of the plan, so the config pins the plan digest,
and a config made for other hypotheses or other kernel texts is refused.

Before anything runs, the config is checked as a whole:

- Some case scans one partial block whose end falls inside a SIMD group: a
  length below 1024 that is not a multiple of 32.
- Some case has two or more levels and a partial last block, and some case
  has three levels and a partial block at level 2, so a ragged block runs
  at every level that a pinned length uses.
- Each pattern has a case, and some integer case has two or more levels, so
  the add kernel meets the exact check too.
- Each output with its guard tail fits one sentinel buffer, the cases fit
  the CPU element bound, and the calls and bytes fit the work bounds.

Files go through the bounded strict reader of ``memory_access_report_io``,
which also rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .gpu_scan_correctness import MAX_RUNS, MIN_RUNS
from .gpu_scan_plan import SIDES, plan_sha256
from .gpu_scan_reference import INTEGERS, MAX_EXACT_LENGTH, PATTERNS
from .gpu_scan_source import (
    BLOCK,
    KERNELS,
    SIMD_WIDTH,
    GpuScanError,
    level_lengths,
    require_count,
    require_length,
)
from .kernel_metal_benchmark import (
    MAX_POINTS,
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)
from .kernel_metal_reference import MAX_SEED
from .matmul_tiled_config import ComparisonControls
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 32
# The CPU replays and checks every case; this bounds its elements.
MAX_CASE_ELEMENTS = 1 << 25
MAX_RUN_CALLS = 1 << 22
MAX_RUN_BYTES = 1 << 40
# One float32 read and one float32 write of each element: what any scan moves.
BYTES_PER_ELEMENT = 8
SIDES_PER_POINT = len(SIDES)

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "plan_sha256",
    "correctness",
    "benchmark",
    "comparison",
}
_CORRECTNESS_FIELDS = {"cases", "runs"}
_BENCHMARK_FIELDS = {
    "lengths",
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}
_COMPARISON_FIELDS = {"min_faster_ratio"}


def scan_calls_per_sample(length: int, target_bytes: int) -> int:
    """The fewest whole scans whose compulsory bytes reach the sample target."""

    try:
        return calls_per_sample(
            BYTES_PER_ELEMENT * require_length(length), target_bytes
        )
    except GpuScanError:
        raise
    except ValueError as error:
        raise GpuScanError(str(error)) from error


@dataclass(frozen=True)
class CorrectnessControls:
    """The ``(length, pattern)`` cases and how many times each side runs."""

    cases: tuple[tuple[int, str], ...]
    runs: int

    def __post_init__(self) -> None:
        cases = self.cases
        if type(cases) is not tuple or not 1 <= len(cases) <= MAX_CASES:
            raise GpuScanError(f"correctness holds 1 to {MAX_CASES} cases")
        for case in cases:
            if type(case) is not tuple or len(case) != 2:
                raise GpuScanError("a case is a [length, pattern] pair")
            length, pattern = case
            require_length(length, "case length")
            if pattern not in PATTERNS:
                raise GpuScanError(f"a pattern must be one of {', '.join(PATTERNS)}")
            if pattern == INTEGERS and length > MAX_EXACT_LENGTH:
                raise GpuScanError(
                    f"an integer case holds at most {MAX_EXACT_LENGTH} elements"
                )
        if len(set(cases)) != len(cases):
            raise GpuScanError("correctness cases must not repeat")
        require_count(self.runs, "runs", MIN_RUNS, MAX_RUNS)
        total = sum(length for length, _ in cases)
        if total > MAX_CASE_ELEMENTS:
            raise GpuScanError(
                f"the cases hold {total} elements; the bound is {MAX_CASE_ELEMENTS}"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {"cases": [list(case) for case in self.cases], "runs": self.runs}


@dataclass(frozen=True)
class BenchmarkControls:
    """The sweep of lengths and the warmups, repeats, byte target and budget."""

    lengths: tuple[int, ...]
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        lengths = self.lengths
        if type(lengths) is not tuple or not 1 <= len(lengths) <= MAX_POINTS:
            raise GpuScanError(f"benchmark lengths must hold 1 to {MAX_POINTS} lengths")
        for length in lengths:
            require_length(length, "benchmark length")
        if len(set(lengths)) != len(lengths):
            raise GpuScanError("benchmark lengths must not repeat")
        require_count(self.warmup, "warmup", 0, MAX_WARMUP)
        require_count(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_count(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )
        require_count(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )
        for length in lengths:
            scan_calls_per_sample(length, self.target_bytes_per_sample)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "lengths": list(self.lengths),
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host at most, counted before anything runs."""

    correctness_cases: int
    case_elements: int
    correctness_runs: int
    benchmark_points: int
    sides_per_point: int
    timed_samples: int
    calls: int
    bytes: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "case_elements": self.case_elements,
            "correctness_runs": self.correctness_runs,
            "benchmark_points": self.benchmark_points,
            "sides_per_point": self.sides_per_point,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "bytes": self.bytes,
        }


def _require_coverage(cases: tuple[tuple[int, str], ...]) -> None:
    """Require a ragged block at every level and both patterns in the cases."""

    lengths = [length for length, _ in cases]
    levels = {length: level_lengths(length) for length in lengths}
    rules = (
        (
            any(length < BLOCK and length % SIMD_WIDTH for length in lengths),
            f"a length below {BLOCK} that is not a multiple of {SIMD_WIDTH}",
        ),
        (
            any(len(levels[n]) >= 2 and n % BLOCK for n in lengths),
            f"a length with two or more levels that is not a multiple of {BLOCK}",
        ),
        (
            any(len(levels[n]) >= 3 and levels[n][1] % BLOCK for n in lengths),
            "a length with three levels and a partial block at level 2",
        ),
        (
            {pattern for _, pattern in cases} == set(PATTERNS),
            "a case of each pattern",
        ),
        (
            any(
                pattern == INTEGERS and len(levels[length]) >= 2
                for length, pattern in cases
            ),
            "an integer case with two or more levels",
        ),
    )
    for held, need in rules:
        if not held:
            raise GpuScanError(f"correctness cases need {need}")


@dataclass(frozen=True)
class GpuScanConfig:
    """Every control one scan run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: BenchmarkControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuScanError("config name must be a lowercase slug")
        require_count(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise GpuScanError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise GpuScanError(
                "plan_sha256 does not name the declared plan; a config for other "
                "hypotheses or kernel texts cannot run this plan"
            )
        for value, kind in (
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        _require_coverage(self.correctness.cases)
        work = self.planned_work()
        for value, limit, label in (
            (work.calls, MAX_RUN_CALLS, "calls"),
            (work.bytes, MAX_RUN_BYTES, "bytes"),
        ):
            if value > limit:
                raise GpuScanError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the cases, runs, samples, calls and bytes of the run up front.

        Every kernel counts, so the counts are an upper bound when a pipeline
        refuses a kernel. The bytes are the compulsory 8 bytes of each element
        for each call of each side.
        """

        timing = self.benchmark
        samples = SIDES_PER_POINT * (timing.warmup + timing.repeats)
        calls = moved = 0
        for length in timing.lengths:
            batch = scan_calls_per_sample(length, timing.target_bytes_per_sample)
            calls += samples * batch
            moved += samples * batch * BYTES_PER_ELEMENT * length
        cases = self.correctness.cases
        return PlannedWork(
            correctness_cases=len(cases),
            case_elements=sum(length for length, _ in cases),
            correctness_runs=len(cases) * (len(KERNELS) + 1) * self.correctness.runs,
            benchmark_points=len(timing.lengths),
            sides_per_point=SIDES_PER_POINT,
            timed_samples=SIDES_PER_POINT * len(timing.lengths) * timing.repeats,
            calls=calls,
            bytes=moved,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "plan_sha256": self.plan_sha256,
            "correctness": self.correctness.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
            "comparison": self.comparison.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise GpuScanError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise GpuScanError(f"{label} must be a JSON array")
    return tuple(value)


def _cases(value: object) -> tuple[tuple[int, str], ...]:
    cases = []
    for item in _list(value, "correctness cases"):
        pair = _list(item, "a correctness case")
        if len(pair) != 2:
            raise GpuScanError("a case is a [length, pattern] pair")
        cases.append(pair)
    return tuple(cases)


def _config(payload: object) -> GpuScanConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise GpuScanError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], _COMPARISON_FIELDS, "comparison")
    return GpuScanConfig(
        name=config["name"],
        seed=config["seed"],
        plan_sha256=config["plan_sha256"],
        correctness=CorrectnessControls(
            cases=_cases(correctness["cases"]), runs=correctness["runs"]
        ),
        benchmark=BenchmarkControls(
            lengths=_list(benchmark["lengths"], "benchmark lengths"),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> GpuScanConfig:
    """Parse controls, reporting every contract violation as ``GpuScanError``.

    The Day 73 and Day 76 controls raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except GpuScanError:
        raise
    except (TypeError, ValueError) as error:
        raise GpuScanError(f"invalid gpu-scan config: {error}") from error


def load_config(path: Path) -> GpuScanConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise GpuScanError(str(error)) from error
    return config_from_mapping(payload)
