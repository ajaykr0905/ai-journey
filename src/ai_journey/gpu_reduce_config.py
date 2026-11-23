"""Strict, bounded controls for one reduction correctness and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases with the number of runs of each kernel, the timing sizes with the
Day 74 controls, and the smallest ratio of median rates that calls one side
faster. A run times every kernel of the plan, so the config pins the plan's
digest, and a config made for other hypotheses or kernels is refused.

Before anything runs, the cases are checked as a whole, so every path of
every kernel runs in some case:

- every input pattern, the exact ``ones`` and ``halves`` among them;
- an n below 32, so one SIMD group holds the whole input and a thread has
  its first load in range and its second load out of range;
- an n whose last 512-element group ends in its second half, so some
  threads of that group have both loads in range and others only one;
- an n with more than 3072 partials for every kernel but R5, so thread 0
  of the combine runs its unrolled four-accumulator loop. That n is above
  1.5 million, so the R5 loop also runs more than once for each thread.

The timing sizes are 1 to 16 distinct counts up to 2^26, and each sample
holds the fewest whole calls whose bytes reach the target, at most 4096.
Files go through the bounded strict reader of ``memory_access_report_io``,
which also rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .gpu_reduce_bound import MAX_CASE_INDEX, PATTERNS, partial_count
from .gpu_reduce_plan import MAX_ELEMENTS, plan_sha256
from .gpu_reduce_source import (
    COMBINE_GROUP_SIZE,
    GROUP_SIZE,
    LOADS_PER_THREAD,
    R5,
    SIMD_WIDTH,
    STEP_KERNELS,
    GpuReduceError,
    require_count,
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
MAX_CASES = MAX_CASE_INDEX + 1
MAX_CASE_ELEMENTS = 1 << 22
MAX_CASE_TOTAL = 1 << 24
MIN_RUNS = 2
MAX_RUNS = 5
BYTES_PER_ELEMENT = 4
# The six step kernels and torch.sum run at every timing size.
SIDES_PER_POINT = len(STEP_KERNELS) + 1
MAX_RUN_CALLS = 1 << 22
MAX_RUN_BYTES = 1 << 40
EXACT_PATTERNS = ("ones", "halves")
GROUP_SPAN = LOADS_PER_THREAD * GROUP_SIZE
UNROLLED_PARTIALS = 3 * COMBINE_GROUP_SIZE

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
    "sizes",
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}
_COMPARISON_FIELDS = {"min_faster_ratio"}


@dataclass(frozen=True)
class ReduceCase:
    """One correctness case: an input pattern and its length."""

    pattern: str
    n: int

    def __post_init__(self) -> None:
        if self.pattern not in PATTERNS:
            raise GpuReduceError(f"pattern must be one of {', '.join(PATTERNS)}")
        require_count(self.n, "case n", 1, MAX_CASE_ELEMENTS)

    def to_mapping(self) -> list[object]:
        """Return the case as the config file spells it."""

        return [self.pattern, self.n]


def _require_paths(cases: tuple[ReduceCase, ...]) -> None:
    """Require every pattern and every path of every kernel in some case."""

    patterns = {case.pattern for case in cases}
    missing = [pattern for pattern in PATTERNS if pattern not in patterns]
    if missing:
        raise GpuReduceError(f"correctness cases need the pattern {missing[0]}")
    sizes = [case.n for case in cases]
    rules = (
        (
            any(n < SIMD_WIDTH for n in sizes),
            f"an n below {SIMD_WIDTH}, so one SIMD group holds the whole input",
        ),
        (
            any(n % GROUP_SPAN > GROUP_SIZE for n in sizes),
            "an n whose last 512-element group ends in its second half",
        ),
        (
            any(
                all(
                    partial_count(name, n) > UNROLLED_PARTIALS
                    for name in STEP_KERNELS
                    if name != R5
                )
                for n in sizes
            ),
            f"an n with more than {UNROLLED_PARTIALS} partials for every kernel "
            "but R5, so the combine runs its unrolled loop",
        ),
    )
    for holds, text in rules:
        if not holds:
            raise GpuReduceError(f"correctness cases need {text}")


@dataclass(frozen=True)
class CorrectnessControls:
    """The seeded cases and how many times each kernel runs on each case."""

    cases: tuple[ReduceCase, ...]
    runs: int

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES:
            raise GpuReduceError(f"correctness needs 1 to {MAX_CASES} cases")
        if not all(isinstance(case, ReduceCase) for case in self.cases):
            raise TypeError("correctness cases must be ReduceCase values")
        if len(set(self.cases)) != len(self.cases):
            raise GpuReduceError("each correctness case may appear only once")
        total = sum(case.n for case in self.cases)
        if total > MAX_CASE_TOTAL:
            raise GpuReduceError(
                f"correctness cases hold {total} elements; the bound is "
                f"{MAX_CASE_TOTAL}"
            )
        require_count(self.runs, "runs", MIN_RUNS, MAX_RUNS)
        _require_paths(self.cases)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {"cases": [case.to_mapping() for case in self.cases], "runs": self.runs}


@dataclass(frozen=True)
class BenchmarkControls:
    """The timing sizes and the Day 74 controls of one rotated run."""

    sizes: tuple[int, ...]
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        if type(self.sizes) is not tuple or not 1 <= len(self.sizes) <= MAX_POINTS:
            raise GpuReduceError(f"a sweep holds 1 to {MAX_POINTS} sizes")
        for n in self.sizes:
            require_count(n, "benchmark n", 1, MAX_ELEMENTS)
        if len(set(self.sizes)) != len(self.sizes):
            raise GpuReduceError("each benchmark size may appear only once")
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
        for n in self.sizes:
            self.calls_per_sample(n)

    def calls_per_sample(self, n: int) -> int:
        """The fewest whole calls whose bytes, read once, reach the target."""

        try:
            return calls_per_sample(BYTES_PER_ELEMENT * n, self.target_bytes_per_sample)
        except GpuReduceError:
            raise
        except ValueError as error:
            raise GpuReduceError(f"benchmark n {n}: {error}") from error

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "sizes": list(self.sizes),
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host at most, counted before anything runs."""

    correctness_cases: int
    correctness_elements: int
    correctness_launches: int
    benchmark_points: int
    sides_per_point: int
    timed_samples: int
    calls: int
    bytes_read: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "correctness_elements": self.correctness_elements,
            "correctness_launches": self.correctness_launches,
            "benchmark_points": self.benchmark_points,
            "sides_per_point": self.sides_per_point,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "bytes_read": self.bytes_read,
        }


@dataclass(frozen=True)
class GpuReduceConfig:
    """Every control one reduction run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: BenchmarkControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuReduceError("config name must be a lowercase slug")
        require_count(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise GpuReduceError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise GpuReduceError(
                "plan_sha256 does not name the declared plan; a config for other "
                "hypotheses or kernels cannot run this plan"
            )
        for value, kind in (
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work.calls, MAX_RUN_CALLS, "calls"),
            (work.bytes_read, MAX_RUN_BYTES, "bytes"),
        ):
            if value > limit:
                raise GpuReduceError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the cases, launches, samples, calls and bytes of the run up front.

        A step-kernel call is two launches: pass 1 and the combine.
        """

        timing = self.benchmark
        samples = SIDES_PER_POINT * (timing.warmup + timing.repeats)
        calls = moved = 0
        for n in timing.sizes:
            batch = timing.calls_per_sample(n)
            calls += samples * batch
            moved += samples * batch * BYTES_PER_ELEMENT * n
        cases = self.correctness.cases
        return PlannedWork(
            correctness_cases=len(cases),
            correctness_elements=sum(case.n for case in cases),
            correctness_launches=len(cases)
            * len(STEP_KERNELS)
            * self.correctness.runs
            * 2,
            benchmark_points=len(timing.sizes),
            sides_per_point=SIDES_PER_POINT,
            timed_samples=SIDES_PER_POINT * len(timing.sizes) * timing.repeats,
            calls=calls,
            bytes_read=moved,
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
        raise GpuReduceError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise GpuReduceError(f"{label} must be a JSON array")
    return tuple(value)


def _cases(value: object) -> tuple[ReduceCase, ...]:
    cases = []
    for item in _list(value, "correctness cases"):
        pair = _list(item, "correctness case")
        if len(pair) != 2:
            raise GpuReduceError("a correctness case is a [pattern, n] pair")
        cases.append(ReduceCase(*pair))
    return tuple(cases)


def _config(payload: object) -> GpuReduceConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise GpuReduceError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], _COMPARISON_FIELDS, "comparison")
    return GpuReduceConfig(
        name=config["name"],
        seed=config["seed"],
        plan_sha256=config["plan_sha256"],
        correctness=CorrectnessControls(
            cases=_cases(correctness["cases"]), runs=correctness["runs"]
        ),
        benchmark=BenchmarkControls(
            sizes=_list(benchmark["sizes"], "benchmark sizes"),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> GpuReduceConfig:
    """Parse controls, reporting every contract violation as ``GpuReduceError``.

    The Day 76 comparison section raises its own lab's error; it is
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except GpuReduceError:
        raise
    except (TypeError, ValueError) as error:
        raise GpuReduceError(f"invalid gpu-reduce config: {error}") from error


def load_config(path: Path) -> GpuReduceConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise GpuReduceError(str(error)) from error
    return config_from_mapping(payload)
