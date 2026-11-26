"""Strict, bounded controls for one Day 82 correctness and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases, the square timing sizes with the Day 74 controls and the smallest
ratio of median times that calls one side faster. A config made for other
hypotheses names another plan digest and is refused.

A correctness case is ``{"m", "n", "k"}``, and every case runs the four
kernels. A timing size ``s`` runs all seven sides on ``s x s`` operands.
Before anything runs, the config is checked as a whole:

- The cases hold a 1 x 1 x 1 product, a K from 2 to 15, below one step of
  16, a K above 16 that is not a multiple of 16, a K of 1000 or more, a case whose m and
  n pass 32 and are not multiples of 8, so fragments cross the edges, and a
  case whose m and n are multiples of 32, so every store is direct.
- The repeats are a multiple of the seven sides, so each side takes each
  position of the rotation equally often.
- The cases, the calls and the operations fit their bounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_benchmark import MAX_RUNTIME_SECONDS, MIN_REPEATS
from .kernel_metal_reference import MAX_SEED
from .matmul_tiled_config import ComparisonControls
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape
from .metal_launch_sweep_benchmark import (
    MAX_TARGET_FLOPS,
    MIN_TARGET_FLOPS,
    matmul_calls_per_sample,
    matmul_flops,
)
from .metal_launch_sweep_shapes import MetalLaunchSweepError
from .metal_tc_plan import SIDES, plan_sha256
from .metal_tc_source import (
    FRAGMENT,
    MAX_DIMENSION,
    STEP,
    TILE,
    VARIANTS,
    MetalTcError,
    require_size,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 12
MAX_SIZES = 6
MAX_CASE_DIMENSION = 1024
MAX_CORRECTNESS_PRODUCTS = 1 << 28
MAX_RUN_CALLS = 1 << 16
MAX_RUN_FLOPS = 1 << 48
MIN_CHECK_REPEATS = 2
MAX_CHECK_REPEATS = 4
MIN_CHECK_ROWS = 2
MAX_CHECK_ROWS = 256
LONG_K = 1000
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
_BENCHMARK_FIELDS = {
    "sizes",
    "check_rows",
    "warmup",
    "repeats",
    "target_flops_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MetalTcError(f"{label} fields do not match the schema")
    return payload


def calls_per_sample(size: int, target_flops: int) -> int:
    """Return the fewest whole calls of one size whose operations reach the target."""

    try:
        return matmul_calls_per_sample(matmul_flops(size, size, size), target_flops)
    except MetalLaunchSweepError as error:
        raise MetalTcError(str(error)) from error


@dataclass(frozen=True)
class TcCase:
    """One correctness product ``C[m, n] = A[m, k] @ B[k, n]``."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            require_size(getattr(self, name), name, 1, MAX_CASE_DIMENSION)

    def shape(self) -> MatmulShape:
        """The shape of the case."""

        return MatmulShape(self.m, self.n, self.k)

    def to_mapping(self) -> dict[str, int]:
        """Return the case as the config file spells it."""

        return {"m": self.m, "n": self.n, "k": self.k}


@dataclass(frozen=True)
class TcPoint:
    """One timing point: square operands of one size."""

    index: int
    size: int

    def shape(self) -> MatmulShape:
        """The square shape of the point."""

        return MatmulShape(self.size, self.size, self.size)

    def to_mapping(self) -> dict[str, int]:
        """Return the point as a report names it."""

        return {"index": self.index, "size": self.size}


def _require_coverage(cases: tuple[TcCase, ...]) -> None:
    checks = [
        (any((c.m, c.n, c.k) == (1, 1, 1) for c in cases), "a 1 x 1 x 1 product"),
        (any(1 < c.k < STEP for c in cases), f"a K from 2 to {STEP - 1}"),
        (
            any(c.k > STEP and c.k % STEP for c in cases),
            f"a K above {STEP} that is not a multiple of {STEP}",
        ),
        (any(c.k >= LONG_K for c in cases), f"a K of {LONG_K} or more"),
        (
            any(
                c.m > TILE and c.n > TILE and c.m % FRAGMENT and c.n % FRAGMENT
                for c in cases
            ),
            f"m and n above {TILE} that are not multiples of {FRAGMENT}",
        ),
        (
            any(c.m % TILE == 0 and c.n % TILE == 0 for c in cases),
            f"m and n that are multiples of {TILE}",
        ),
    ]
    for covered, noun in checks:
        if not covered:
            raise MetalTcError(f"correctness cases need {noun}")


@dataclass(frozen=True)
class CorrectnessControls:
    """The cases and how many runs measure determinism."""

    cases: tuple[TcCase, ...]
    repeats: int

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES:
            raise MetalTcError(f"correctness holds 1 to {MAX_CASES} cases")
        if not all(isinstance(case, TcCase) for case in self.cases):
            raise TypeError("cases must be TcCase values")
        if len(set(self.cases)) != len(self.cases):
            raise MetalTcError("each correctness case appears once")
        _require_coverage(self.cases)
        require_size(self.repeats, "repeats", MIN_CHECK_REPEATS, MAX_CHECK_REPEATS)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "cases": [case.to_mapping() for case in self.cases],
            "repeats": self.repeats,
        }


@dataclass(frozen=True)
class TimingControls:
    """The sizes, checked rows, warmups, repeats, batch and budget."""

    sizes: tuple[int, ...]
    check_rows: int
    warmup: int
    repeats: int
    target_flops_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        if type(self.sizes) is not tuple or not 1 <= len(self.sizes) <= MAX_SIZES:
            raise MetalTcError(f"the benchmark holds 1 to {MAX_SIZES} sizes")
        for size in self.sizes:
            require_size(size, "size", 1, MAX_DIMENSION)
        if list(self.sizes) != sorted(set(self.sizes)):
            raise MetalTcError("sizes must be distinct and in increasing order")
        require_size(self.check_rows, "check_rows", MIN_CHECK_ROWS, MAX_CHECK_ROWS)
        require_size(self.warmup, "warmup", 0, MAX_WARMUP)
        require_size(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        if self.repeats % len(SIDES):
            raise MetalTcError(
                f"repeats must be a multiple of the {len(SIDES)} timed sides, so "
                "each side takes each position of the rotation equally often"
            )
        require_size(
            self.target_flops_per_sample,
            "target_flops_per_sample",
            MIN_TARGET_FLOPS,
            MAX_TARGET_FLOPS,
        )
        require_size(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )
        for size in self.sizes:
            calls_per_sample(size, self.target_flops_per_sample)

    def points(self) -> tuple[TcPoint, ...]:
        """Every timing point, in size order."""

        return tuple(TcPoint(index, size) for index, size in enumerate(self.sizes))

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "sizes": list(self.sizes),
            "check_rows": self.check_rows,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_flops_per_sample": self.target_flops_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class MetalTcConfig:
    """Every control one Day 82 run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: TimingControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalTcError("config name must be a lowercase slug")
        require_size(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise MetalTcError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise MetalTcError(
                "plan_sha256 does not name the declared plan; a config for other "
                "hypotheses cannot run this plan"
            )
        for value, kind in (
            (self.correctness, CorrectnessControls),
            (self.benchmark, TimingControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work["correctness_products"], MAX_CORRECTNESS_PRODUCTS, "products"),
            (work["calls"], MAX_RUN_CALLS, "side calls"),
            (work["flops"], MAX_RUN_FLOPS, "operations"),
        ):
            if value > limit:
                raise MetalTcError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> dict[str, int]:
        """Count the cases, samples, calls and operations of the run up front."""

        timing = self.benchmark
        samples = timing.warmup + timing.repeats
        calls = flops = 0
        points = timing.points()
        for point in points:
            batch = calls_per_sample(point.size, timing.target_flops_per_sample)
            calls += len(SIDES) * samples * batch
            work = matmul_flops(point.size, point.size, point.size)
            flops += len(SIDES) * samples * batch * work
        cases = self.correctness.cases
        return {
            "correctness_cases": len(cases),
            "correctness_products": sum(case.m * case.n * case.k for case in cases),
            "correctness_launches": len(cases)
            * len(VARIANTS)
            * self.correctness.repeats,
            "benchmark_points": len(points),
            "sides_per_point": len(SIDES),
            "timed_samples": len(SIDES) * len(points) * timing.repeats,
            "calls": calls,
            "flops": flops,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping that a report embeds and its digest covers."""

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


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise MetalTcError(f"{label} must be a JSON array")
    return value


def _case(payload: object) -> TcCase:
    case = _fields(payload, {"m", "n", "k"}, "correctness case")
    return TcCase(case["m"], case["n"], case["k"])


def _config(payload: object) -> MetalTcConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MetalTcError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], {"cases", "repeats"}, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], {"min_faster_ratio"}, "comparison")
    return MetalTcConfig(
        name=config["name"],
        seed=config["seed"],
        plan_sha256=config["plan_sha256"],
        correctness=CorrectnessControls(
            cases=tuple(_case(item) for item in _list(correctness["cases"], "cases")),
            repeats=correctness["repeats"],
        ),
        benchmark=TimingControls(
            sizes=tuple(_list(benchmark["sizes"], "sizes")),
            check_rows=benchmark["check_rows"],
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_flops_per_sample=benchmark["target_flops_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MetalTcConfig:
    """Parse controls, reporting every contract violation as ``MetalTcError``.

    The Day 73 and Day 76 controls raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except MetalTcError:
        raise
    except (TypeError, ValueError) as error:
        raise MetalTcError(f"invalid metal-tc config: {error}") from error


def load_config(path: Path) -> MetalTcConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MetalTcError(str(error)) from error
    return config_from_mapping(payload)
