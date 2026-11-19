"""Strict, bounded controls for one tiled Metal matmul correctness and benchmark run.

A config names the seed, the threadgroup of the naive Day 74 kernel, the
registered tiled variants to compile, the correctness shapes, the benchmark
sweep with the Day 74 timing controls, and the smallest ratio of median
rates that calls one side faster. Parsing rejects unknown and missing fields
at every level, booleans posing as integers, and out-of-range values. The
config is then checked as a whole before anything runs:

- For every tile T of the run, some correctness shape has an M, some an N
  and some a K that is not a multiple of T, and some K above T is not a
  multiple of T. Every variant thus pads both axes, and a phase after the
  first ends early, in some case.
- The correctness shapes pad both axes of the naive group, as on Day 74.
- Each correctness output with its guard tail fits one sentinel buffer, and
  the CPU replay of all cases fits its step bound.
- Every sweep point fits the batch limit, and the run's total calls and
  FLOPs fit the Day 74 work bounds.

The planned work counts every named variant, so it is an upper bound: a
variant that its pipeline refuses never launches. Files go through the
bounded strict reader of ``memory_access_report_io``, which also rejects
duplicate keys.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import MAX_SEED, MAX_VECTOR_BUFFER
from .matmul_metal_benchmark import calls_per_sample
from .matmul_metal_config import (
    MAX_REPLAY_STEPS,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    BenchmarkControls,
    CorrectnessControls,
    LaunchControls,
)
from .matmul_metal_mapping import X_COLUMNS, matmul_launch, matmul_shape
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import TiledVariant, variants_named
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_FASTER_RATIO = 4.0
# The naive kernel and torch.matmul run beside the variants at every point.
FIXED_SIDES = 2

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "naive",
    "variants",
    "correctness",
    "benchmark",
    "comparison",
}
_NAIVE_FIELDS = {"group_size"}
_CORRECTNESS_FIELDS = {"shapes"}
_BENCHMARK_FIELDS = {
    "shapes",
    "warmup",
    "repeats",
    "target_flops_per_sample",
    "max_runtime_seconds",
}
_COMPARISON_FIELDS = {"min_faster_ratio"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MatmulTiledError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MatmulTiledError(f"{label} must be a JSON array")
    return tuple(value)


def _shapes(value: object, label: str) -> tuple[MatmulShape, ...]:
    shapes = []
    for item in _list(value, label):
        triple = _list(item, f"{label} item")
        if len(triple) != 3:
            raise MatmulTiledError(f"{label} must be [m, n, k] triples")
        shapes.append(matmul_shape(*triple))
    return tuple(shapes)


@dataclass(frozen=True)
class ComparisonControls:
    """The smallest ratio of median rates that calls one side faster."""

    min_faster_ratio: float

    def __post_init__(self) -> None:
        ratio = self.min_faster_ratio
        if type(ratio) is not float or not math.isfinite(ratio):
            raise TypeError("min_faster_ratio must be a finite float")
        if not 1.0 < ratio <= MAX_FASTER_RATIO:
            raise MatmulTiledError(
                f"min_faster_ratio must be above 1 and at most {MAX_FASTER_RATIO}"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {"min_faster_ratio": self.min_faster_ratio}


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host at most, counted before anything runs."""

    correctness_cases: int
    replay_steps: int
    correctness_launches: int
    benchmark_points: int
    sides_per_point: int
    timed_samples: int
    calls: int
    flops: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "replay_steps": self.replay_steps,
            "correctness_launches": self.correctness_launches,
            "benchmark_points": self.benchmark_points,
            "sides_per_point": self.sides_per_point,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "flops": self.flops,
        }


def _require_padding(shapes: tuple[MatmulShape, ...], tile: int, label: str) -> None:
    """Require each axis of one tile to pad, and a short phase after the first."""

    for axis, noun in (("m", "an M"), ("n", "an N"), ("k", "a K")):
        if all(getattr(shape, axis) % tile == 0 for shape in shapes):
            raise MatmulTiledError(
                f"correctness shapes need {noun} that is not a multiple of "
                f"{label} {tile}"
            )
    if not any(shape.k > tile and shape.k % tile for shape in shapes):
        raise MatmulTiledError(
            f"correctness shapes need a K above {label} {tile} that is not a "
            "multiple of it"
        )


@dataclass(frozen=True)
class MatmulTiledConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    naive: LaunchControls
    variants: tuple[str, ...]
    correctness: CorrectnessControls
    benchmark: BenchmarkControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MatmulTiledError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.naive, LaunchControls),
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        if type(self.variants) is not tuple:
            raise TypeError("variants must be a tuple of names")
        shapes = self.correctness.shapes
        for variant in self.tiled_variants():
            _require_padding(shapes, variant.tile, "tile")
        group_x, group_y = self.naive.group_size
        for axis, group, name in (("n", group_x, "x"), ("m", group_y, "y")):
            if all(getattr(shape, axis) % group == 0 for shape in shapes):
                raise MatmulTiledError(
                    f"correctness shapes need an {axis.upper()} that is not a "
                    f"multiple of the naive group {name} size {group}"
                )
        for shape in shapes:
            for group in (
                self.naive.group_size,
                *(variant.group_size for variant in self.tiled_variants()),
            ):
                launch = matmul_launch(X_COLUMNS, shape, group)
                if launch.launched_threads > MAX_VECTOR_BUFFER:
                    raise MatmulTiledError(
                        f"correctness shape {[shape.m, shape.n, shape.k]} needs "
                        f"{launch.launched_threads} output elements with its guard "
                        f"tail; the bound is {MAX_VECTOR_BUFFER}"
                    )
        work = self.planned_work()
        for value, limit, label in (
            (work.replay_steps, MAX_REPLAY_STEPS, "replay steps"),
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.flops, MAX_RUN_FLOPS, "FLOPs"),
        ):
            if value > limit:
                raise MatmulTiledError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def tiled_variants(self) -> tuple[TiledVariant, ...]:
        """Return the registered variants this config names, in its order."""

        return variants_named(self.variants)

    def planned_work(self) -> PlannedWork:
        """Count the cases, launches, samples, calls and FLOPs of the run up front.

        The batch sizes are the ones the run will use, so a point that would
        need an oversized sample fails here, before anything is launched.
        """

        timing = self.benchmark
        sides = FIXED_SIDES + len(self.variants)
        samples = sides * (timing.warmup + timing.repeats)
        calls = flops = 0
        for shape in timing.shapes:
            batch = calls_per_sample(shape.flops, timing.target_flops_per_sample)
            calls += samples * batch
            flops += samples * batch * shape.flops
        cases = self.correctness.shapes
        return PlannedWork(
            correctness_cases=len(cases),
            replay_steps=sum(shape.m * shape.n * shape.k for shape in cases),
            correctness_launches=len(cases) * (1 + len(self.variants)),
            benchmark_points=len(timing.shapes),
            sides_per_point=sides,
            timed_samples=sides * len(timing.shapes) * timing.repeats,
            calls=calls,
            flops=flops,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "naive": self.naive.to_mapping(),
            "variants": list(self.variants),
            "correctness": self.correctness.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
            "comparison": self.comparison.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> MatmulTiledConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MatmulTiledError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    naive = _fields(config["naive"], _NAIVE_FIELDS, "naive")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], _COMPARISON_FIELDS, "comparison")
    return MatmulTiledConfig(
        name=config["name"],
        seed=config["seed"],
        naive=LaunchControls(group_size=_list(naive["group_size"], "group_size")),
        variants=_list(config["variants"], "variants"),
        correctness=CorrectnessControls(
            shapes=_shapes(correctness["shapes"], "correctness shapes")
        ),
        benchmark=BenchmarkControls(
            shapes=_shapes(benchmark["shapes"], "benchmark shapes"),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_flops_per_sample=benchmark["target_flops_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MatmulTiledConfig:
    """Parse controls, reporting every contract violation as ``MatmulTiledError``.

    The Day 74 sections raise their own lab's errors; they are rewrapped, so
    a caller catches one type.
    """

    try:
        return _config(payload)
    except MatmulTiledError:
        raise
    except (TypeError, ValueError) as error:
        raise MatmulTiledError(f"invalid matmul-tiled config: {error}") from error


def load_config(path: Path) -> MatmulTiledConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``MatmulTiledError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MatmulTiledError(str(error)) from error
    return config_from_mapping(payload)
