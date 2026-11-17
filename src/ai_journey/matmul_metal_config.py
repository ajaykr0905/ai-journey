"""Strict, bounded controls for one Metal matmul correctness and benchmark run.

A config names the seed, the threadgroup size that both mappings share, the
shapes of the correctness cases, and the benchmark sweep with its timing
controls. Parsing rejects unknown and missing fields at every level,
booleans posing as integers, and out-of-range values. The config is also
checked as a whole before anything runs:

- The correctness shapes must leave padding threads on both axes of both
  mappings, so every guard runs, and one shape must be non-square.
- Each correctness output, with its guard tail, must fit one sentinel buffer,
  and the CPU replay of all cases must fit its step bound.
- Every sweep point must fit the batch limit, and the run's total calls and
  FLOPs must fit the work bounds.

Files go through the bounded strict reader of ``memory_access_report_io``,
which also rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_benchmark import MAX_POINTS, MAX_RUNTIME_SECONDS, MIN_REPEATS
from .kernel_metal_geometry import MAX_THREADGROUP_THREADS, require_integer
from .kernel_metal_reference import MAX_SEED, MAX_VECTOR_BUFFER
from .matmul_metal_benchmark import (
    MAX_TARGET_FLOPS,
    MIN_TARGET_FLOPS,
    calls_per_sample,
)
from .matmul_metal_mapping import (
    MAPPINGS,
    MatmulMetalError,
    matmul_launch,
    matmul_shape,
)
from .matmul_metal_runtime import SIDES
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
# The CPU replays every correctness case one fma step per output and k.
MAX_REPLAY_STEPS = 1 << 28
MAX_RUN_CALLS = 1 << 22
MAX_RUN_FLOPS = 1 << 45

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {"schema_version", "name", "seed", "launch", "correctness", "benchmark"}
_LAUNCH_FIELDS = {"group_size"}
_CORRECTNESS_FIELDS = {"shapes"}
_BENCHMARK_FIELDS = {
    "shapes",
    "warmup",
    "repeats",
    "target_flops_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MatmulMetalError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MatmulMetalError(f"{label} must be a JSON array")
    return tuple(value)


def _parse_shapes(value: object, label: str) -> tuple[MatmulShape, ...]:
    shapes = []
    for item in _list(value, label):
        triple = _list(item, f"{label} item")
        if len(triple) != 3:
            raise MatmulMetalError(f"{label} must be [m, n, k] triples")
        shapes.append(matmul_shape(*triple))
    return tuple(shapes)


def _shapes(values: object, label: str) -> tuple[MatmulShape, ...]:
    """Require 1 to ``MAX_POINTS`` distinct bounded shapes."""

    if type(values) is not tuple or not 1 <= len(values) <= MAX_POINTS:
        raise MatmulMetalError(f"{label} must hold 1 to {MAX_POINTS} shapes")
    for shape in values:
        if not isinstance(shape, MatmulShape):
            raise TypeError(f"{label} must hold MatmulShape values")
        matmul_shape(shape.m, shape.n, shape.k)
    if len(set(values)) != len(values):
        raise MatmulMetalError(f"{label} must not repeat a shape")
    return values


def _triples(shapes: tuple[MatmulShape, ...]) -> list[list[int]]:
    return [[shape.m, shape.n, shape.k] for shape in shapes]


@dataclass(frozen=True)
class LaunchControls:
    """The ``(x, y)`` threadgroup size that both mappings use: CUDA's block."""

    group_size: tuple[int, int]

    def __post_init__(self) -> None:
        group = self.group_size
        if type(group) is not tuple or len(group) != 2:
            raise MatmulMetalError("group_size must be an [x, y] pair")
        for axis, size in zip("xy", group):
            require_integer(size, f"group_size {axis}", 1, MAX_THREADGROUP_THREADS)
        if group[0] * group[1] > MAX_THREADGROUP_THREADS:
            raise MatmulMetalError(
                f"group_size holds more than {MAX_THREADGROUP_THREADS} threads"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {"group_size": list(self.group_size)}


@dataclass(frozen=True)
class CorrectnessControls:
    """The shapes of the seeded correctness cases."""

    shapes: tuple[MatmulShape, ...]

    def __post_init__(self) -> None:
        _shapes(self.shapes, "correctness shapes")

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {"shapes": _triples(self.shapes)}


@dataclass(frozen=True)
class BenchmarkControls:
    """The sweep of shapes and the warmups, repeats, batch target and budget."""

    shapes: tuple[MatmulShape, ...]
    warmup: int
    repeats: int
    target_flops_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        _shapes(self.shapes, "benchmark shapes")
        require_integer(self.warmup, "warmup", 0, MAX_WARMUP)
        require_integer(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_integer(
            self.target_flops_per_sample,
            "target_flops_per_sample",
            MIN_TARGET_FLOPS,
            MAX_TARGET_FLOPS,
        )
        require_integer(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "shapes": _triples(self.shapes),
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_flops_per_sample": self.target_flops_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host, counted before anything runs."""

    correctness_cases: int
    replay_steps: int
    benchmark_points: int
    timed_samples: int
    calls: int
    flops: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "replay_steps": self.replay_steps,
            "benchmark_points": self.benchmark_points,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "flops": self.flops,
        }


@dataclass(frozen=True)
class MatmulMetalConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    launch: LaunchControls
    correctness: CorrectnessControls
    benchmark: BenchmarkControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MatmulMetalError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.launch, LaunchControls),
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        group_x, group_y = self.launch.group_size
        shapes = self.correctness.shapes
        for axis, label in (("m", "M"), ("n", "N")):
            for group, name in ((group_x, "x"), (group_y, "y")):
                if all(getattr(shape, axis) % group == 0 for shape in shapes):
                    raise MatmulMetalError(
                        f"correctness shapes need an {label} that is not a multiple "
                        f"of the group {name} size {group}, so both mappings pad "
                        "both axes"
                    )
        if all(shape.m == shape.n for shape in shapes):
            raise MatmulMetalError("correctness shapes need a non-square shape")
        for shape in shapes:
            for mapping in MAPPINGS:
                launch = matmul_launch(mapping, shape, self.launch.group_size)
                if launch.launched_threads > MAX_VECTOR_BUFFER:
                    raise MatmulMetalError(
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
                raise MatmulMetalError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the cases, samples, calls and FLOPs of the run up front.

        The batch sizes are the ones the run will use, so a point that would
        need an oversized sample fails here, before anything is launched.
        """

        timing = self.benchmark
        samples = len(SIDES) * (timing.warmup + timing.repeats)
        calls = flops = 0
        for shape in timing.shapes:
            batch = calls_per_sample(shape.flops, timing.target_flops_per_sample)
            calls += samples * batch
            flops += samples * batch * shape.flops
        return PlannedWork(
            correctness_cases=len(self.correctness.shapes),
            replay_steps=sum(
                shape.m * shape.n * shape.k for shape in self.correctness.shapes
            ),
            benchmark_points=len(timing.shapes),
            timed_samples=len(SIDES) * len(timing.shapes) * timing.repeats,
            calls=calls,
            flops=flops,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "launch": self.launch.to_mapping(),
            "correctness": self.correctness.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> MatmulMetalConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MatmulMetalError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    launch = _fields(config["launch"], _LAUNCH_FIELDS, "launch")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    return MatmulMetalConfig(
        name=config["name"],
        seed=config["seed"],
        launch=LaunchControls(group_size=_list(launch["group_size"], "group_size")),
        correctness=CorrectnessControls(
            shapes=_parse_shapes(correctness["shapes"], "correctness shapes")
        ),
        benchmark=BenchmarkControls(
            shapes=_parse_shapes(benchmark["shapes"], "benchmark shapes"),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_flops_per_sample=benchmark["target_flops_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
    )


def config_from_mapping(payload: object) -> MatmulMetalConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except TypeError as error:
        raise MatmulMetalError(f"invalid matmul-metal config: {error}") from error


def load_config(path: Path) -> MatmulMetalConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``MatmulMetalError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MatmulMetalError(str(error)) from error
    return config_from_mapping(payload)
