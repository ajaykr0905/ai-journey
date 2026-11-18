"""Strict, bounded controls for one Metal launch-shape sweep.

A config names the seed, the 2D threadgroup shapes of the matmul sweep with
its correctness cases, sizes and FLOP target, the 1D group sizes of the
vector sweep with its correctness length, lengths and byte target, and the
timing controls. Parsing rejects unknown and missing fields at every level,
booleans in place of integers, and values out of range. The config is also
checked as a whole before anything runs:

- every shape has 2 or more threads on each axis, and the correctness cases
  leave padding threads on each axis of each shape, so every bounds guard
  runs;
- the vector correctness length is not a multiple of any group size;
- every sweep point fits the batch limit, and the planned calls, FLOPs and
  bytes of the run fit the work bounds.

The planned work assumes that each pipeline admits every shape, so it is an
upper bound of what the run launches. Files are read through the bounded
strict reader of ``memory_access_report_io``, which also rejects duplicate
keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_benchmark import (
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)
from .kernel_metal_geometry import KernelMetalError, require_integer
from .kernel_metal_reference import MAX_SEED, MAX_VECTOR_LENGTH
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .metal_launch_sweep_benchmark import (
    MAX_TARGET_FLOPS,
    MIN_TARGET_FLOPS,
    VECTOR_BYTES_PER_ELEMENT,
    matmul_calls_per_sample,
    matmul_flops,
)
from .metal_launch_sweep_shapes import (
    MAX_MATRIX_SIDE,
    MAX_SHAPES,
    MetalLaunchSweepError,
    group_shape,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 8
MAX_CASE_SIDE = 512
MAX_POINTS = 4
MAX_RUN_CALLS = 1 << 22
MAX_RUN_FLOPS = 1 << 44
MAX_RUN_BYTES = 1 << 38

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {"schema_version", "name", "seed", "matmul", "vector", "timing"}
_MATMUL_FIELDS = {
    "group_shapes",
    "correctness_cases",
    "sizes",
    "target_flops_per_sample",
}
_VECTOR_FIELDS = {
    "group_sizes",
    "correctness_length",
    "lengths",
    "target_bytes_per_sample",
}
_TIMING_FIELDS = {"warmup", "repeats", "max_runtime_seconds"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MetalLaunchSweepError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MetalLaunchSweepError(f"{label} must be a JSON array")
    return tuple(value)


def _nested(value: object, label: str) -> tuple[tuple[object, ...], ...]:
    return tuple(_list(item, f"{label} item") for item in _list(value, label))


def _count(values: object, maximum: int, label: str) -> tuple[object, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= maximum:
        raise MetalLaunchSweepError(f"{label} must hold 1 to {maximum} items")
    if len(set(values)) != len(values):
        raise MetalLaunchSweepError(f"{label} must not repeat an item")
    return values


def _triples(values: object, side: int, label: str) -> tuple[tuple[int, int, int], ...]:
    for triple in _count(values, MAX_CASES, label):
        if type(triple) is not tuple or len(triple) != 3:
            raise MetalLaunchSweepError(f"{label} must be [m, n, k] triples")
        for name, size in zip("mnk", triple):
            require_integer(size, f"{label} {name}", 1, side)
    return values


@dataclass(frozen=True)
class MatmulControls:
    """The 2D shapes, correctness cases and timed sizes of the matmul sweep."""

    group_shapes: tuple[tuple[int, int], ...]
    correctness_cases: tuple[tuple[int, int, int], ...]
    sizes: tuple[tuple[int, int, int], ...]
    target_flops_per_sample: int

    def __post_init__(self) -> None:
        for shape in _count(self.group_shapes, MAX_SHAPES, "matmul group_shapes"):
            group_shape(shape, 2, "matmul group_shapes item")
            if min(shape) < 2:
                raise MetalLaunchSweepError(
                    "matmul group_shapes need 2 or more threads on each axis, or "
                    "the guard of that axis never runs"
                )
        _triples(self.correctness_cases, MAX_CASE_SIDE, "matmul correctness_cases")
        sizes = _triples(self.sizes, MAX_MATRIX_SIDE, "matmul sizes")
        if len(sizes) > MAX_POINTS:
            raise MetalLaunchSweepError(f"matmul sizes hold at most {MAX_POINTS} sizes")
        flops = [matmul_flops(*size) for size in sizes]
        if any(later <= earlier for earlier, later in zip(flops, flops[1:])):
            raise MetalLaunchSweepError(
                "matmul sizes must grow in FLOPs at every point"
            )
        require_integer(
            self.target_flops_per_sample,
            "target_flops_per_sample",
            MIN_TARGET_FLOPS,
            MAX_TARGET_FLOPS,
        )
        for gx, gy in self.group_shapes:
            if all(n % gx == 0 for _, n, _ in self.correctness_cases) or all(
                m % gy == 0 for m, _, _ in self.correctness_cases
            ):
                raise MetalLaunchSweepError(
                    f"matmul correctness_cases need an n that is not a multiple of "
                    f"{gx} and an m that is not a multiple of {gy}"
                )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "group_shapes": [list(shape) for shape in self.group_shapes],
            "correctness_cases": [list(case) for case in self.correctness_cases],
            "sizes": [list(size) for size in self.sizes],
            "target_flops_per_sample": self.target_flops_per_sample,
        }


@dataclass(frozen=True)
class VectorControls:
    """The 1D group sizes, correctness length and timed lengths of the vector sweep."""

    group_sizes: tuple[int, ...]
    correctness_length: int
    lengths: tuple[int, ...]
    target_bytes_per_sample: int

    def __post_init__(self) -> None:
        sizes = _count(self.group_sizes, MAX_SHAPES, "vector group_sizes")
        for size in sizes:
            group_shape((size,), 1, "vector group_sizes item")
            if size < 2:
                raise MetalLaunchSweepError(
                    "a vector group size of 1 divides every length, so its guard "
                    "never runs"
                )
        if any(later <= earlier for earlier, later in zip(sizes, sizes[1:])):
            raise MetalLaunchSweepError("vector group_sizes must be increasing")
        length = require_integer(
            self.correctness_length, "correctness_length", 1, MAX_VECTOR_LENGTH
        )
        if any(length % size == 0 for size in sizes):
            raise MetalLaunchSweepError(
                "vector correctness_length must not be a multiple of a group size"
            )
        lengths = _count(self.lengths, MAX_POINTS, "vector lengths")
        for value in lengths:
            require_integer(value, "vector lengths item", 1, MAX_VECTOR_LENGTH)
        if any(later <= earlier for earlier, later in zip(lengths, lengths[1:])):
            raise MetalLaunchSweepError("vector lengths must be increasing")
        require_integer(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "group_sizes": list(self.group_sizes),
            "correctness_length": self.correctness_length,
            "lengths": list(self.lengths),
            "target_bytes_per_sample": self.target_bytes_per_sample,
        }


@dataclass(frozen=True)
class TimingControls:
    """Untimed and timed samples per shape, and the wall-clock budget of a run."""

    warmup: int
    repeats: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        require_integer(self.warmup, "warmup", 0, MAX_WARMUP)
        require_integer(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_integer(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it."""

        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config can ask of the host if each pipeline admits every shape."""

    mapping_checks: int
    layout_probes: int
    correctness_launches: int
    benchmark_points: int
    timed_samples: int
    calls: int
    matmul_flops: int
    vector_bytes: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "mapping_checks": self.mapping_checks,
            "layout_probes": self.layout_probes,
            "correctness_launches": self.correctness_launches,
            "benchmark_points": self.benchmark_points,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "matmul_flops": self.matmul_flops,
            "vector_bytes": self.vector_bytes,
        }


@dataclass(frozen=True)
class MetalLaunchSweepConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    matmul: MatmulControls
    vector: VectorControls
    timing: TimingControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalLaunchSweepError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.matmul, MatmulControls),
            (self.vector, VectorControls),
            (self.timing, TimingControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.matmul_flops, MAX_RUN_FLOPS, "matmul FLOPs"),
            (work.vector_bytes, MAX_RUN_BYTES, "vector bytes"),
        ):
            if value > limit:
                raise MetalLaunchSweepError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def matmul_batches(self) -> tuple[int, ...]:
        """Return the calls per sample of each matmul size, in run order."""

        target = self.matmul.target_flops_per_sample
        return tuple(
            matmul_calls_per_sample(matmul_flops(*size), target)
            for size in self.matmul.sizes
        )

    def vector_batches(self) -> tuple[int, ...]:
        """Return the calls per sample of each vector length, in run order."""

        target = self.vector.target_bytes_per_sample
        return tuple(
            calls_per_sample(VECTOR_BYTES_PER_ELEMENT * length, target)
            for length in self.vector.lengths
        )

    def planned_work(self) -> PlannedWork:
        """Count the checks, samples, calls, FLOPs and bytes of a run up front.

        The batch sizes are the ones the run will use, so a point that would
        need an oversized sample fails here, before anything is launched.
        """

        shapes = len(self.matmul.group_shapes)
        sizes = len(self.vector.group_sizes)
        samples = self.timing.warmup + self.timing.repeats
        matmul_calls = [shapes * samples * batch for batch in self.matmul_batches()]
        vector_calls = [sizes * samples * batch for batch in self.vector_batches()]
        return PlannedWork(
            mapping_checks=shapes,
            layout_probes=shapes + sizes,
            correctness_launches=shapes * len(self.matmul.correctness_cases) + sizes,
            benchmark_points=len(self.matmul.sizes) + len(self.vector.lengths),
            timed_samples=self.timing.repeats
            * (shapes * len(self.matmul.sizes) + sizes * len(self.vector.lengths)),
            calls=sum(matmul_calls) + sum(vector_calls),
            matmul_flops=sum(
                calls * matmul_flops(*size)
                for calls, size in zip(matmul_calls, self.matmul.sizes)
            ),
            vector_bytes=sum(
                calls * VECTOR_BYTES_PER_ELEMENT * length
                for calls, length in zip(vector_calls, self.vector.lengths)
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "matmul": self.matmul.to_mapping(),
            "vector": self.vector.to_mapping(),
            "timing": self.timing.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> MetalLaunchSweepConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MetalLaunchSweepError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    matmul = _fields(config["matmul"], _MATMUL_FIELDS, "matmul")
    vector = _fields(config["vector"], _VECTOR_FIELDS, "vector")
    timing = _fields(config["timing"], _TIMING_FIELDS, "timing")
    return MetalLaunchSweepConfig(
        name=config["name"],
        seed=config["seed"],
        matmul=MatmulControls(
            group_shapes=_nested(matmul["group_shapes"], "matmul group_shapes"),
            correctness_cases=_nested(
                matmul["correctness_cases"], "matmul correctness_cases"
            ),
            sizes=_nested(matmul["sizes"], "matmul sizes"),
            target_flops_per_sample=matmul["target_flops_per_sample"],
        ),
        vector=VectorControls(
            group_sizes=_list(vector["group_sizes"], "vector group_sizes"),
            correctness_length=vector["correctness_length"],
            lengths=_list(vector["lengths"], "vector lengths"),
            target_bytes_per_sample=vector["target_bytes_per_sample"],
        ),
        timing=TimingControls(
            warmup=timing["warmup"],
            repeats=timing["repeats"],
            max_runtime_seconds=timing["max_runtime_seconds"],
        ),
    )


def config_from_mapping(payload: object) -> MetalLaunchSweepConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except MetalLaunchSweepError:
        raise
    except (KernelMetalError, TypeError) as error:
        raise MetalLaunchSweepError(
            f"invalid metal-launch-sweep config: {error}"
        ) from error


def load_config(path: Path) -> MetalLaunchSweepConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the
    byte bound, and rejects duplicate keys, non-finite constants and
    non-object roots; its errors are reported as ``MetalLaunchSweepError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MetalLaunchSweepError(str(error)) from error
    return config_from_mapping(payload)
