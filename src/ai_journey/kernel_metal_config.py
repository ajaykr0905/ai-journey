"""Strict, bounded controls for one Metal kernel correctness and benchmark run.

A config names the seed, the threadgroup size of each kernel, the extents of
the correctness cases, and the benchmark sweep with its timing controls.
Parsing rejects unknown and missing fields at every level, booleans posing
as integers, and out-of-range values. The config is also checked as a whole
before anything runs: the correctness cases must include extents that are
not multiples of the group size on every axis, so the bounds guards are
exercised, and every sweep point must fit the batch limit and the run's
total calls and accounted bytes must fit the work bounds. Files are read
through the bounded strict reader of ``memory_access_report_io``, which also
rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_benchmark import (
    MAX_POINTS,
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)
from .kernel_metal_correctness import ALL_RGB_TRIPLES, UNIFORM
from .kernel_metal_geometry import (
    MAX_THREADGROUP_THREADS,
    KernelMetalError,
    require_integer,
)
from .kernel_metal_reference import (
    ALL_TRIPLES_SHAPE,
    MAX_SEED,
    MAX_VECTOR_LENGTH,
    require_image_shape,
)
from .kernel_metal_source import KERNELS, RGB_TO_GRAY, VECTOR_ADD
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_RUN_CALLS = 1 << 22
MAX_RUN_BYTES = 1 << 38

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {"schema_version", "name", "seed", "launch", "correctness", "benchmark"}
_LAUNCH_FIELDS = {"vector_add_group_size", "rgb_to_gray_group_size"}
_CORRECTNESS_FIELDS = {
    "vector_add_lengths",
    "rgb_to_gray_shapes",
    "rgb_to_gray_all_triples",
}
_BENCHMARK_FIELDS = {
    "vector_add_lengths",
    "rgb_to_gray_shapes",
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise KernelMetalError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise KernelMetalError(f"{label} must be a JSON array")
    return tuple(value)


def _shapes(value: object, label: str) -> tuple[tuple[object, ...], ...]:
    return tuple(_list(item, f"{label} item") for item in _list(value, label))


def _lengths(values: object, label: str) -> tuple[int, ...]:
    """Require 1 to ``MAX_POINTS`` strictly increasing vector lengths."""

    if type(values) is not tuple or not 1 <= len(values) <= MAX_POINTS:
        raise KernelMetalError(f"{label} must hold 1 to {MAX_POINTS} lengths")
    for length in values:
        require_integer(length, label, 1, MAX_VECTOR_LENGTH)
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise KernelMetalError(f"{label} must be strictly increasing")
    return values


def _image_shapes(values: object, label: str) -> tuple[tuple[int, int], ...]:
    """Require 1 to ``MAX_POINTS`` distinct ``(height, width)`` shapes."""

    if type(values) is not tuple or not 1 <= len(values) <= MAX_POINTS:
        raise KernelMetalError(f"{label} must hold 1 to {MAX_POINTS} shapes")
    for shape in values:
        if type(shape) is not tuple or len(shape) != 2:
            raise KernelMetalError(f"{label} must be [height, width] pairs")
        require_image_shape(*shape)
    if len(set(values)) != len(values):
        raise KernelMetalError(f"{label} must not repeat a shape")
    return values


@dataclass(frozen=True)
class LaunchControls:
    """Threadgroup size of each kernel: CUDA's block size."""

    vector_add_group_size: int
    rgb_to_gray_group_size: tuple[int, int]

    def __post_init__(self) -> None:
        require_integer(
            self.vector_add_group_size,
            "vector_add_group_size",
            1,
            MAX_THREADGROUP_THREADS,
        )
        group = self.rgb_to_gray_group_size
        if type(group) is not tuple or len(group) != 2:
            raise KernelMetalError("rgb_to_gray_group_size must be an [x, y] pair")
        for axis, size in zip("xy", group):
            require_integer(
                size, f"rgb_to_gray_group_size {axis}", 1, MAX_THREADGROUP_THREADS
            )
        if group[0] * group[1] > MAX_THREADGROUP_THREADS:
            raise KernelMetalError(
                f"rgb_to_gray_group_size holds more than {MAX_THREADGROUP_THREADS} "
                "threads"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "vector_add_group_size": self.vector_add_group_size,
            "rgb_to_gray_group_size": list(self.rgb_to_gray_group_size),
        }


@dataclass(frozen=True)
class CorrectnessControls:
    """Extents of the correctness cases, and whether to add every RGB triple."""

    vector_add_lengths: tuple[int, ...]
    rgb_to_gray_shapes: tuple[tuple[int, int], ...]
    rgb_to_gray_all_triples: bool

    def __post_init__(self) -> None:
        _lengths(self.vector_add_lengths, "correctness vector_add_lengths")
        _image_shapes(self.rgb_to_gray_shapes, "correctness rgb_to_gray_shapes")
        if type(self.rgb_to_gray_all_triples) is not bool:
            raise KernelMetalError("rgb_to_gray_all_triples must be true or false")

    def gray_cases(self) -> tuple[tuple[int, tuple[int, int], str], ...]:
        """Return ``(index, shape, pattern)`` per case: seeded images, then all triples."""

        cases = [
            (index, shape, UNIFORM)
            for index, shape in enumerate(self.rgb_to_gray_shapes)
        ]
        if self.rgb_to_gray_all_triples:
            cases.append((len(cases), ALL_TRIPLES_SHAPE, ALL_RGB_TRIPLES))
        return tuple(cases)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "vector_add_lengths": list(self.vector_add_lengths),
            "rgb_to_gray_shapes": [list(shape) for shape in self.rgb_to_gray_shapes],
            "rgb_to_gray_all_triples": self.rgb_to_gray_all_triples,
        }


@dataclass(frozen=True)
class BenchmarkControls:
    """The sweep of extents and the warmups, repeats, batch target and budget."""

    vector_add_lengths: tuple[int, ...]
    rgb_to_gray_shapes: tuple[tuple[int, int], ...]
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        _lengths(self.vector_add_lengths, "benchmark vector_add_lengths")
        shapes = _image_shapes(self.rgb_to_gray_shapes, "benchmark rgb_to_gray_shapes")
        pixels = [height * width for height, width in shapes]
        if any(later <= earlier for earlier, later in zip(pixels, pixels[1:])):
            raise KernelMetalError(
                "benchmark rgb_to_gray_shapes must grow in pixels at every point"
            )
        require_integer(self.warmup, "warmup", 0, MAX_WARMUP)
        require_integer(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_integer(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )
        require_integer(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "vector_add_lengths": list(self.vector_add_lengths),
            "rgb_to_gray_shapes": [list(shape) for shape in self.rgb_to_gray_shapes],
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host, counted before anything runs."""

    correctness_cases: int
    benchmark_points: int
    timed_samples: int
    calls: int
    accounted_bytes: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "benchmark_points": self.benchmark_points,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "accounted_bytes": self.accounted_bytes,
        }


@dataclass(frozen=True)
class KernelMetalConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    launch: LaunchControls
    correctness: CorrectnessControls
    benchmark: BenchmarkControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise KernelMetalError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.launch, LaunchControls),
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        group = self.launch.vector_add_group_size
        if all(length % group == 0 for length in self.correctness.vector_add_lengths):
            raise KernelMetalError(
                "correctness vector_add_lengths need one length that is not a "
                f"multiple of the group size {group}"
            )
        group_x, group_y = self.launch.rgb_to_gray_group_size
        shapes = self.correctness.rgb_to_gray_shapes
        if all(width % group_x == 0 for _, width in shapes) or all(
            height % group_y == 0 for height, _ in shapes
        ):
            raise KernelMetalError(
                "correctness rgb_to_gray_shapes need a width that is not a multiple "
                "of the group x size and a height that is not a multiple of the "
                "group y size"
            )
        work = self.planned_work()
        for value, limit, label in (
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.accounted_bytes, MAX_RUN_BYTES, "accounted bytes"),
        ):
            if value > limit:
                raise KernelMetalError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def benchmark_extents(self) -> tuple[tuple[str, int], ...]:
        """Return ``(kernel, items per call)`` for every sweep point, in run order."""

        return tuple(
            [(VECTOR_ADD, length) for length in self.benchmark.vector_add_lengths]
            + [
                (RGB_TO_GRAY, height * width)
                for height, width in self.benchmark.rgb_to_gray_shapes
            ]
        )

    def planned_work(self) -> PlannedWork:
        """Count the cases, samples, calls and bytes of the run up front.

        The batch sizes are the ones the run will use, so a point that would
        need an oversized sample fails here, before anything is launched.
        """

        timing = self.benchmark
        samples = timing.warmup + timing.repeats
        calls = accounted = 0
        for kernel, items in self.benchmark_extents():
            per_call = KERNELS[kernel].bytes_per_item * items
            batch = calls_per_sample(per_call, timing.target_bytes_per_sample)
            # Both sides, the Metal kernel and PyTorch, run every sample.
            calls += 2 * samples * batch
            accounted += 2 * samples * batch * per_call
        points = len(self.benchmark_extents())
        return PlannedWork(
            correctness_cases=len(self.correctness.vector_add_lengths)
            + len(self.correctness.gray_cases()),
            benchmark_points=points,
            timed_samples=2 * points * timing.repeats,
            calls=calls,
            accounted_bytes=accounted,
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


def _config(payload: object) -> KernelMetalConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise KernelMetalError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    launch = _fields(config["launch"], _LAUNCH_FIELDS, "launch")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    return KernelMetalConfig(
        name=config["name"],
        seed=config["seed"],
        launch=LaunchControls(
            vector_add_group_size=launch["vector_add_group_size"],
            rgb_to_gray_group_size=_list(
                launch["rgb_to_gray_group_size"], "rgb_to_gray_group_size"
            ),
        ),
        correctness=CorrectnessControls(
            vector_add_lengths=_list(
                correctness["vector_add_lengths"], "correctness vector_add_lengths"
            ),
            rgb_to_gray_shapes=_shapes(
                correctness["rgb_to_gray_shapes"], "correctness rgb_to_gray_shapes"
            ),
            rgb_to_gray_all_triples=correctness["rgb_to_gray_all_triples"],
        ),
        benchmark=BenchmarkControls(
            vector_add_lengths=_list(
                benchmark["vector_add_lengths"], "benchmark vector_add_lengths"
            ),
            rgb_to_gray_shapes=_shapes(
                benchmark["rgb_to_gray_shapes"], "benchmark rgb_to_gray_shapes"
            ),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
    )


def config_from_mapping(payload: object) -> KernelMetalConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except TypeError as error:
        raise KernelMetalError(f"invalid kernel-metal config: {error}") from error


def load_config(path: Path) -> KernelMetalConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``KernelMetalError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise KernelMetalError(str(error)) from error
    return config_from_mapping(payload)
