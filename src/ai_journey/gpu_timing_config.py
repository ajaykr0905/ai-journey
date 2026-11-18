"""Strict, bounded controls for one three-clock GPU timing run.

A config names the seed, the threadgroup size of both Day 74 mappings, the
sweep of matmul shapes with its warmups, repeats and time budget, and the
Day 74 measured FMA ceiling that the bottleneck inference compares with.
Parsing rejects unknown and missing fields at every level, booleans posing
as integers, and out-of-range values. The config is also checked as a whole:

- The sweep must run from a tiny shape to larger ones in strictly rising
  FLOPs, so the smallest point measures overhead and the fit has a span.
- It needs at least three points, the least a line fit with a residual takes.
- The run's total launches and FLOPs must fit the Day 74 work bounds.

Each timed sample holds one launch, not a batch, because the launch cost
itself is what this lab measures. Files go through the bounded strict reader
of ``memory_access_report_io``, which also rejects duplicate keys.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .gpu_timing_clocks import CLOCKS, GpuTimingError
from .gpu_timing_fit import MIN_FIT_POINTS
from .kernel_metal_benchmark import MAX_POINTS, MAX_RUNTIME_SECONDS, MIN_REPEATS
from .kernel_metal_reference import MAX_SEED
from .matmul_metal_config import MAX_RUN_CALLS, MAX_RUN_FLOPS, LaunchControls
from .matmul_metal_mapping import matmul_shape
from .matmul_metal_runtime import SIDES
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
# The smallest sweep point must be at most a 8 x 8 x 8 matmul.
MAX_TINY_FLOPS = 1 << 10
MAX_CEILING_GFLOPS = 1e7
MAX_SOURCE_TEXT = 200

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {"schema_version", "name", "seed", "launch", "sweep", "reference"}
_LAUNCH_FIELDS = {"group_size"}
_SWEEP_FIELDS = {"shapes", "warmup", "repeats", "max_runtime_seconds"}
_REFERENCE_FIELDS = {"fma_ceiling_gflops_per_second", "source"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise GpuTimingError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise GpuTimingError(f"{label} must be a JSON array")
    return tuple(value)


@dataclass(frozen=True)
class SweepControls:
    """The shapes, from tiny to large, and the warmups, repeats and budget."""

    shapes: tuple[MatmulShape, ...]
    warmup: int
    repeats: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        shapes = self.shapes
        if type(shapes) is not tuple or not MIN_FIT_POINTS <= len(shapes) <= MAX_POINTS:
            raise GpuTimingError(
                f"the sweep must hold {MIN_FIT_POINTS} to {MAX_POINTS} shapes"
            )
        for shape in shapes:
            if not isinstance(shape, MatmulShape):
                raise TypeError("sweep shapes must be MatmulShape values")
            matmul_shape(shape.m, shape.n, shape.k)
        flops = [shape.flops for shape in shapes]
        if any(later <= earlier for earlier, later in zip(flops, flops[1:])):
            raise GpuTimingError("sweep shapes must rise strictly in FLOPs")
        if flops[0] > MAX_TINY_FLOPS:
            raise GpuTimingError(
                f"the first sweep shape must be tiny: at most {MAX_TINY_FLOPS} FLOPs"
            )
        require_count(self.warmup, "warmup", 0, MAX_WARMUP)
        require_count(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_count(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "shapes": [[shape.m, shape.n, shape.k] for shape in self.shapes],
            "warmup": self.warmup,
            "repeats": self.repeats,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


def require_count(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return a true integer in bounds, reported as a timing error."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise GpuTimingError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class ReferenceCeiling:
    """The Day 74 measured FMA ceiling and the record it comes from.

    It is a number measured on this host by an earlier lab, not a vendor
    peak, and the bottleneck inference quotes it with its source.
    """

    fma_ceiling_gflops_per_second: float
    source: str

    def __post_init__(self) -> None:
        value = self.fma_ceiling_gflops_per_second
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 < value <= MAX_CEILING_GFLOPS
        ):
            raise GpuTimingError(
                "fma_ceiling_gflops_per_second must be a float in "
                f"(0, {MAX_CEILING_GFLOPS}]"
            )
        if (
            not isinstance(self.source, str)
            or not 1 <= len(self.source) <= MAX_SOURCE_TEXT
            or not self.source.isascii()
            or not self.source.isprintable()
        ):
            raise GpuTimingError(
                f"reference source must be 1 to {MAX_SOURCE_TEXT} printable ASCII"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "fma_ceiling_gflops_per_second": self.fma_ceiling_gflops_per_second,
            "source": self.source,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host, counted before anything runs."""

    sweep_points: int
    sides: int
    timed_samples: int
    launches: int
    flops: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "sweep_points": self.sweep_points,
            "sides": self.sides,
            "timed_samples": self.timed_samples,
            "launches": self.launches,
            "flops": self.flops,
        }


@dataclass(frozen=True)
class GpuTimingConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    launch: LaunchControls
    sweep: SweepControls
    reference: ReferenceCeiling

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuTimingError("config name must be a lowercase slug")
        require_count(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.launch, LaunchControls),
            (self.sweep, SweepControls),
            (self.reference, ReferenceCeiling),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work.launches, MAX_RUN_CALLS, "launches"),
            (work.flops, MAX_RUN_FLOPS, "FLOPs"),
        ):
            if value > limit:
                raise GpuTimingError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the sides, samples, launches and FLOPs of the run up front.

        Every side runs ``warmup + repeats`` one-launch samples on each clock.
        The empty kernel is one more side with no FLOPs.
        """

        sweep = self.sweep
        per_side = len(CLOCKS) * (sweep.warmup + sweep.repeats)
        sides = 1 + len(SIDES) * len(sweep.shapes)
        return PlannedWork(
            sweep_points=len(sweep.shapes),
            sides=sides,
            timed_samples=sides * len(CLOCKS) * sweep.repeats,
            launches=sides * per_side,
            flops=sum(len(SIDES) * per_side * shape.flops for shape in sweep.shapes),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "launch": self.launch.to_mapping(),
            "sweep": self.sweep.to_mapping(),
            "reference": self.reference.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _shapes(value: object) -> tuple[MatmulShape, ...]:
    shapes = []
    for item in _list(value, "sweep shapes"):
        triple = _list(item, "sweep shapes item")
        if len(triple) != 3:
            raise GpuTimingError("sweep shapes must be [m, n, k] triples")
        shapes.append(matmul_shape(*triple))
    return tuple(shapes)


def _config(payload: object) -> GpuTimingConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise GpuTimingError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    launch = _fields(config["launch"], _LAUNCH_FIELDS, "launch")
    sweep = _fields(config["sweep"], _SWEEP_FIELDS, "sweep")
    reference = _fields(config["reference"], _REFERENCE_FIELDS, "reference")
    return GpuTimingConfig(
        name=config["name"],
        seed=config["seed"],
        launch=LaunchControls(group_size=_list(launch["group_size"], "group_size")),
        sweep=SweepControls(
            shapes=_shapes(sweep["shapes"]),
            warmup=sweep["warmup"],
            repeats=sweep["repeats"],
            max_runtime_seconds=sweep["max_runtime_seconds"],
        ),
        reference=ReferenceCeiling(
            fma_ceiling_gflops_per_second=reference["fma_ceiling_gflops_per_second"],
            source=reference["source"],
        ),
    )


def config_from_mapping(payload: object) -> GpuTimingConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except TypeError as error:
        raise GpuTimingError(f"invalid gpu-timing config: {error}") from error


def load_config(path: Path) -> GpuTimingConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``GpuTimingError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise GpuTimingError(str(error)) from error
    return config_from_mapping(payload)
