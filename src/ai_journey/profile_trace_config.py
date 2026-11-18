"""Strict, bounded controls for one profiler capture of the Day 74 matmul kernels.

A config names the seed and threadgroup size of the Day 74 naive kernels,
the 1 to 3 shapes to profile, the untimed warmup and profiled iterations,
the analysis controls, and the measured Day 74 ceilings that the
bottleneck inference compares against. Parsing rejects unknown and missing
fields at every level, booleans posing as numbers, and out-of-range values,
and it bounds the total work before anything runs.

The ceilings are values measured on the development host by the Day 74
labs (``docs/gpu-ceilings-lab.md`` and ``docs/matmul-metal-lab.md``). They
are copied into the config, so a report binds the exact figures it used;
they are not hardware or vendor peaks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_geometry import KernelMetalError
from .matmul_metal_config import LaunchControls
from .matmul_metal_mapping import matmul_shape
from .matmul_metal_runtime import SIDES
from .memory_access_pattern import MemoryAccessError
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape
from .profile_trace_analysis import MAX_TOP_N
from .profile_trace_events import ProfileTraceError, require_integer
from .profile_trace_timeline import MAX_ROWS, MAX_WIDTH, MIN_WIDTH

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_SHAPES = 3
MAX_WARMUP = 10
MIN_ITERATIONS = 2
MAX_ITERATIONS = 20
MAX_SEED = (1 << 32) - 1
# Every side of every warmup and profiled iteration, at about 150 GFLOP/s
# for the slowest naive kernel, takes about one minute at most.
MAX_CAPTURE_FLOPS = 1 << 43
MAX_CEILING = 1e6

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "group_size",
    "shapes",
    "warmup",
    "iterations",
    "analysis",
    "day_74_ceilings",
}
_ANALYSIS_FIELDS = {"top_n", "gap_count", "timeline_width", "timeline_depth"}
_CEILING_FIELDS = {
    "fma_gflops_per_second",
    "bandwidth_gb_per_second",
    "torch_matmul_gflops_per_second",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise ProfileTraceError(f"{label} fields do not match the schema")
    return payload


def _ceiling(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ProfileTraceError(f"{label} must be a finite JSON number")
    if not 0 < value <= MAX_CEILING:
        raise ProfileTraceError(f"{label} must be in (0, {MAX_CEILING:g}]")
    return float(value)


@dataclass(frozen=True)
class AnalysisControls:
    """How many top ops and gaps a report lists, and the timeline's size."""

    top_n: int
    gap_count: int
    timeline_width: int
    timeline_depth: int

    def __post_init__(self) -> None:
        require_integer(self.top_n, "top_n", 1, MAX_TOP_N)
        require_integer(self.gap_count, "gap_count", 1, MAX_TOP_N)
        require_integer(self.timeline_width, "timeline_width", MIN_WIDTH, MAX_WIDTH)
        require_integer(self.timeline_depth, "timeline_depth", 1, MAX_ROWS)

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it."""

        return {
            "top_n": self.top_n,
            "gap_count": self.gap_count,
            "timeline_width": self.timeline_width,
            "timeline_depth": self.timeline_depth,
        }


@dataclass(frozen=True)
class Day74Ceilings:
    """The measured Day 74 ceilings of this host, in GFLOP/s and GB/s."""

    fma_gflops_per_second: float
    bandwidth_gb_per_second: float
    torch_matmul_gflops_per_second: float

    def __post_init__(self) -> None:
        for field in sorted(_CEILING_FIELDS):
            value = getattr(self, field)
            if type(value) is not float:
                raise ProfileTraceError(f"{field} must be a float")
            _ceiling(value, field)

    def to_mapping(self) -> dict[str, float]:
        """Return the section as the config file spells it."""

        return {field: getattr(self, field) for field in sorted(_CEILING_FIELDS)}


@dataclass(frozen=True)
class ProfileTraceConfig:
    """Every control one capture reads, validated as a whole."""

    name: str
    seed: int
    group_size: tuple[int, int]
    shapes: tuple[MatmulShape, ...]
    warmup: int
    iterations: int
    analysis: AnalysisControls
    ceilings: Day74Ceilings

    def __post_init__(self) -> None:
        if type(self.name) is not str or _NAME.fullmatch(self.name) is None:
            raise ProfileTraceError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        try:
            LaunchControls(self.group_size)
        except (KernelMetalError, TypeError) as error:
            raise ProfileTraceError(str(error)) from error
        if type(self.shapes) is not tuple or not 1 <= len(self.shapes) <= MAX_SHAPES:
            raise ProfileTraceError(f"shapes must hold 1 to {MAX_SHAPES} shapes")
        for shape in self.shapes:
            if not isinstance(shape, MatmulShape):
                raise ProfileTraceError("shapes must hold MatmulShape values")
        if len(set(self.shapes)) != len(self.shapes):
            raise ProfileTraceError("shapes must not repeat a shape")
        require_integer(self.warmup, "warmup", 0, MAX_WARMUP)
        require_integer(self.iterations, "iterations", MIN_ITERATIONS, MAX_ITERATIONS)
        if not isinstance(self.analysis, AnalysisControls):
            raise ProfileTraceError("analysis must be AnalysisControls")
        if not isinstance(self.ceilings, Day74Ceilings):
            raise ProfileTraceError("day_74_ceilings must be Day74Ceilings")
        if self.capture_flops() > MAX_CAPTURE_FLOPS:
            raise ProfileTraceError(
                f"config needs {self.capture_flops()} FLOPs; the bound is "
                f"{MAX_CAPTURE_FLOPS}"
            )

    def capture_flops(self) -> int:
        """Count the FLOPs of every side of every warmup and profiled iteration."""

        rounds = len(SIDES) * (self.warmup + self.iterations)
        return sum(rounds * shape.flops for shape in self.shapes)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "group_size": list(self.group_size),
            "shapes": [[shape.m, shape.n, shape.k] for shape in self.shapes],
            "warmup": self.warmup,
            "iterations": self.iterations,
            "analysis": self.analysis.to_mapping(),
            "day_74_ceilings": self.ceilings.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _shape(value: object) -> MatmulShape:
    if type(value) is not list or len(value) != 3:
        raise ProfileTraceError("shapes must be [m, n, k] triples")
    return matmul_shape(*value)


def _config(payload: object) -> ProfileTraceConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise ProfileTraceError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    group = config["group_size"]
    if type(group) is not list:
        raise ProfileTraceError("group_size must be an [x, y] pair")
    shapes = config["shapes"]
    if type(shapes) is not list:
        raise ProfileTraceError("shapes must be a JSON array")
    analysis = _fields(config["analysis"], _ANALYSIS_FIELDS, "analysis")
    ceilings = _fields(config["day_74_ceilings"], _CEILING_FIELDS, "day_74_ceilings")
    return ProfileTraceConfig(
        name=config["name"],
        seed=config["seed"],
        group_size=tuple(group),
        shapes=tuple(_shape(item) for item in shapes),
        warmup=config["warmup"],
        iterations=config["iterations"],
        analysis=AnalysisControls(**analysis),
        ceilings=Day74Ceilings(
            **{key: _ceiling(value, key) for key, value in ceilings.items()}
        ),
    )


def config_from_mapping(payload: object) -> ProfileTraceConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except (KernelMetalError, MemoryAccessError, TypeError) as error:
        raise ProfileTraceError(f"invalid profile-trace config: {error}") from error


def load_config(path: Path) -> ProfileTraceConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except MemoryAccessError as error:
        raise ProfileTraceError(str(error)) from error
    return config_from_mapping(payload)
