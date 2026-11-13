"""Strict declared plan for the memory-access lab.

A plan names the warp patterns to analyze, the shared-tile transposes, the
tiled-matmul sweep, the declared roofline ceilings with their public source,
and a small CPU benchmark budget. Parsing rejects unknown and missing fields
at every level, booleans posing as integers, and anything outside the bounds
of the models, so a report can always be rebuilt from its embedded plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

from .memory_access_baseline import (
    BACKENDS,
    BASELINE_DTYPE,
    MAX_BASELINE_SEED,
    MatmulCase,
)
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_pattern import (
    MatrixLayout,
    MemoryAccessError,
    MemoryGeometry,
    WarpAccess,
    broadcast_access,
    column_access,
    contiguous_access,
    permuted_access,
    row_access,
    strided_access,
)
from .memory_access_roofline import MAX_SWEEP_TILES
from .memory_access_tiling import MAX_MATMUL_TILE, MatmulShape, matmul_element_bytes
from .memory_access_transpose import TransposeAnalysis, analyze_tile_transpose
from .roofline import RooflineHardware

PLAN_SCHEMA_VERSION = 1
MAX_PATTERNS = 64
MAX_TRANSPOSES = 16
MAX_SWEEP_SHAPES = 16
MAX_BENCHMARK_SHAPES = 8
MAX_BENCHMARK_FLOPS = 1 << 36
MAX_SOURCE_TEXT = 600
MAX_CEILING = 1e30

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PATTERN_FIELDS = {
    "contiguous": ({"access_bytes"}, {"base_bytes", "active_mask"}),
    "strided": ({"access_bytes", "stride_elements"}, {"base_bytes", "active_mask"}),
    "broadcast": ({"access_bytes", "address_bytes"}, {"active_mask"}),
    "row": (
        {"access_bytes", "rows", "columns", "leading_dimension", "row"},
        {"base_bytes", "start_column", "active_mask"},
    ),
    "column": (
        {"access_bytes", "rows", "columns", "leading_dimension", "column"},
        {"base_bytes", "start_row", "active_mask"},
    ),
    "permutation": ({"access_bytes", "seed"}, {"base_bytes", "active_mask"}),
}
PATTERN_KINDS = tuple(_PATTERN_FIELDS)
_PLAN_FIELDS = {
    "schema_version",
    "name",
    "geometry",
    "patterns",
    "transposes",
    "matmul",
    "roofline",
    "benchmark",
}


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON with sorted keys, compact separators, and ASCII escapes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise MemoryAccessError("value is not canonical finite JSON") from error


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise MemoryAccessError(f"{label} must be a lowercase slug")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise MemoryAccessError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MemoryAccessError(f"{label} fields do not match schema")
    return payload


def _unique_names(items: tuple[object, ...], label: str) -> None:
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise MemoryAccessError(f"{label} names must be unique")


@dataclass(frozen=True)
class PatternSpec:
    """One declared warp access pattern and its integer parameters."""

    name: str
    kind: str
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _name(self.name, "pattern name")
        if self.kind not in _PATTERN_FIELDS:
            raise MemoryAccessError(
                f"pattern kind must be one of {', '.join(PATTERN_KINDS)}"
            )
        if type(self.parameters) is not tuple or any(
            type(item) is not tuple or len(item) != 2 or not isinstance(item[0], str)
            for item in self.parameters
        ):
            raise TypeError("parameters must be a tuple of (name, value) pairs")
        keys = [key for key, _ in self.parameters]
        if keys != sorted(set(keys)):
            raise MemoryAccessError("pattern parameters must be unique and sorted")
        required, optional = _PATTERN_FIELDS[self.kind]
        if not required <= set(keys) <= required | optional:
            raise MemoryAccessError(f"{self.kind} pattern fields do not match schema")
        for key, value in self.parameters:
            if type(value) is not int:
                raise TypeError(f"{key} must be an integer")
        self.build()

    def build(self) -> WarpAccess:
        """Generate the declared warp request."""

        values = dict(self.parameters)
        width = values["access_bytes"]
        base = values.get("base_bytes", 0)
        if self.kind == "contiguous":
            access = contiguous_access(access_bytes=width, base_bytes=base)
        elif self.kind == "strided":
            access = strided_access(
                stride_elements=values["stride_elements"],
                access_bytes=width,
                base_bytes=base,
            )
        elif self.kind == "broadcast":
            access = broadcast_access(
                address_bytes=values["address_bytes"], access_bytes=width
            )
        elif self.kind == "permutation":
            access = permuted_access(
                seed=values["seed"], access_bytes=width, base_bytes=base
            )
        else:
            layout = MatrixLayout(
                rows=values["rows"],
                columns=values["columns"],
                leading_dimension=values["leading_dimension"],
                element_bytes=width,
                base_bytes=base,
            )
            if self.kind == "row":
                access = row_access(
                    layout,
                    row=values["row"],
                    start_column=values.get("start_column", 0),
                )
            else:
                access = column_access(
                    layout,
                    column=values["column"],
                    start_row=values.get("start_row", 0),
                )
        if "active_mask" in values:
            access = access.masked(values["active_mask"])
        return access

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, **dict(self.parameters)}


def pattern_from_mapping(payload: object) -> PatternSpec:
    if type(payload) is not dict or not {"name", "kind"} <= set(payload):
        raise MemoryAccessError("pattern must be an object with a name and kind")
    return PatternSpec(
        name=payload["name"],
        kind=payload["kind"],
        parameters=tuple(
            sorted(
                (key, value)
                for key, value in payload.items()
                if key not in {"name", "kind"}
            )
        ),
    )


@dataclass(frozen=True)
class TransposeSpec:
    """One declared shared-tile transpose comparison."""

    name: str
    leading_dimension: int
    element_bytes: int
    matrix_size: int

    def __post_init__(self) -> None:
        _name(self.name, "transpose name")
        self.analyze()

    def analyze(self) -> TransposeAnalysis:
        return analyze_tile_transpose(
            leading_dimension=self.leading_dimension,
            element_bytes=self.element_bytes,
            matrix_size=self.matrix_size,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "leading_dimension": self.leading_dimension,
            "element_bytes": self.element_bytes,
            "matrix_size": self.matrix_size,
        }


def _shapes(payload: object, label: str, maximum: int) -> tuple[MatmulShape, ...]:
    if type(payload) is not list or not 1 <= len(payload) <= maximum:
        raise MemoryAccessError(f"{label} must list 1 to {maximum} shapes")
    shapes = tuple(
        MatmulShape(**_fields(item, {"m", "n", "k"}, "shape")) for item in payload
    )
    if len(set(shapes)) != len(shapes):
        raise MemoryAccessError(f"{label} must not repeat a shape")
    return shapes


@dataclass(frozen=True)
class MatmulSweepSpec:
    """Shapes and tile widths for the tiled-matmul traffic sweep."""

    dtype: str
    shapes: tuple[MatmulShape, ...]
    tiles: tuple[int, ...]

    def __post_init__(self) -> None:
        matmul_element_bytes(self.dtype)
        if (
            type(self.shapes) is not tuple
            or not 1 <= len(self.shapes) <= MAX_SWEEP_SHAPES
            or any(not isinstance(shape, MatmulShape) for shape in self.shapes)
            or len(set(self.shapes)) != len(self.shapes)
        ):
            raise MemoryAccessError(
                f"sweep shapes must be 1 to {MAX_SWEEP_SHAPES} unique MatmulShapes"
            )
        if type(self.tiles) is not tuple or not 1 <= len(self.tiles) <= MAX_SWEEP_TILES:
            raise MemoryAccessError(f"sweep must list 1 to {MAX_SWEEP_TILES} tiles")
        for tile in self.tiles:
            _integer(tile, "tile", 1, MAX_MATMUL_TILE)
        if list(self.tiles) != sorted(set(self.tiles)):
            raise MemoryAccessError("sweep tiles must be unique and increasing")

    def to_mapping(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shapes": [shape.to_mapping() for shape in self.shapes],
            "tiles": list(self.tiles),
        }


@dataclass(frozen=True)
class DeclaredRoofline:
    """Peak ceilings declared from a cited public datasheet; never measured."""

    name: str
    peak_flops_per_second: int | float
    memory_bandwidth_bytes_per_second: int | float
    source_url: str
    source_note: str
    declared: bool = True

    def __post_init__(self) -> None:
        _name(self.name, "roofline name")
        if self.declared is not True:
            raise MemoryAccessError("roofline ceilings must be declared inputs")
        if (
            not isinstance(self.source_url, str)
            or not self.source_url.startswith("https://")
            or len(self.source_url) > 300
            or any(character.isspace() for character in self.source_url)
        ):
            raise MemoryAccessError("source_url must be one https URL")
        if (
            not isinstance(self.source_note, str)
            or not self.source_note
            or self.source_note != " ".join(self.source_note.split())
            or len(self.source_note) > MAX_SOURCE_TEXT
        ):
            raise MemoryAccessError("source_note must be canonical nonempty text")
        for name in ("peak_flops_per_second", "memory_bandwidth_bytes_per_second"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= MAX_CEILING:
                raise MemoryAccessError(f"{name} must be positive, finite and bounded")

    @property
    def hardware(self) -> RooflineHardware:
        return RooflineHardware(
            peak_flops_per_second=self.peak_flops_per_second,
            memory_bandwidth_bytes_per_second=self.memory_bandwidth_bytes_per_second,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "declared": self.declared,
            "peak_flops_per_second": self.peak_flops_per_second,
            "memory_bandwidth_bytes_per_second": self.memory_bandwidth_bytes_per_second,
            "source_url": self.source_url,
            "source_note": self.source_note,
        }


@dataclass(frozen=True)
class BenchmarkSpec:
    """A bounded CPU matmul measurement budget that finishes in seconds."""

    backends: tuple[str, ...]
    shapes: tuple[MatmulShape, ...]
    warmup: int
    repeats: int
    seed: int
    dtype: str = BASELINE_DTYPE

    def __post_init__(self) -> None:
        if (
            type(self.backends) is not tuple
            or not self.backends
            or len(set(self.backends)) != len(self.backends)
            or any(backend not in BACKENDS for backend in self.backends)
        ):
            raise MemoryAccessError(
                f"backends must be unique entries from {', '.join(BACKENDS)}"
            )
        if self.dtype != BASELINE_DTYPE:
            raise MemoryAccessError("benchmark dtype must be float32")
        if (
            type(self.shapes) is not tuple
            or not 1 <= len(self.shapes) <= MAX_BENCHMARK_SHAPES
            or any(not isinstance(shape, MatmulShape) for shape in self.shapes)
            or len(set(self.shapes)) != len(self.shapes)
        ):
            raise MemoryAccessError(
                f"benchmark shapes must be 1 to {MAX_BENCHMARK_SHAPES} unique shapes"
            )
        _integer(self.warmup, "warmup", 0, MAX_WARMUP)
        _integer(self.repeats, "repeats", 1, MAX_REPEATS)
        _integer(self.seed, "seed", 0, MAX_BASELINE_SEED)
        cases = self.cases()
        work = sum(case.shape.flops for case in cases) * (self.warmup + self.repeats)
        if work > MAX_BENCHMARK_FLOPS:
            raise MemoryAccessError("benchmark exceeds its 2**36 FLOP budget")

    def cases(self) -> tuple[MatmulCase, ...]:
        return tuple(
            MatmulCase(backend, shape)
            for backend in self.backends
            for shape in self.shapes
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "backends": list(self.backends),
            "dtype": self.dtype,
            "shapes": [shape.to_mapping() for shape in self.shapes],
            "warmup": self.warmup,
            "repeats": self.repeats,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MemoryAccessPlan:
    """Every declared input needed to rebuild both lab reports."""

    name: str
    geometry: MemoryGeometry
    patterns: tuple[PatternSpec, ...]
    transposes: tuple[TransposeSpec, ...]
    matmul: MatmulSweepSpec
    roofline: DeclaredRoofline
    benchmark: BenchmarkSpec

    def __post_init__(self) -> None:
        _name(self.name, "plan name")
        for value, kind in (
            (self.geometry, MemoryGeometry),
            (self.matmul, MatmulSweepSpec),
            (self.roofline, DeclaredRoofline),
            (self.benchmark, BenchmarkSpec),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"plan section must be {kind.__name__}")
        for items, kind, maximum, label in (
            (self.patterns, PatternSpec, MAX_PATTERNS, "patterns"),
            (self.transposes, TransposeSpec, MAX_TRANSPOSES, "transposes"),
        ):
            if (
                type(items) is not tuple
                or not 1 <= len(items) <= maximum
                or any(not isinstance(item, kind) for item in items)
            ):
                raise MemoryAccessError(f"{label} must hold 1 to {maximum} entries")
            _unique_names(items, label)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "name": self.name,
            "geometry": self.geometry.to_mapping(),
            "patterns": [pattern.to_mapping() for pattern in self.patterns],
            "transposes": [item.to_mapping() for item in self.transposes],
            "matmul": self.matmul.to_mapping(),
            "roofline": self.roofline.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every declared input to one SHA-256 digest."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _plan(payload: object) -> MemoryAccessPlan:
    plan = _fields(payload, _PLAN_FIELDS, "plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise MemoryAccessError("plan schema_version must be 1")
    for key, maximum in (("patterns", MAX_PATTERNS), ("transposes", MAX_TRANSPOSES)):
        if type(plan[key]) is not list:
            raise MemoryAccessError(f"{key} must be a list")
        if not 1 <= len(plan[key]) <= maximum:
            raise MemoryAccessError(f"{key} must hold 1 to {maximum} entries")
    matmul = _fields(plan["matmul"], {"dtype", "shapes", "tiles"}, "matmul")
    if type(matmul["tiles"]) is not list:
        raise MemoryAccessError("matmul tiles must be a list")
    roofline = _fields(
        plan["roofline"],
        {
            "name",
            "declared",
            "peak_flops_per_second",
            "memory_bandwidth_bytes_per_second",
            "source_url",
            "source_note",
        },
        "roofline",
    )
    benchmark = _fields(
        plan["benchmark"],
        {"backends", "dtype", "shapes", "warmup", "repeats", "seed"},
        "benchmark",
    )
    if type(benchmark["backends"]) is not list:
        raise MemoryAccessError("benchmark backends must be a list")
    geometry = _fields(plan["geometry"], set(MemoryGeometry().to_mapping()), "geometry")
    return MemoryAccessPlan(
        name=plan["name"],
        geometry=MemoryGeometry(**geometry),
        patterns=tuple(pattern_from_mapping(item) for item in plan["patterns"]),
        transposes=tuple(
            TransposeSpec(
                **_fields(
                    item,
                    {"name", "leading_dimension", "element_bytes", "matrix_size"},
                    "transpose",
                )
            )
            for item in plan["transposes"]
        ),
        matmul=MatmulSweepSpec(
            dtype=matmul["dtype"],
            shapes=_shapes(matmul["shapes"], "matmul shapes", MAX_SWEEP_SHAPES),
            tiles=tuple(matmul["tiles"]),
        ),
        roofline=DeclaredRoofline(**roofline),
        benchmark=BenchmarkSpec(
            backends=tuple(benchmark["backends"]),
            shapes=_shapes(
                benchmark["shapes"], "benchmark shapes", MAX_BENCHMARK_SHAPES
            ),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            seed=benchmark["seed"],
            dtype=benchmark["dtype"],
        ),
    )


def plan_from_mapping(payload: object) -> MemoryAccessPlan:
    """Parse a plan, reporting every contract violation as MemoryAccessError."""

    try:
        return _plan(payload)
    except TypeError as error:
        raise MemoryAccessError(f"invalid memory-access plan: {error}") from error
