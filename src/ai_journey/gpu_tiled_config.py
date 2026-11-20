"""Strict plans for one tiled CUDA contract run, and the pinned Day 76 plan.

A plan declares every input of a run: the compute capability whose limits
the lab names, the tile widths, the seed and shapes of the correctness
cases, and the register counts of the occupancy estimate. Unknown or missing
fields are rejected at every level, and booleans never pass as integers.
Case ``i`` draws its operands as the Day 74 Metal lab draws its case ``i``,
with ``matmul_inputs(seed, CORRECTNESS_STREAM, i, shape)``. Thus, with the
Day 74 seed and shapes, a replay digest here names the bits that a Day 74
report names for the same case.

``PINNED_PLAN`` is the Day 76 plan: the nine correctness shapes of
``config/day-74-matmul-metal.json`` in its order, its seed 74, all three
tiles, and 32, 40 and 64 registers per thread. Every register count must
leave a block resident at every tile. A run simulates every thread of every
launch for every k and replays every case twice, so ``MAX_SIMULATED_STEPS``
bounds that work: launched threads times k for each tile and case, plus two
times m * n * k for each case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .gpu_tiled_cuda import TILES, GpuTiledError, require_shape, require_tile
from .gpu_tiled_occupancy import estimate_occupancy
from .kernel_launch_config import COMPUTE_CAPABILITY
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_plan import _array, _fields
from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import MAX_SEED
from .matmul_metal_reference import MAX_CASE_INDEX
from .matmul_model_kernels import X_COLUMNS, matmul_launch
from .memory_access_tiling import MatmulShape

SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024
# Case i reads operand stream i, and the Day 74 lab has MAX_CASE_INDEX + 1.
MAX_CASES = min(16, MAX_CASE_INDEX + 1)
MAX_REGISTER_COUNTS = 6
MAX_REGISTERS_PER_THREAD = 255
MAX_SIMULATED_STEPS = 1 << 30
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PLAN_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "compute_capability",
    "seed",
    "tiles",
    "shapes",
    "registers_per_thread",
}


def _increasing(values: object, label: str, maximum: int) -> tuple[object, ...]:
    if type(values) is not tuple or not 1 <= len(values) <= maximum:
        raise GpuTiledError(f"{label} must hold 1 to {maximum} entries")
    for value in values:
        if type(value) is not int:
            raise TypeError(f"{label} must hold integers")
    if list(values) != sorted(set(values)):
        raise GpuTiledError(f"{label} must increase without a repeat")
    return values


def simulated_steps(tiles: tuple[int, ...], shapes: tuple[MatmulShape, ...]) -> int:
    """Return launched threads times k over every tile and case, plus replays."""

    launched = sum(
        matmul_launch(shape, (tile, tile), X_COLUMNS).launched_threads * shape.k
        for tile in tiles
        for shape in shapes
    )
    return launched + 2 * sum(shape.m * shape.n * shape.k for shape in shapes)


@dataclass(frozen=True)
class GpuTiledPlan:
    """Every declared input of one tiled CUDA contract run."""

    name: str
    statement: str
    compute_capability: str
    seed: int
    tiles: tuple[int, ...]
    shapes: tuple[MatmulShape, ...]
    registers_per_thread: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise GpuTiledError("plan name must be a lowercase slug")
        if (
            type(self.statement) is not str
            or not 1 <= len(self.statement) <= 600
            or self.statement != self.statement.strip()
            or not self.statement.isascii()
            or not self.statement.isprintable()
        ):
            raise GpuTiledError("statement must be printable ASCII text")
        if self.compute_capability != COMPUTE_CAPABILITY:
            raise GpuTiledError(
                f"compute_capability must be {COMPUTE_CAPABILITY!r}, the one "
                "whose limits the lab names"
            )
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for tile in _increasing(self.tiles, "tiles", len(TILES)):
            require_tile(tile)
        shapes = self.shapes
        if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_CASES:
            raise GpuTiledError(f"shapes must hold 1 to {MAX_CASES} shapes")
        for shape in shapes:
            require_shape(shape)
        if len(set(shapes)) != len(shapes):
            raise GpuTiledError("shapes must not repeat a shape")
        registers = _increasing(
            self.registers_per_thread, "registers_per_thread", MAX_REGISTER_COUNTS
        )
        for count in registers:
            require_integer(count, "registers_per_thread", 1, MAX_REGISTERS_PER_THREAD)
            for tile in self.tiles:
                estimate_occupancy(tile=tile, registers_per_thread=count)
        steps = simulated_steps(self.tiles, shapes)
        if steps > MAX_SIMULATED_STEPS:
            raise GpuTiledError(
                f"the plan needs {steps} simulated steps; a run allows at most "
                f"{MAX_SIMULATED_STEPS}"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "statement": self.statement,
            "compute_capability": self.compute_capability,
            "seed": self.seed,
            "tiles": list(self.tiles),
            "shapes": [[shape.m, shape.n, shape.k] for shape in self.shapes],
            "registers_per_thread": list(self.registers_per_thread),
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> GpuTiledPlan:
        """Parse a plan mapping without ignoring, defaulting or coercing a field."""

        plan = _fields(mapping, _PLAN_FIELDS, "plan")
        version = plan["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise GpuTiledError("unsupported plan schema_version")
        shapes = []
        for item in _array(plan["shapes"], "shapes"):
            triple = _array(item, "shape")
            if len(triple) != 3:
                raise GpuTiledError("each shape must be an [m, n, k] triple")
            shapes.append(MatmulShape(*triple))
        return cls(
            name=plan["name"],
            statement=plan["statement"],
            compute_capability=plan["compute_capability"],
            seed=plan["seed"],
            tiles=_array(plan["tiles"], "tiles"),
            shapes=tuple(shapes),
            registers_per_thread=_array(
                plan["registers_per_thread"], "registers_per_thread"
            ),
        )


def load_plan(path: Path) -> GpuTiledPlan:
    """Load one bounded, duplicate-free, exact-field plan file."""

    payload = read_bounded(path, limit=MAX_PLAN_BYTES, label="plan")
    try:
        return GpuTiledPlan.from_mapping(loads_strict(payload, label="plan"))
    except TypeError as error:
        raise GpuTiledError(f"invalid plan: {error}") from error


PINNED_PLAN = GpuTiledPlan(
    name="day-76-gpu-tiled-cuda",
    statement=(
        "Inputs for a CPU contract of the Day 76 tiled CUDA matmul and of a "
        "row-major cuBLAS SGEMM call under compute capability 8.0 limits. The "
        "shapes and seed are the Day 74 correctness cases. The register counts "
        "are assumptions. No GPU runs, no CUDA or cuBLAS text is compiled or "
        "called, and no time is measured."
    ),
    compute_capability=COMPUTE_CAPABILITY,
    seed=74,
    tiles=TILES,
    shapes=tuple(
        MatmulShape(*triple)
        for triple in (
            (1, 1, 1),
            (17, 33, 9),
            (32, 32, 16),
            (31, 47, 129),
            (129, 65, 1000),
            (255, 257, 511),
            (64, 48, 4096),
            (3, 1000, 2048),
            (1000, 999, 100),
        )
    ),
    registers_per_thread=(32, 40, 64),
)
