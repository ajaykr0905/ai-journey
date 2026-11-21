"""Strict plans for one Triton interpreter check, and the pinned Day 79 plan.

A plan names the seed, the correctness shapes and the declared configs that a
check runs. Unknown or missing fields are refused at every level, and a bool
never passes as an integer. Case ``i`` draws its normal operands as the Day 74
Metal lab draws its case ``i``, so with seed 74 they are the operands of the
Day 76 CUDA contract cases, and a later GPU run can give both kernels the same
inputs.

Each config of a plan must meet a ragged edge on each axis in some case: an M
that is not a multiple of BLOCK_M, an N that is not a multiple of BLOCK_N,
and a K above BLOCK_K that is not a multiple of it, so that a later K block is
short. The interpreter runs each program as Python code, so
``MAX_INTERPRETER_STEPS`` bounds the work of a check: programs times K blocks,
summed over configs, cases and both operand families.

``PINNED_PLAN`` is ``config/day-79-triton-matmul.json``: the nine Day 74
correctness shapes in their order, seed 74 and all six declared configs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_plan import _array, _fields
from .kernel_metal_reference import MAX_SEED
from .matmul_metal_mapping import MAX_DIMENSION
from .matmul_metal_reference import MAX_CASE_INDEX
from .memory_access_tiling import MatmulShape
from .triton_matmul_correctness import FAMILIES
from .triton_matmul_kernel import (
    AUTOTUNE_CONFIGS,
    CONFIGS_BY_NAME,
    TritonConfig,
    TritonMatmulError,
    _integer,
    k_blocks,
    programs,
    require_shape,
)

SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024
MAX_CASES = min(16, MAX_CASE_INDEX + 1)
MAX_INTERPRETER_STEPS = 1 << 17
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PLAN_FIELDS = {"schema_version", "name", "seed", "correctness", "configs"}


def interpreter_steps(
    shapes: tuple[MatmulShape, ...], configs: tuple[TritonConfig, ...]
) -> int:
    """Return programs times K blocks over every config, shape and family."""

    return len(FAMILIES) * sum(
        programs(shape, config) * k_blocks(shape, config)
        for config in configs
        for shape in shapes
    )


def _ragged(shapes: tuple[MatmulShape, ...], config: TritonConfig) -> bool:
    return (
        any(shape.m % config.block_m for shape in shapes)
        and any(shape.n % config.block_n for shape in shapes)
        and any(
            shape.k > config.block_k and shape.k % config.block_k for shape in shapes
        )
    )


@dataclass(frozen=True)
class TritonMatmulPlan:
    """Every declared input of one interpreter check."""

    name: str
    seed: int
    shapes: tuple[MatmulShape, ...]
    configs: tuple[TritonConfig, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise TritonMatmulError("plan name must be a lowercase slug")
        _integer(self.seed, "seed", 0, MAX_SEED)
        if type(self.shapes) is not tuple or not 1 <= len(self.shapes) <= MAX_CASES:
            raise TritonMatmulError(f"shapes must hold 1 to {MAX_CASES} shapes")
        for shape in self.shapes:
            require_shape(shape)
        if len(set(self.shapes)) != len(self.shapes):
            raise TritonMatmulError("shapes must not repeat a shape")
        if type(self.configs) is not tuple or not self.configs:
            raise TritonMatmulError("configs must name at least one config")
        if any(config not in AUTOTUNE_CONFIGS for config in self.configs) or list(
            self.configs
        ) != sorted(set(self.configs), key=AUTOTUNE_CONFIGS.index):
            raise TritonMatmulError(
                "configs must be declared configs, in their declared order, "
                "with no repeat"
            )
        for config in self.configs:
            if not _ragged(self.shapes, config):
                raise TritonMatmulError(
                    f"{config.name} needs a ragged M, a ragged N and a short "
                    "later K block among the shapes"
                )
        steps = interpreter_steps(self.shapes, self.configs)
        if steps > MAX_INTERPRETER_STEPS:
            raise TritonMatmulError(
                f"the plan needs {steps} interpreter steps; a check allows at "
                f"most {MAX_INTERPRETER_STEPS}"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "correctness": {
                "shapes": [[shape.m, shape.n, shape.k] for shape in self.shapes]
            },
            "configs": [config.name for config in self.configs],
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> TritonMatmulPlan:
        """Parse a plan mapping without ignoring, defaulting or coercing a field."""

        try:
            plan = _fields(mapping, _PLAN_FIELDS, "plan")
            correctness = _fields(plan["correctness"], {"shapes"}, "correctness")
            triples = [
                _array(item, "shape")
                for item in _array(correctness["shapes"], "shapes")
            ]
            names = _array(plan["configs"], "configs")
        except KernelLaunchError as error:
            raise TritonMatmulError(str(error)) from error
        version = plan["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise TritonMatmulError("unsupported plan schema_version")
        shapes = []
        for triple in triples:
            if len(triple) != 3:
                raise TritonMatmulError("each shape must be an [m, n, k] triple")
            shapes.append(
                MatmulShape(
                    *(
                        _integer(value, axis, 1, MAX_DIMENSION)
                        for value, axis in zip(triple, "mnk")
                    )
                )
            )
        for name in names:
            if type(name) is not str or name not in CONFIGS_BY_NAME:
                raise TritonMatmulError(f"unknown config {name!r}")
        return cls(
            name=plan["name"],
            seed=plan["seed"],
            shapes=tuple(shapes),
            configs=tuple(CONFIGS_BY_NAME[name] for name in names),
        )


def load_plan(path: Path) -> TritonMatmulPlan:
    """Load one bounded, duplicate-free, exact-field plan file."""

    try:
        payload = read_bounded(path, limit=MAX_PLAN_BYTES, label="plan")
        mapping = loads_strict(payload, label="plan")
    except KernelLaunchError as error:
        raise TritonMatmulError(str(error)) from error
    try:
        return TritonMatmulPlan.from_mapping(mapping)
    except TypeError as error:
        raise TritonMatmulError(f"invalid plan: {error}") from error


PINNED_PLAN = TritonMatmulPlan(
    name="day-79-triton-matmul",
    seed=74,
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
    configs=AUTOTUNE_CONFIGS,
)
