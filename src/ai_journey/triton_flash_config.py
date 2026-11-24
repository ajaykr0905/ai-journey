"""Strict plans for one flash attention interpreter check, and the pinned plan.

A plan names the seed, the attention cases and the declared configs that a
check runs. Unknown or missing fields are refused at every level, and a bool
never passes as an integer. Each config must meet, in some case, an N that is
not a multiple of BLOCK_M and an N above BLOCK_N that is not a multiple of
BLOCK_N, so that the last query block and the last key block are short. A
plan must hold a causal case and a non-causal case.

The interpreter runs each program as Python code, so ``MAX_INTERPRETER_STEPS``
bounds programs times key blocks, summed over configs, cases and both input
families. The report stores every output value, so ``MAX_STORED_VALUES``
bounds ``BH * N * (D + 2)`` over the same runs.

``PINNED_PLAN`` is ``config/day-81-triton-flash.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_plan import _array, _fields
from .kernel_metal_reference import MAX_SEED
from .triton_flash_kernel import (
    CONFIGS_BY_NAME,
    FLASH_CONFIGS,
    FlashConfig,
    TritonFlashError,
    _integer,
    key_blocks,
    query_blocks,
)
from .triton_flash_reference import FAMILIES, MAX_CASE_INDEX, AttentionCase

SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024
MAX_CASES = min(16, MAX_CASE_INDEX + 1)
MAX_INTERPRETER_STEPS = 1 << 14
MAX_STORED_VALUES = 1 << 18
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PLAN_FIELDS = {"schema_version", "name", "seed", "cases", "configs"}
_CASE_FIELDS = {"batch_heads", "sequence", "head_dim", "causal"}


def interpreter_steps(
    cases: tuple[AttentionCase, ...], configs: tuple[FlashConfig, ...]
) -> int:
    """Return programs times key blocks over every config, case and family."""

    return len(FAMILIES) * sum(
        case.batch_heads
        * query_blocks(case.sequence, config)
        * key_blocks(case.sequence, config)
        for config in configs
        for case in cases
    )


def stored_values(
    cases: tuple[AttentionCase, ...], configs: tuple[FlashConfig, ...]
) -> int:
    """Return the float32 values of O, m and l that a report stores."""

    per_config = sum(
        case.batch_heads * case.sequence * (case.head_dim + 2) for case in cases
    )
    return len(FAMILIES) * len(configs) * per_config


def _ragged(cases: tuple[AttentionCase, ...], config: FlashConfig) -> bool:
    return any(case.sequence % config.block_m for case in cases) and any(
        case.sequence > config.block_n and case.sequence % config.block_n
        for case in cases
    )


@dataclass(frozen=True)
class TritonFlashPlan:
    """Every declared input of one interpreter check."""

    name: str
    seed: int
    cases: tuple[AttentionCase, ...]
    configs: tuple[FlashConfig, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise TritonFlashError("plan name must be a lowercase slug")
        _integer(self.seed, "seed", 0, MAX_SEED)
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES:
            raise TritonFlashError(f"cases must hold 1 to {MAX_CASES} cases")
        for case in self.cases:
            if type(case) is not AttentionCase:
                raise TypeError("each case must be AttentionCase")
        if len(set(self.cases)) != len(self.cases):
            raise TritonFlashError("cases must not repeat a case")
        if {case.causal for case in self.cases} != {True, False}:
            raise TritonFlashError("a plan needs a causal and a non-causal case")
        if type(self.configs) is not tuple or not self.configs:
            raise TritonFlashError("configs must name at least one config")
        if any(config not in FLASH_CONFIGS for config in self.configs) or list(
            self.configs
        ) != sorted(set(self.configs), key=FLASH_CONFIGS.index):
            raise TritonFlashError(
                "configs must be declared configs, in their declared order, "
                "with no repeat"
            )
        for config in self.configs:
            if not _ragged(self.cases, config):
                raise TritonFlashError(
                    f"{config.name} needs a short last query block and a short "
                    "last key block among the cases"
                )
        steps = interpreter_steps(self.cases, self.configs)
        if steps > MAX_INTERPRETER_STEPS:
            raise TritonFlashError(
                f"the plan needs {steps} interpreter steps; a check allows at "
                f"most {MAX_INTERPRETER_STEPS}"
            )
        values = stored_values(self.cases, self.configs)
        if values > MAX_STORED_VALUES:
            raise TritonFlashError(
                f"the plan stores {values} output values; a report holds at "
                f"most {MAX_STORED_VALUES}"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "cases": [case.to_mapping() for case in self.cases],
            "configs": [config.name for config in self.configs],
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> TritonFlashPlan:
        """Parse a plan mapping without ignoring, defaulting or coercing a field."""

        try:
            plan = _fields(mapping, _PLAN_FIELDS, "plan")
            cases = [
                _fields(item, _CASE_FIELDS, "case")
                for item in _array(plan["cases"], "cases")
            ]
            names = _array(plan["configs"], "configs")
        except KernelLaunchError as error:
            raise TritonFlashError(str(error)) from error
        version = plan["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise TritonFlashError("unsupported plan schema_version")
        for name in names:
            if type(name) is not str or name not in CONFIGS_BY_NAME:
                raise TritonFlashError(f"unknown config {name!r}")
        return cls(
            name=plan["name"],
            seed=plan["seed"],
            cases=tuple(
                AttentionCase(
                    batch_heads=item["batch_heads"],
                    sequence=item["sequence"],
                    head_dim=item["head_dim"],
                    causal=item["causal"],
                )
                for item in cases
            ),
            configs=tuple(CONFIGS_BY_NAME[name] for name in names),
        )


def load_plan(path: Path) -> TritonFlashPlan:
    """Load one bounded, duplicate-free, exact-field plan file."""

    try:
        payload = read_bounded(path, limit=MAX_PLAN_BYTES, label="plan")
        mapping = loads_strict(payload, label="plan")
    except KernelLaunchError as error:
        raise TritonFlashError(str(error)) from error
    try:
        return TritonFlashPlan.from_mapping(mapping)
    except TypeError as error:
        raise TritonFlashError(f"invalid plan: {error}") from error


PINNED_PLAN = TritonFlashPlan(
    name="day-81-triton-flash",
    seed=81,
    cases=(
        AttentionCase(1, 1, 16, False),
        AttentionCase(2, 17, 16, True),
        AttentionCase(2, 70, 32, False),
        AttentionCase(1, 129, 64, True),
        AttentionCase(1, 77, 64, False),
        AttentionCase(1, 40, 128, True),
    ),
    configs=FLASH_CONFIGS,
)
