"""Strict plans for one matmul model lab run.

A plan declares every input of a run: the compute capability, one 2D block
for every launch, the simulated cases with their extents and input seeds, the
seeds of the permuted launch orders, the extra shapes that only the FLOP and
byte accounting uses, the register counts of the occupancy estimate, and the
labeled ceiling pairs of the roofline. Unknown or missing fields are rejected
at every level, booleans never pass as integers, every launch must pass the
Day 73 compute capability 8.0 limits and the warp model's ``block_x``
rule, and each simulated case stays small enough for a quick pure-Python run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import COMPUTE_CAPABILITY, require_int
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_plan import _array, _fields
from .kernel_launch_reference import MAX_INPUT_SEED
from .kernel_launch_simulator import MAX_ORDER_SEED, LaunchOrder
from .matmul_model_access import MatmulLaunch
from .matmul_model_kernels import MAPPINGS
from .matmul_model_occupancy import ARCHITECTURE, estimate_occupancy
from .matmul_model_reference import MatmulModelError
from .matmul_model_roofline import Ceilings
from .memory_access_tiling import MAX_MATMUL_DIMENSION, MatmulShape

SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024
MAX_CASES = 4
MAX_CASE_THREADS = 1 << 14
MAX_CASE_MULTIPLY_ADDS = 1 << 17
MAX_ORDER_SEEDS = 4
MAX_ACCOUNTING_SHAPES = 4
MAX_REGISTER_COUNTS = 6
MAX_CEILINGS = 4
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PLAN_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "compute_capability",
    "block",
    "cases",
    "permuted_order_seeds",
    "accounting_shapes",
    "registers_per_thread",
    "ceilings",
}
_SHAPE_FIELDS = {"m", "n", "k"}
_CASE_FIELDS = {"name", *_SHAPE_FIELDS, "input_seed"}


def _shape(mapping: object, label: str) -> MatmulShape:
    fields = _fields(mapping, _SHAPE_FIELDS, label)
    for name in ("m", "n", "k"):
        require_int(
            fields[name], f"{label}.{name}", minimum=1, maximum=MAX_MATMUL_DIMENSION
        )
    return MatmulShape(m=fields["m"], n=fields["n"], k=fields["k"])


def _bounded_tuple(value: object, label: str, maximum: int) -> tuple[object, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= maximum:
        raise MatmulModelError(f"{label} must hold 1 to {maximum} entries")
    if len(set(value)) != len(value):
        raise MatmulModelError(f"{label} must not repeat an entry")
    return value


@dataclass(frozen=True)
class MatmulCase:
    """One simulated case: a name, the extents of ``C = A @ B`` and a seed."""

    name: str
    shape: MatmulShape
    input_seed: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise MatmulModelError("case name must be a lowercase slug")
        if type(self.shape) is not MatmulShape:
            raise TypeError("case shape must be MatmulShape")
        require_int(self.input_seed, "input_seed", minimum=0, maximum=MAX_INPUT_SEED)
        products = self.shape.m * self.shape.n * self.shape.k
        if products > MAX_CASE_MULTIPLY_ADDS:
            raise MatmulModelError(
                f"case {self.name} needs {products} multiply-adds; a simulated "
                f"case allows at most {MAX_CASE_MULTIPLY_ADDS}"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            **self.shape.to_mapping(),
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class MatmulModelPlan:
    """Every declared input of one matmul model lab run."""

    name: str
    statement: str
    compute_capability: str
    block: tuple[int, int]
    cases: tuple[MatmulCase, ...]
    permuted_order_seeds: tuple[int, ...]
    accounting_shapes: tuple[MatmulShape, ...]
    registers_per_thread: tuple[int, ...]
    ceilings: tuple[Ceilings, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise MatmulModelError("plan name must be a lowercase slug")
        if (
            type(self.statement) is not str
            or not 1 <= len(self.statement) <= 600
            or self.statement != self.statement.strip()
            or not self.statement.isascii()
            or not self.statement.isprintable()
        ):
            raise MatmulModelError("statement must be printable ASCII text")
        if self.compute_capability != COMPUTE_CAPABILITY:
            raise MatmulModelError(
                f"compute_capability must be {COMPUTE_CAPABILITY!r}, the one "
                "whose limits the lab names"
            )
        if type(self.block) is not tuple or len(self.block) != 2:
            raise MatmulModelError("block must be a (block_x, block_y) pair")
        cases = _bounded_tuple(self.cases, "cases", MAX_CASES)
        if any(type(case) is not MatmulCase for case in cases):
            raise TypeError("cases must be MatmulCase values")
        if len({case.name for case in cases}) != len(cases):
            raise MatmulModelError("case names must be unique")
        for case in cases:
            for mapping in MAPPINGS:
                launched = self.launch(case, mapping).config.launched_threads
                if launched > MAX_CASE_THREADS:
                    raise MatmulModelError(
                        f"case {case.name} launches {launched} threads with "
                        f"{mapping}; a case allows at most {MAX_CASE_THREADS}"
                    )
        seeds = _bounded_tuple(
            self.permuted_order_seeds, "permuted_order_seeds", MAX_ORDER_SEEDS
        )
        for seed in seeds:
            require_int(seed, "order seed", minimum=0, maximum=MAX_ORDER_SEED)
        shapes = self.accounting_shapes
        if type(shapes) is not tuple or len(shapes) > MAX_ACCOUNTING_SHAPES:
            raise MatmulModelError(
                f"accounting_shapes must hold 0 to {MAX_ACCOUNTING_SHAPES} shapes"
            )
        if any(type(shape) is not MatmulShape for shape in shapes):
            raise TypeError("accounting_shapes must be MatmulShape values")
        if len(set(shapes)) != len(shapes):
            raise MatmulModelError("accounting_shapes must not repeat a shape")
        registers = _bounded_tuple(
            self.registers_per_thread, "registers_per_thread", MAX_REGISTER_COUNTS
        )
        for count in registers:
            require_int(
                count,
                "registers_per_thread",
                minimum=1,
                maximum=ARCHITECTURE.max_registers_per_thread,
            )
        if list(registers) != sorted(registers):
            raise MatmulModelError("registers_per_thread must increase")
        for count in registers:
            estimate_occupancy(
                threads_per_block=self.threads_per_block, registers_per_thread=count
            )
        ceilings = _bounded_tuple(self.ceilings, "ceilings", MAX_CEILINGS)
        if any(type(item) is not Ceilings for item in ceilings):
            raise TypeError("ceilings must be Ceilings values")
        if len({item.label for item in ceilings}) != len(ceilings):
            raise MatmulModelError("ceiling labels must be unique")

    @property
    def threads_per_block(self) -> int:
        return self.block[0] * self.block[1]

    @property
    def orders(self) -> tuple[LaunchOrder, ...]:
        """Return the canonical order, then one permuted order per seed."""

        return (
            LaunchOrder.canonical(),
            *(LaunchOrder.permuted(seed) for seed in self.permuted_order_seeds),
        )

    def launch(self, case: MatmulCase, mapping: str) -> MatmulLaunch:
        """Return the warp-model launch of one case under one thread mapping."""

        return MatmulLaunch(case.shape, self.block, mapping)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "statement": self.statement,
            "compute_capability": self.compute_capability,
            "block": list(self.block),
            "cases": [case.to_mapping() for case in self.cases],
            "permuted_order_seeds": list(self.permuted_order_seeds),
            "accounting_shapes": [
                shape.to_mapping() for shape in self.accounting_shapes
            ],
            "registers_per_thread": list(self.registers_per_thread),
            "ceilings": [item.to_mapping() for item in self.ceilings],
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> MatmulModelPlan:
        """Parse a plan mapping without ignoring, defaulting or coercing a field."""

        plan = _fields(mapping, _PLAN_FIELDS, "plan")
        version = plan["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise MatmulModelError("unsupported plan schema_version")
        cases = []
        for item in _array(plan["cases"], "cases"):
            case = _fields(item, _CASE_FIELDS, "case")
            cases.append(
                MatmulCase(
                    name=case["name"],
                    shape=_shape({name: case[name] for name in _SHAPE_FIELDS}, "case"),
                    input_seed=case["input_seed"],
                )
            )
        return cls(
            name=plan["name"],
            statement=plan["statement"],
            compute_capability=plan["compute_capability"],
            block=_array(plan["block"], "block"),
            cases=tuple(cases),
            permuted_order_seeds=_array(
                plan["permuted_order_seeds"], "permuted_order_seeds"
            ),
            accounting_shapes=tuple(
                _shape(item, "accounting shape")
                for item in _array(plan["accounting_shapes"], "accounting_shapes")
            ),
            registers_per_thread=_array(
                plan["registers_per_thread"], "registers_per_thread"
            ),
            ceilings=tuple(
                Ceilings.from_mapping(item)
                for item in _array(plan["ceilings"], "ceilings")
            ),
        )


def load_plan(path: Path) -> MatmulModelPlan:
    """Load one bounded, duplicate-free, exact-field plan file."""

    payload = read_bounded(path, limit=MAX_PLAN_BYTES, label="plan")
    try:
        return MatmulModelPlan.from_mapping(loads_strict(payload, label="plan"))
    except TypeError as error:
        raise MatmulModelError(f"invalid plan: {error}") from error
