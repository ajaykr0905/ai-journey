"""Strict plans for one kernel-launch lab run.

A plan declares every input of a run: the compute capability, the vector add
and grayscale extents with their block shapes and input seeds, the seeds of
the permuted launch orders, and the extents of the unguarded demos. Unknown
or missing fields are rejected at every level, booleans never pass as
integers, and every launch must pass the named compute-capability limits and
stay small enough for a quick pure-Python run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import (
    COMPUTE_CAPABILITY,
    UINT_MAX,
    KernelLaunchError,
    LaunchConfig,
    launch_1d,
    launch_2d,
    require_int,
)
from .kernel_launch_io import loads_strict, read_bounded
from .kernel_launch_reference import MAX_INPUT_SEED
from .kernel_launch_simulator import MAX_ORDER_SEED, LaunchOrder

SCHEMA_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024
MAX_PLAN_THREADS = 1 << 16
MAX_ORDER_SEEDS = 8
MAX_DEMOS = 8
DEMO_KERNELS = ("vector_add", "rgb_to_grayscale")
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_PLAN_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "compute_capability",
    "vector_add",
    "rgb_to_grayscale",
    "permuted_order_seeds",
    "unguarded_demos",
}


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise KernelLaunchError(f"{label} must be a JSON object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(str(key) for key in set(value) - expected)
        raise KernelLaunchError(
            f"{label} fields mismatch: missing={missing}, extra={extra}"
        )
    return value


def _bounded_launch(config: LaunchConfig, label: str) -> LaunchConfig:
    if config.launched_threads > MAX_PLAN_THREADS:
        raise KernelLaunchError(
            f"{label} launches {config.launched_threads} threads; a plan allows "
            f"at most {MAX_PLAN_THREADS}"
        )
    return config


def _block(value: object, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise KernelLaunchError(f"{label} must be a (block_x, block_y) pair")
    return value


@dataclass(frozen=True)
class VectorAddCase:
    """The 1D vector add: ``n`` float32 elements, one thread per element."""

    n: int
    threads_per_block: int
    input_seed: int

    def __post_init__(self) -> None:
        require_int(self.n, "vector_add.n", minimum=1, maximum=MAX_PLAN_THREADS)
        require_int(
            self.threads_per_block,
            "vector_add.threads_per_block",
            minimum=1,
            maximum=UINT_MAX,
        )
        require_int(
            self.input_seed, "vector_add.input_seed", minimum=0, maximum=MAX_INPUT_SEED
        )
        self.launch_for(self.n)

    def launch_for(self, n: int) -> LaunchConfig:
        """Return the bounded 1D launch of this block size over ``n`` elements."""

        return _bounded_launch(launch_1d(n, self.threads_per_block), "vector_add")

    def to_mapping(self) -> dict[str, object]:
        return {
            "n": self.n,
            "threads_per_block": self.threads_per_block,
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class GrayscaleCase:
    """The 2D grayscale conversion of a ``height x width x 3`` uint8 image."""

    width: int
    height: int
    block: tuple[int, int]
    input_seed: int

    def __post_init__(self) -> None:
        require_int(self.width, "rgb_to_grayscale.width", minimum=1, maximum=4096)
        require_int(self.height, "rgb_to_grayscale.height", minimum=1, maximum=4096)
        for value in _block(self.block, "rgb_to_grayscale.block"):
            require_int(value, "rgb_to_grayscale.block", minimum=1, maximum=UINT_MAX)
        require_int(
            self.input_seed,
            "rgb_to_grayscale.input_seed",
            minimum=0,
            maximum=MAX_INPUT_SEED,
        )
        self.launch_for(self.width, self.height)

    def launch_for(self, width: int, height: int) -> LaunchConfig:
        """Return the bounded 2D launch of this block shape over an image."""

        return _bounded_launch(
            launch_2d(width, height, *self.block), "rgb_to_grayscale"
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "block": list(self.block),
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class UnguardedDemo:
    """One unguarded launch: a kernel name and its 1D or 2D extent."""

    kernel: str
    extent: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.kernel) is not str or self.kernel not in DEMO_KERNELS:
            raise KernelLaunchError(f"demo kernel must be one of {DEMO_KERNELS}")
        axes = 1 if self.kernel == "vector_add" else 2
        if type(self.extent) is not tuple or len(self.extent) != axes:
            raise KernelLaunchError(f"a {self.kernel} demo extent has {axes} axes")
        for value in self.extent:
            require_int(value, "demo extent", minimum=1, maximum=MAX_PLAN_THREADS)

    def to_mapping(self) -> dict[str, object]:
        return {"kernel": self.kernel, "extent": list(self.extent)}


@dataclass(frozen=True)
class KernelLaunchPlan:
    """Every declared input of one simulated kernel-launch lab run."""

    name: str
    statement: str
    compute_capability: str
    vector_add: VectorAddCase
    rgb_to_grayscale: GrayscaleCase
    permuted_order_seeds: tuple[int, ...]
    unguarded_demos: tuple[UnguardedDemo, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise KernelLaunchError("plan name must be a lowercase slug")
        if (
            type(self.statement) is not str
            or not 1 <= len(self.statement) <= 600
            or self.statement != self.statement.strip()
            or not self.statement.isascii()
            or not self.statement.isprintable()
        ):
            raise KernelLaunchError("statement must be printable ASCII text")
        if self.compute_capability != COMPUTE_CAPABILITY:
            raise KernelLaunchError(
                f"compute_capability must be {COMPUTE_CAPABILITY!r}, the one "
                "whose limits the lab names"
            )
        if type(self.vector_add) is not VectorAddCase:
            raise TypeError("vector_add must be VectorAddCase")
        if type(self.rgb_to_grayscale) is not GrayscaleCase:
            raise TypeError("rgb_to_grayscale must be GrayscaleCase")
        seeds = self.permuted_order_seeds
        if type(seeds) is not tuple or not 1 <= len(seeds) <= MAX_ORDER_SEEDS:
            raise KernelLaunchError(
                f"permuted_order_seeds must hold 1 to {MAX_ORDER_SEEDS} seeds"
            )
        for seed in seeds:
            require_int(seed, "order seed", minimum=0, maximum=MAX_ORDER_SEED)
        if len(set(seeds)) != len(seeds):
            raise KernelLaunchError("permuted_order_seeds must not repeat a seed")
        demos = self.unguarded_demos
        if (
            type(demos) is not tuple
            or not 1 <= len(demos) <= MAX_DEMOS
            or any(type(demo) is not UnguardedDemo for demo in demos)
        ):
            raise KernelLaunchError(
                f"unguarded_demos must hold 1 to {MAX_DEMOS} UnguardedDemo entries"
            )
        if len(set(demos)) != len(demos):
            raise KernelLaunchError("unguarded_demos must not repeat a demo")
        for demo in demos:
            self.demo_launch(demo)

    @property
    def orders(self) -> tuple[LaunchOrder, ...]:
        """Return the canonical order, then one permuted order per seed."""

        return (
            LaunchOrder.canonical(),
            *(LaunchOrder.permuted(seed) for seed in self.permuted_order_seeds),
        )

    def demo_launch(self, demo: UnguardedDemo) -> LaunchConfig:
        """Size a demo with the block shape of the matching kernel case."""

        if demo.kernel == "vector_add":
            return self.vector_add.launch_for(*demo.extent)
        return self.rgb_to_grayscale.launch_for(*demo.extent)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "statement": self.statement,
            "compute_capability": self.compute_capability,
            "vector_add": self.vector_add.to_mapping(),
            "rgb_to_grayscale": self.rgb_to_grayscale.to_mapping(),
            "permuted_order_seeds": list(self.permuted_order_seeds),
            "unguarded_demos": [demo.to_mapping() for demo in self.unguarded_demos],
        }

    @classmethod
    def from_mapping(cls, mapping: object) -> KernelLaunchPlan:
        """Parse a plan mapping without ignoring, defaulting or coercing a field."""

        plan = _fields(mapping, _PLAN_FIELDS, "plan")
        version = plan["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise KernelLaunchError("unsupported plan schema_version")
        vector = _fields(
            plan["vector_add"], {"n", "threads_per_block", "input_seed"}, "vector_add"
        )
        gray = _fields(
            plan["rgb_to_grayscale"],
            {"width", "height", "block", "input_seed"},
            "rgb_to_grayscale",
        )
        demos = _array(plan["unguarded_demos"], "unguarded_demos")
        return cls(
            name=plan["name"],
            statement=plan["statement"],
            compute_capability=plan["compute_capability"],
            vector_add=VectorAddCase(**vector),
            rgb_to_grayscale=GrayscaleCase(
                width=gray["width"],
                height=gray["height"],
                block=_array(gray["block"], "rgb_to_grayscale.block"),
                input_seed=gray["input_seed"],
            ),
            permuted_order_seeds=_array(
                plan["permuted_order_seeds"], "permuted_order_seeds"
            ),
            unguarded_demos=tuple(
                UnguardedDemo(
                    kernel=item["kernel"],
                    extent=_array(item["extent"], "demo extent"),
                )
                for item in (
                    _fields(demo, {"kernel", "extent"}, "unguarded demo")
                    for demo in demos
                )
            ),
        )


def _array(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise KernelLaunchError(f"{label} must be a JSON array")
    return tuple(value)


def load_plan(path: Path) -> KernelLaunchPlan:
    """Load one bounded, duplicate-free, exact-field plan file."""

    payload = read_bounded(path, limit=MAX_PLAN_BYTES, label="plan")
    try:
        return KernelLaunchPlan.from_mapping(loads_strict(payload, label="plan"))
    except TypeError as error:
        raise KernelLaunchError(f"invalid plan: {error}") from error
