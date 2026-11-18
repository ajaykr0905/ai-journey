"""Strict pinned config for profiling the Day 73 launch simulator.

The config fixes everything a profiling snapshot depends on: three
simulator workloads at fixed sizes and seeds (the Day 73 vector add and
grayscale kernels and the Day 74 naive matmul), how many hotspots to name,
and the warmup and repeat counts of the wall-clock timing. It also lists
the optimization rounds in order, each with the function it targets and
the change it makes. Unknown or missing fields are rejected at every level,
and booleans never pass as integers.

A snapshot binds to ``workload_sha256``, the digest of every field except
the rounds. A round is added to the config after its "before" snapshot, so
the snapshots of earlier rounds still bind to the same workloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .kernel_launch_config import launch_1d, launch_2d, require_int
from .kernel_launch_io import canonical_sha256, loads_strict, read_bounded
from .kernel_launch_plan import _array, _fields
from .kernel_launch_reference import MAX_INPUT_SEED
from .matmul_model_kernels import MAPPINGS, matmul_launch
from .matmul_model_plan import MAX_CASE_INNER, MAX_CASE_MULTIPLY_ADDS
from .memory_access_tiling import MAX_MATMUL_DIMENSION, MatmulShape

SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_STATEMENT = 600
MAX_CHANGE = 400
# Each workload stays small enough that one profiled pass takes well under
# a second of pure Python on a laptop.
MAX_WORKLOAD_THREADS = 1 << 14
MAX_VECTOR_ELEMENTS = 1 << 14
MAX_IMAGE_PIXELS = 1 << 14
MAX_TOP_FUNCTIONS = 25
MAX_WARMUP = 10
MIN_REPEATS = 5
MAX_REPEATS = 50
MAX_ROUNDS = 8
BASELINE = "baseline"
VECTOR_ADD = "vector_add"
RGB_TO_GRAYSCALE = "rgb_to_grayscale"
NAIVE_MATMUL = "naive_matmul"
WORKLOADS = (VECTOR_ADD, RGB_TO_GRAYSCALE, NAIVE_MATMUL)
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
# A target names a function of this package as ``module:qualname``.
_TARGET = re.compile(r"ai_journey\.[a-z0-9_]+:[A-Za-z_][A-Za-z0-9_.<>]{0,127}")
_CONFIG_FIELDS = {
    "schema_version",
    "name",
    "statement",
    "workloads",
    "profile",
    "timing",
    "rounds",
}
_VECTOR_FIELDS = {"n", "threads_per_block", "input_seed"}
_GRAY_FIELDS = {"width", "height", "block", "input_seed"}
_MATMUL_FIELDS = {"m", "n", "k", "block", "mapping", "input_seed"}


class SimProfileError(ValueError):
    """Raised when a simulator-profiling input, run or report breaks its contract."""


class SimProfileVerificationError(SimProfileError):
    """Raised when a profiling report was read but fails its checks."""


def _text(value: object, label: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= maximum
        or value != value.strip()
        or not value.isascii()
        or not value.isprintable()
    ):
        raise SimProfileError(f"{label} must be printable ASCII text")
    return value


def _block(value: object, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise SimProfileError(f"{label} must be a (block_x, block_y) pair")
    return value


def _bounded_launch(threads: int, label: str) -> None:
    if threads > MAX_WORKLOAD_THREADS:
        raise SimProfileError(
            f"{label} launches {threads} threads; a workload allows at most "
            f"{MAX_WORKLOAD_THREADS}"
        )


@dataclass(frozen=True)
class VectorAddWorkload:
    """The Day 73 guarded vector add over ``n`` seeded float32 pairs."""

    n: int
    threads_per_block: int
    input_seed: int

    def __post_init__(self) -> None:
        require_int(self.n, "vector_add.n", minimum=1, maximum=MAX_VECTOR_ELEMENTS)
        require_int(self.input_seed, "input_seed", minimum=0, maximum=MAX_INPUT_SEED)
        threads = launch_1d(self.n, self.threads_per_block).launched_threads
        _bounded_launch(threads, VECTOR_ADD)

    def to_mapping(self) -> dict[str, object]:
        return {
            "n": self.n,
            "threads_per_block": self.threads_per_block,
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class GrayscaleWorkload:
    """The Day 73 guarded RGB-to-grayscale kernel over a seeded image."""

    width: int
    height: int
    block: tuple[int, int]
    input_seed: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            require_int(getattr(self, name), name, minimum=1, maximum=MAX_IMAGE_PIXELS)
        if self.width * self.height > MAX_IMAGE_PIXELS:
            raise SimProfileError(f"an image holds at most {MAX_IMAGE_PIXELS} pixels")
        require_int(self.input_seed, "input_seed", minimum=0, maximum=MAX_INPUT_SEED)
        block = _block(self.block, "rgb_to_grayscale.block")
        config = launch_2d(self.width, self.height, *block)
        _bounded_launch(config.launched_threads, RGB_TO_GRAYSCALE)

    def to_mapping(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "block": list(self.block),
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class MatmulWorkload:
    """The Day 74 naive matmul ``C = A @ B`` under one thread mapping."""

    shape: MatmulShape
    block: tuple[int, int]
    mapping: str
    input_seed: int

    def __post_init__(self) -> None:
        if type(self.shape) is not MatmulShape:
            raise TypeError("matmul shape must be MatmulShape")
        if self.shape.k > MAX_CASE_INNER:
            raise SimProfileError(
                f"naive_matmul.k must be at most {MAX_CASE_INNER}, the Day 74 limit"
            )
        products = self.shape.m * self.shape.n * self.shape.k
        if products > MAX_CASE_MULTIPLY_ADDS:
            raise SimProfileError(
                f"naive_matmul needs {products} multiply-adds; at most "
                f"{MAX_CASE_MULTIPLY_ADDS} are allowed"
            )
        if type(self.mapping) is not str or self.mapping not in MAPPINGS:
            raise SimProfileError(f"naive_matmul.mapping must be one of {MAPPINGS}")
        require_int(self.input_seed, "input_seed", minimum=0, maximum=MAX_INPUT_SEED)
        block = _block(self.block, "naive_matmul.block")
        config = matmul_launch(self.shape, block, self.mapping)
        _bounded_launch(config.launched_threads, NAIVE_MATMUL)

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.shape.to_mapping(),
            "block": list(self.block),
            "mapping": self.mapping,
            "input_seed": self.input_seed,
        }


@dataclass(frozen=True)
class RoundSpec:
    """One optimization round: the function it targets and the one change."""

    target: str
    change: str

    def __post_init__(self) -> None:
        if type(self.target) is not str or _TARGET.fullmatch(self.target) is None:
            raise SimProfileError(
                "a round target must name a package function as module:qualname"
            )
        _text(self.change, "round change", MAX_CHANGE)

    def to_mapping(self) -> dict[str, str]:
        return {"target": self.target, "change": self.change}


def stage_name(index: int) -> str:
    """Name stage ``index``: 0 is the baseline, then one stage per round."""

    require_int(index, "stage index", minimum=0, maximum=MAX_ROUNDS)
    return BASELINE if index == 0 else f"round-{index}"


@dataclass(frozen=True)
class SimProfileConfig:
    """Every declared input of a profiling snapshot, and the round plan."""

    name: str
    statement: str
    vector_add: VectorAddWorkload
    rgb_to_grayscale: GrayscaleWorkload
    naive_matmul: MatmulWorkload
    top_functions: int
    warmup: int
    repeats: int
    rounds: tuple[RoundSpec, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SLUG.fullmatch(self.name) is None:
            raise SimProfileError("config name must be a lowercase slug")
        _text(self.statement, "statement", MAX_STATEMENT)
        for field, kind in (
            ("vector_add", VectorAddWorkload),
            ("rgb_to_grayscale", GrayscaleWorkload),
            ("naive_matmul", MatmulWorkload),
        ):
            if type(getattr(self, field)) is not kind:
                raise TypeError(f"{field} must be {kind.__name__}")
        require_int(
            self.top_functions, "top_functions", minimum=1, maximum=MAX_TOP_FUNCTIONS
        )
        require_int(self.warmup, "warmup", minimum=0, maximum=MAX_WARMUP)
        require_int(self.repeats, "repeats", minimum=MIN_REPEATS, maximum=MAX_REPEATS)
        if type(self.rounds) is not tuple or len(self.rounds) > MAX_ROUNDS:
            raise SimProfileError(f"rounds must hold 0 to {MAX_ROUNDS} rounds")
        if any(type(item) is not RoundSpec for item in self.rounds):
            raise TypeError("rounds must be RoundSpec values")

    @property
    def stages(self) -> tuple[str, ...]:
        """Return the baseline stage, then one stage after each round."""

        return tuple(stage_name(index) for index in range(len(self.rounds) + 1))

    def workloads_mapping(self) -> dict[str, object]:
        return {
            VECTOR_ADD: self.vector_add.to_mapping(),
            RGB_TO_GRAYSCALE: self.rgb_to_grayscale.to_mapping(),
            NAIVE_MATMUL: self.naive_matmul.to_mapping(),
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "statement": self.statement,
            "workloads": self.workloads_mapping(),
            "profile": {"top_functions": self.top_functions},
            "timing": {"warmup": self.warmup, "repeats": self.repeats},
            "rounds": [item.to_mapping() for item in self.rounds],
        }

    def snapshot_mapping(self) -> dict[str, object]:
        """Return every field that a snapshot depends on: all but the rounds."""

        return {
            key: value for key, value in self.to_mapping().items() if key != "rounds"
        }

    @property
    def workload_sha256(self) -> str:
        return canonical_sha256(self.snapshot_mapping())

    @classmethod
    def from_mapping(cls, mapping: object) -> SimProfileConfig:
        """Parse a config mapping without ignoring, defaulting or coercing a field."""

        config = _fields(mapping, _CONFIG_FIELDS, "config")
        version = config["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise SimProfileError("unsupported config schema_version")
        workloads = _fields(config["workloads"], set(WORKLOADS), "workloads")
        vector = _fields(workloads[VECTOR_ADD], _VECTOR_FIELDS, VECTOR_ADD)
        gray = _fields(workloads[RGB_TO_GRAYSCALE], _GRAY_FIELDS, RGB_TO_GRAYSCALE)
        matmul = _fields(workloads[NAIVE_MATMUL], _MATMUL_FIELDS, NAIVE_MATMUL)
        for name in ("m", "n", "k"):
            require_int(
                matmul[name],
                f"naive_matmul.{name}",
                minimum=1,
                maximum=MAX_MATMUL_DIMENSION,
            )
        profile = _fields(config["profile"], {"top_functions"}, "profile")
        timing = _fields(config["timing"], {"warmup", "repeats"}, "timing")
        rounds = tuple(
            RoundSpec(**_fields(item, {"target", "change"}, "round"))
            for item in _array(config["rounds"], "rounds")
        )
        return cls(
            name=config["name"],
            statement=config["statement"],
            vector_add=VectorAddWorkload(**vector),
            rgb_to_grayscale=GrayscaleWorkload(
                width=gray["width"],
                height=gray["height"],
                block=_array(gray["block"], "rgb_to_grayscale.block"),
                input_seed=gray["input_seed"],
            ),
            naive_matmul=MatmulWorkload(
                shape=MatmulShape(m=matmul["m"], n=matmul["n"], k=matmul["k"]),
                block=_array(matmul["block"], "naive_matmul.block"),
                mapping=matmul["mapping"],
                input_seed=matmul["input_seed"],
            ),
            top_functions=profile["top_functions"],
            warmup=timing["warmup"],
            repeats=timing["repeats"],
            rounds=rounds,
        )


def load_config(path: Path) -> SimProfileConfig:
    """Load one bounded, duplicate-free, exact-field config file."""

    payload = read_bounded(path, limit=MAX_CONFIG_BYTES, label="config")
    try:
        return SimProfileConfig.from_mapping(loads_strict(payload, label="config"))
    except TypeError as error:
        raise SimProfileError(f"invalid config: {error}") from error
