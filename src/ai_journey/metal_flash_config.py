"""Strict, bounded controls for one Day 81 correctness, memory and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases, the timing shapes with the Day 74 controls, the attention modes and
the smallest ratio of median times that calls one side faster. A config
made for other hypotheses names another plan digest and is refused.

A correctness case is ``{"sequences", "length", "inputs"}`` for a dense
batch or ``{"lengths", "inputs"}`` for a ragged one, laid out with
``CORRECTNESS_GAP`` guard rows after each sequence. Every case runs N0 and
F1 in both modes. A timing shape is a dense ``{"sequences", "length"}``
batch, because the PyTorch call takes a B x H x N x 64 tensor, and it runs
once in each mode of the config. Before anything runs, the config is
checked as a whole:

- The cases hold a sequence of 1 token, a sequence shorter than a query
  block of 32, a sequence longer than 32 that is not a multiple of 32, a
  sequence of 1000 or more tokens, a ragged case and a ``wide_scores`` case.
- The repeats are a multiple of the three timed sides, so each side takes
  each position of the rotation equally often.
- The cases, the calls and the bytes fit their bounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_benchmark import (
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)
from .kernel_metal_reference import MAX_SEED
from .matmul_tiled_config import ComparisonControls
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .metal_flash_plan import SIDES, plan_sha256
from .metal_flash_reference import (
    CORRECTNESS_GAP,
    DISTRIBUTIONS,
    MAX_LENGTH,
    WIDE_SCORES,
    FlashLayout,
    dense_layout,
)
from .metal_flash_source import (
    BLOCK_QUERIES,
    FLOAT32_BYTES,
    HEAD_DIM,
    MODES,
    MetalFlashError,
    require_size,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 8
MAX_SHAPES = 6
MAX_CORRECTNESS_PAIRS = 1 << 24
MAX_RUN_CALLS = 1 << 16
MAX_RUN_BYTES = 1 << 42
MIN_CHECK_REPEATS = 2
MAX_CHECK_REPEATS = 4
MIN_CHECK_ROWS = 2
MAX_CHECK_ROWS = 256
LONG_SEQUENCE = 1000
COMPULSORY = (
    "16 * 64 bytes for each token: read its rows of Q, K and V once and write "
    "its row of O once, as float32"
)

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "plan_sha256",
    "correctness",
    "benchmark",
    "comparison",
}
_BENCHMARK_FIELDS = {
    "inputs",
    "shapes",
    "modes",
    "check_rows",
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MetalFlashError(f"{label} fields do not match the schema")
    return payload


def _distribution(value: object) -> str:
    if value not in DISTRIBUTIONS:
        raise MetalFlashError(
            f"inputs must be one of {', '.join(DISTRIBUTIONS)}, not {value!r}"
        )
    return value


def compulsory_bytes(layout: FlashLayout) -> int:
    """Bytes that any attention forward must move for a packed layout."""

    return 4 * FLOAT32_BYTES * HEAD_DIM * layout.tokens


@dataclass(frozen=True)
class FlashCase:
    """One correctness batch: its sequence lengths and its input distribution.

    ``length`` is set for a dense case and ``None`` for a ragged one, so the
    mapping keeps the form that the config file used.
    """

    lengths: tuple[int, ...]
    inputs: str
    length: int | None = None

    def __post_init__(self) -> None:
        _distribution(self.inputs)
        layout = self.layout()
        if self.length is not None and set(self.lengths) != {self.length}:
            raise MetalFlashError("a dense case holds sequences of its length only")
        if self.length is None and layout.dense:
            raise MetalFlashError("a ragged case needs two or more lengths")

    def layout(self) -> FlashLayout:
        """The gapped layout the case runs with."""

        return FlashLayout(self.lengths, CORRECTNESS_GAP)

    @property
    def ragged(self) -> bool:
        """Whether the sequences of the case have more than one length."""

        return self.length is None

    def to_mapping(self) -> dict[str, object]:
        """Return the case as the config file spells it."""

        if self.length is not None:
            return {
                "sequences": len(self.lengths),
                "length": self.length,
                "inputs": self.inputs,
            }
        return {"lengths": list(self.lengths), "inputs": self.inputs}


@dataclass(frozen=True)
class FlashShape:
    """One timing shape: ``sequences`` (batch times heads) of ``length`` tokens."""

    sequences: int
    length: int

    def __post_init__(self) -> None:
        self.layout()

    def layout(self) -> FlashLayout:
        """The packed layout of the shape."""

        return dense_layout(self.sequences, self.length)

    def to_mapping(self) -> dict[str, int]:
        """Return the shape as the config file spells it."""

        return {"sequences": self.sequences, "length": self.length}


@dataclass(frozen=True)
class FlashPoint:
    """One timing point: a shape in one attention mode."""

    index: int
    shape: FlashShape
    mode: str

    def to_mapping(self) -> dict[str, object]:
        """Return the point as a report names it."""

        return {"index": self.index, **self.shape.to_mapping(), "mode": self.mode}


def _require_coverage(cases: tuple[FlashCase, ...]) -> None:
    lengths = [length for case in cases for length in case.lengths]
    checks = [
        (1 in lengths, "a sequence of 1 token"),
        (
            any(1 < length < BLOCK_QUERIES for length in lengths),
            "a sequence shorter than a query block",
        ),
        (
            any(
                length > BLOCK_QUERIES and length % BLOCK_QUERIES for length in lengths
            ),
            "a sequence longer than 32 that is not a multiple of 32",
        ),
        (
            any(length >= LONG_SEQUENCE for length in lengths),
            f"a sequence of {LONG_SEQUENCE} or more tokens",
        ),
        (any(case.ragged for case in cases), "a ragged case"),
        (any(case.inputs == WIDE_SCORES for case in cases), "a wide_scores case"),
    ]
    for covered, noun in checks:
        if not covered:
            raise MetalFlashError(f"correctness cases need {noun}")


@dataclass(frozen=True)
class CorrectnessControls:
    """The cases and how many runs measure determinism."""

    cases: tuple[FlashCase, ...]
    repeats: int

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES:
            raise MetalFlashError(f"correctness holds 1 to {MAX_CASES} cases")
        if not all(isinstance(case, FlashCase) for case in self.cases):
            raise TypeError("cases must be FlashCase values")
        _require_coverage(self.cases)
        require_size(self.repeats, "repeats", MIN_CHECK_REPEATS, MAX_CHECK_REPEATS)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "cases": [case.to_mapping() for case in self.cases],
            "repeats": self.repeats,
        }


@dataclass(frozen=True)
class TimingControls:
    """The shapes, modes, checked rows, warmups, repeats, batch and budget."""

    inputs: str
    shapes: tuple[FlashShape, ...]
    modes: tuple[str, ...]
    check_rows: int
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        _distribution(self.inputs)
        if type(self.shapes) is not tuple or not 1 <= len(self.shapes) <= MAX_SHAPES:
            raise MetalFlashError(f"the benchmark holds 1 to {MAX_SHAPES} shapes")
        if not all(isinstance(shape, FlashShape) for shape in self.shapes):
            raise TypeError("shapes must be FlashShape values")
        if len(set(self.shapes)) != len(self.shapes):
            raise MetalFlashError("each timing shape appears once")
        if (
            type(self.modes) is not tuple
            or not self.modes
            or len(set(self.modes)) != len(self.modes)
            or not all(mode in MODES for mode in self.modes)
        ):
            raise MetalFlashError(f"modes must be distinct values of {MODES}")
        if tuple(mode for mode in MODES if mode in self.modes) != self.modes:
            raise MetalFlashError("modes keep the order full, then causal")
        require_size(self.check_rows, "check_rows", MIN_CHECK_ROWS, MAX_CHECK_ROWS)
        require_size(self.warmup, "warmup", 0, MAX_WARMUP)
        require_size(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        if self.repeats % len(SIDES):
            raise MetalFlashError(
                f"repeats must be a multiple of the {len(SIDES)} timed sides, so "
                "each side takes each position of the rotation equally often"
            )
        require_size(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )
        require_size(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def points(self) -> tuple[FlashPoint, ...]:
        """Every timing point: each shape in each mode, shape by shape."""

        pairs = [(shape, mode) for shape in self.shapes for mode in self.modes]
        return tuple(
            FlashPoint(index, shape, mode) for index, (shape, mode) in enumerate(pairs)
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "inputs": self.inputs,
            "shapes": [shape.to_mapping() for shape in self.shapes],
            "modes": list(self.modes),
            "check_rows": self.check_rows,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class MetalFlashConfig:
    """Every control one Day 81 run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: TimingControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalFlashError("config name must be a lowercase slug")
        require_size(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise MetalFlashError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise MetalFlashError(
                "plan_sha256 does not name the declared plan; a config for other "
                "hypotheses cannot run this plan"
            )
        for value, kind in (
            (self.correctness, CorrectnessControls),
            (self.benchmark, TimingControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work["correctness_pairs"], MAX_CORRECTNESS_PAIRS, "correctness pairs"),
            (work["calls"], MAX_RUN_CALLS, "side calls"),
            (work["compulsory_bytes"], MAX_RUN_BYTES, "compulsory bytes"),
        ):
            if value > limit:
                raise MetalFlashError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> dict[str, int]:
        """Count the cases, samples, calls and bytes of the run up front."""

        timing = self.benchmark
        samples = timing.warmup + timing.repeats
        calls = moved = 0
        points = timing.points()
        for point in points:
            work = compulsory_bytes(point.shape.layout())
            batch = calls_per_sample(work, timing.target_bytes_per_sample)
            calls += len(SIDES) * samples * batch
            moved += len(SIDES) * samples * batch * work
        cases = self.correctness.cases
        return {
            "correctness_cases": len(cases),
            "correctness_pairs": sum(case.layout().score_elements for case in cases),
            "correctness_launches": len(cases)
            * len(MODES)
            * 2
            * self.correctness.repeats,
            "benchmark_points": len(points),
            "sides_per_point": len(SIDES),
            "timed_samples": len(SIDES) * len(points) * timing.repeats,
            "calls": calls,
            "compulsory_bytes": moved,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping that a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "plan_sha256": self.plan_sha256,
            "correctness": self.correctness.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
            "comparison": self.comparison.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise MetalFlashError(f"{label} must be a JSON array")
    return value


def _case(payload: object) -> FlashCase:
    if type(payload) is dict and set(payload) == {"sequences", "length", "inputs"}:
        length = require_size(payload["length"], "length", 1, MAX_LENGTH)
        layout = dense_layout(payload["sequences"], length)
        return FlashCase(layout.lengths, payload["inputs"], length)
    case = _fields(payload, {"lengths", "inputs"}, "correctness case")
    return FlashCase(tuple(_list(case["lengths"], "lengths")), case["inputs"])


def _shape(payload: object) -> FlashShape:
    shape = _fields(payload, {"sequences", "length"}, "benchmark shape")
    return FlashShape(shape["sequences"], shape["length"])


def _config(payload: object) -> MetalFlashConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MetalFlashError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], {"cases", "repeats"}, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], {"min_faster_ratio"}, "comparison")
    return MetalFlashConfig(
        name=config["name"],
        seed=config["seed"],
        plan_sha256=config["plan_sha256"],
        correctness=CorrectnessControls(
            cases=tuple(_case(item) for item in _list(correctness["cases"], "cases")),
            repeats=correctness["repeats"],
        ),
        benchmark=TimingControls(
            inputs=benchmark["inputs"],
            shapes=tuple(_shape(item) for item in _list(benchmark["shapes"], "shapes")),
            modes=tuple(_list(benchmark["modes"], "modes")),
            check_rows=benchmark["check_rows"],
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MetalFlashConfig:
    """Parse controls, reporting every contract violation as ``MetalFlashError``.

    The Day 73 and Day 76 controls raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except MetalFlashError:
        raise
    except (TypeError, ValueError) as error:
        raise MetalFlashError(f"invalid metal-flash config: {error}") from error


def load_config(path: Path) -> MetalFlashConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MetalFlashError(str(error)) from error
    return config_from_mapping(payload)
