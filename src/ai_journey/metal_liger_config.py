"""Strict, bounded controls for one Day 80 correctness and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases of each family, the timing shapes of each family with the Day 74
controls, and the smallest ratio of median times that calls one side
faster. A config made for other hypotheses names another plan digest and
is refused.

A correctness case is ``{"rows", "cols", "inputs"}`` for a dense batch or
``{"lengths", "inputs"}`` for a ragged one, and its rows are laid out with
``CORRECTNESS_GAP`` guard slots after each row. A timing shape is a dense
``{"rows", "cols"}`` batch, because the PyTorch sequences take a matrix.
Before anything runs, the config is checked as a whole:

- The cases of each family hold a row of 1 element, a row shorter than a
  SIMD group, a row longer than a SIMD group that is not a multiple of 32,
  a ragged case and a case of large-magnitude inputs (``uniform_80`` or an
  offset of 1000). The cross-entropy cases also hold a row longer than
  ``16 * 1024``, so each thread takes more than 16 columns.
- The repeats are a multiple of the two timed sides, so each side takes
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
from .metal_fused_reference import (
    DISTRIBUTIONS,
    OFFSET_MINUS_1000,
    OFFSET_PLUS_1000,
    UNIFORM_80,
)
from .metal_fused_source import MAX_GROUP, SIMD_WIDTH, SLOTS
from .metal_liger_plan import plan_sha256
from .metal_liger_reference import (
    CORRECTNESS_GAP,
    LigerLayout,
    dense_layout,
)
from .metal_liger_source import (
    CROSS_ENTROPY,
    FAMILIES,
    FLOAT32_BYTES,
    RMSNORM,
    SWIGLU,
    UINT32_BYTES,
    MetalLigerError,
    require_size,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 8
MAX_POINTS = 4
MAX_CORRECTNESS_ELEMENTS = 1 << 23
MAX_RUN_CALLS = 1 << 20
MAX_RUN_BYTES = 1 << 42
MIN_CHECK_REPEATS = 2
MAX_CHECK_REPEATS = 4
# The fused kernel and its PyTorch sequence run at every point.
SIDES_PER_POINT = 2
LARGE_MAGNITUDE = (UNIFORM_80, OFFSET_PLUS_1000, OFFSET_MINUS_1000)
COMPULSORY = {
    RMSNORM: "8 n + 4 H bytes: read x once, write y once, read the H weights once",
    SWIGLU: "12 n bytes: read a and b once, write y once",
    CROSS_ENTROPY: (
        "4 n + 8 r bytes: read each logit once, read one 4-byte target and "
        "write one loss for each row"
    ),
}

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
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MetalLigerError(f"{label} fields do not match the schema")
    return payload


def _distribution(value: object) -> str:
    if value not in DISTRIBUTIONS:
        raise MetalLigerError(
            f"inputs must be one of {', '.join(DISTRIBUTIONS)}, not {value!r}"
        )
    return value


def compulsory_bytes(family: str, layout: LigerLayout) -> int:
    """Bytes that any version of one family must move for a packed layout."""

    if family == RMSNORM:
        return 2 * FLOAT32_BYTES * layout.elements + FLOAT32_BYTES * layout.longest
    if family == SWIGLU:
        return 3 * FLOAT32_BYTES * layout.elements
    if family == CROSS_ENTROPY:
        return (
            FLOAT32_BYTES * layout.elements
            + (UINT32_BYTES + FLOAT32_BYTES) * layout.rows
        )
    raise MetalLigerError(f"{family!r} is not a kernel family")


@dataclass(frozen=True)
class LigerCase:
    """One correctness batch: its row lengths and its input distribution.

    ``cols`` is set for a dense case and ``None`` for a ragged one, so the
    mapping keeps the form that the config file used.
    """

    family: str
    lengths: tuple[int, ...]
    inputs: str
    cols: int | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise MetalLigerError(f"{self.family!r} is not a kernel family")
        _distribution(self.inputs)
        layout = self.layout()
        layout.group_size(self.family)
        if self.cols is not None and set(self.lengths) != {self.cols}:
            raise MetalLigerError("a dense case holds rows of its cols only")
        if self.cols is None and layout.dense:
            raise MetalLigerError("a ragged case needs rows of two or more lengths")

    def layout(self) -> LigerLayout:
        """The gapped layout the case runs with."""

        return LigerLayout(self.lengths, CORRECTNESS_GAP)

    @property
    def ragged(self) -> bool:
        """Whether the rows of the case have more than one length."""

        return self.cols is None

    def to_mapping(self) -> dict[str, object]:
        """Return the case as the config file spells it."""

        if self.cols is not None:
            return {"rows": len(self.lengths), "cols": self.cols, "inputs": self.inputs}
        return {"lengths": list(self.lengths), "inputs": self.inputs}


@dataclass(frozen=True)
class LigerShape:
    """One timing point of one family: a dense ``rows x cols`` batch."""

    family: str
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise MetalLigerError(f"{self.family!r} is not a kernel family")
        self.layout().group_size(self.family)

    def layout(self) -> LigerLayout:
        """The packed layout of the point."""

        return dense_layout(self.rows, self.cols)

    def to_mapping(self) -> dict[str, int]:
        """Return the shape as the config file spells it."""

        return {"rows": self.rows, "cols": self.cols}


def _require_coverage(family: str, cases: tuple[LigerCase, ...]) -> None:
    lengths = [length for case in cases for length in case.lengths]
    checks = [
        (1 in lengths, "a row of 1 element"),
        (
            any(1 < length < SIMD_WIDTH for length in lengths),
            "a row shorter than a SIMD group",
        ),
        (
            any(length > SIMD_WIDTH and length % SIMD_WIDTH for length in lengths),
            "a row longer than a SIMD group that is not a multiple of 32",
        ),
        (any(case.ragged for case in cases), "a ragged case"),
        (
            any(case.inputs in LARGE_MAGNITUDE for case in cases),
            "a case of large-magnitude inputs",
        ),
    ]
    if family == CROSS_ENTROPY:
        checks.append(
            (
                any(length > SLOTS * MAX_GROUP for length in lengths),
                f"a row longer than {SLOTS * MAX_GROUP}",
            )
        )
    for covered, noun in checks:
        if not covered:
            raise MetalLigerError(f"{family} correctness cases need {noun}")


@dataclass(frozen=True)
class CorrectnessControls:
    """The cases of each family and how many runs measure determinism."""

    cases: dict[str, tuple[LigerCase, ...]]
    repeats: int

    def __post_init__(self) -> None:
        if type(self.cases) is not dict or tuple(self.cases) != FAMILIES:
            raise MetalLigerError("correctness holds the cases of every family")
        for family, cases in self.cases.items():
            if type(cases) is not tuple or not 1 <= len(cases) <= MAX_CASES:
                raise MetalLigerError(f"{family} holds 1 to {MAX_CASES} cases")
            if not all(
                isinstance(case, LigerCase) and case.family == family for case in cases
            ):
                raise TypeError(f"{family} cases must be its own LigerCase values")
            _require_coverage(family, cases)
        require_size(self.repeats, "repeats", MIN_CHECK_REPEATS, MAX_CHECK_REPEATS)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "cases": {
                family: [case.to_mapping() for case in cases]
                for family, cases in self.cases.items()
            },
            "repeats": self.repeats,
        }


@dataclass(frozen=True)
class TimingControls:
    """The shapes of each family, inputs, warmups, repeats, batch and budget."""

    inputs: str
    shapes: dict[str, tuple[LigerShape, ...]]
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        _distribution(self.inputs)
        if type(self.shapes) is not dict or tuple(self.shapes) != FAMILIES:
            raise MetalLigerError("the benchmark holds the shapes of every family")
        for family, shapes in self.shapes.items():
            if type(shapes) is not tuple or not 1 <= len(shapes) <= MAX_POINTS:
                raise MetalLigerError(f"{family} holds 1 to {MAX_POINTS} shapes")
            if not all(
                isinstance(shape, LigerShape) and shape.family == family
                for shape in shapes
            ):
                raise TypeError(f"{family} shapes must be its own LigerShape values")
        require_size(self.warmup, "warmup", 0, MAX_WARMUP)
        require_size(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        if self.repeats % SIDES_PER_POINT:
            raise MetalLigerError(
                f"repeats must be a multiple of the {SIDES_PER_POINT} timed sides, so "
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

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "inputs": self.inputs,
            "shapes": {
                family: [shape.to_mapping() for shape in shapes]
                for family, shapes in self.shapes.items()
            },
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class MetalLigerConfig:
    """Every control one Day 80 run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: TimingControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalLigerError("config name must be a lowercase slug")
        require_size(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise MetalLigerError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise MetalLigerError(
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
            (work["correctness_elements"], MAX_CORRECTNESS_ELEMENTS, "elements"),
            (work["calls"], MAX_RUN_CALLS, "side calls"),
            (work["compulsory_bytes"], MAX_RUN_BYTES, "compulsory bytes"),
        ):
            if value > limit:
                raise MetalLigerError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def required_threads(self) -> int:
        """The largest group that any launch of the run needs."""

        groups = [
            case.layout().group_size(family)
            for family, cases in self.correctness.cases.items()
            for case in cases
        ]
        groups += [
            shape.layout().group_size(family)
            for family, shapes in self.benchmark.shapes.items()
            for shape in shapes
        ]
        return max(groups)

    def planned_work(self) -> dict[str, int]:
        """Count the cases, samples, calls and bytes of the run up front."""

        timing = self.benchmark
        samples = timing.warmup + timing.repeats
        calls = moved = points = 0
        for family, shapes in timing.shapes.items():
            for shape in shapes:
                work = compulsory_bytes(family, shape.layout())
                batch = calls_per_sample(work, timing.target_bytes_per_sample)
                calls += SIDES_PER_POINT * samples * batch
                moved += SIDES_PER_POINT * samples * batch * work
                points += 1
        cases = [case for group in self.correctness.cases.values() for case in group]
        return {
            "correctness_cases": len(cases),
            "correctness_elements": sum(case.layout().elements for case in cases),
            "correctness_launches": len(cases) * self.correctness.repeats,
            "benchmark_points": points,
            "sides_per_point": SIDES_PER_POINT,
            "timed_samples": SIDES_PER_POINT * points * timing.repeats,
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
        raise MetalLigerError(f"{label} must be a JSON array")
    return value


def _case(family: str, payload: object) -> LigerCase:
    if type(payload) is dict and set(payload) == {"rows", "cols", "inputs"}:
        rows, cols = payload["rows"], payload["cols"]
        return LigerCase(
            family, dense_layout(rows, cols).lengths, payload["inputs"], cols
        )
    case = _fields(payload, {"lengths", "inputs"}, f"{family} correctness case")
    lengths = _list(case["lengths"], "lengths")
    return LigerCase(family, tuple(lengths), case["inputs"])


def _shape(family: str, payload: object) -> LigerShape:
    shape = _fields(payload, {"rows", "cols"}, f"{family} benchmark shape")
    return LigerShape(family, shape["rows"], shape["cols"])


def _by_family(payload: object, label: str) -> dict[str, list[object]]:
    groups = _fields(payload, set(FAMILIES), label)
    return {family: _list(groups[family], f"{label} {family}") for family in FAMILIES}


def _config(payload: object) -> MetalLigerConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MetalLigerError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], {"cases", "repeats"}, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], {"min_faster_ratio"}, "comparison")
    cases = _by_family(correctness["cases"], "cases")
    shapes = _by_family(benchmark["shapes"], "shapes")
    return MetalLigerConfig(
        name=config["name"],
        seed=config["seed"],
        plan_sha256=config["plan_sha256"],
        correctness=CorrectnessControls(
            cases={
                family: tuple(_case(family, item) for item in items)
                for family, items in cases.items()
            },
            repeats=correctness["repeats"],
        ),
        benchmark=TimingControls(
            inputs=benchmark["inputs"],
            shapes={
                family: tuple(_shape(family, item) for item in items)
                for family, items in shapes.items()
            },
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MetalLigerConfig:
    """Parse controls, reporting every contract violation as ``MetalLigerError``.

    The Day 73 and Day 76 controls raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except MetalLigerError:
        raise
    except (TypeError, ValueError) as error:
        raise MetalLigerError(f"invalid metal-liger config: {error}") from error


def load_config(path: Path) -> MetalLigerConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MetalLigerError(str(error)) from error
    return config_from_mapping(payload)
