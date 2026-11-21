"""Strict, bounded controls for one fused softmax correctness and timing run.

A config names the seed, the digest of the declared plan, the correctness
cases, the timing shapes with the Day 74 controls and the smallest ratio of
median times that calls one side faster. A run times every step of the
plan, so the config pins the plan's digest, and a config made for other
hypotheses is refused.

A correctness case is ``{"rows", "cols"}`` for a dense batch or
``{"lengths"}`` for a ragged one, with a named input distribution, and its
rows are laid out with ``CORRECTNESS_GAP`` guard slots after each row. A
timing shape is ``{"rows", "cols"}`` or ``{"rows", "max_length"}``; a ragged
shape draws its lengths from the seed. Before anything runs, the config is
checked as a whole:

- Some case has a row of 1 element, a row shorter than a SIMD group, and a
  row longer than a SIMD group that is not a multiple of 32. In a case with
  such a row, the longest row is longer than G, so thread 0 fills two or
  more slots of it.
- Some case is ragged, some case draws ``uniform_80``, and some case draws
  an offset distribution, where softmax with no max subtraction fails.
- The repeats are a multiple of the five timed sides, so each side takes
  each position of the rotation equally often.
- The cases, the probe, the launches, the calls and the bytes fit their
  bounds.

Files go through the bounded strict reader of ``memory_access_report_io``,
which also rejects duplicate keys.
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
from .metal_fused_plan import plan_sha256
from .metal_fused_reference import (
    CORRECTNESS_GAP,
    DISTRIBUTIONS,
    EXP_PROBE_COUNT,
    MAX_CASE_INDEX,
    OFFSET_MINUS_1000,
    OFFSET_PLUS_1000,
    UNIFORM_80,
    RowLayout,
    dense_layout,
    ragged_layout,
)
from .metal_fused_source import (
    FLOAT32_BYTES,
    SIMD_WIDTH,
    VARIANTS,
    MetalFusedError,
    require_count,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_CASES = 16
MAX_POINTS = 8
MAX_CORRECTNESS_ELEMENTS = 1 << 22
MAX_RUN_CALLS = 1 << 20
MAX_RUN_BYTES = 1 << 42
MIN_CHECK_REPEATS = 2
MAX_CHECK_REPEATS = 4
# The four steps and torch.softmax run at every point.
SIDES_PER_POINT = len(VARIANTS) + 1
LAUNCHES_PER_CALL = {
    name: variant.launches_per_call for name, variant in VARIANTS.items()
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
_CORRECTNESS_FIELDS = {"cases", "repeats"}
_BENCHMARK_FIELDS = {
    "inputs",
    "shapes",
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}
_COMPARISON_FIELDS = {"min_faster_ratio"}
_DENSE = {"rows", "cols"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MetalFusedError(f"{label} fields do not match the schema")
    return payload


def _distribution(value: object) -> str:
    if value not in DISTRIBUTIONS:
        raise MetalFusedError(
            f"inputs must be one of {', '.join(DISTRIBUTIONS)}, not {value!r}"
        )
    return value


@dataclass(frozen=True)
class CorrectnessCase:
    """One correctness batch: its row lengths and its input distribution.

    ``cols`` is set for a dense case and ``None`` for a ragged one, so the
    mapping keeps the form the config file used.
    """

    lengths: tuple[int, ...]
    inputs: str
    cols: int | None = None

    def __post_init__(self) -> None:
        _distribution(self.inputs)
        layout = self.layout()
        if self.cols is not None and set(self.lengths) != {self.cols}:
            raise MetalFusedError("a dense case holds rows of its cols only")
        if self.cols is None and layout.dense:
            raise MetalFusedError("a ragged case needs rows of two or more lengths")

    def layout(self) -> RowLayout:
        """The gapped layout the case runs with."""

        return RowLayout(self.lengths, CORRECTNESS_GAP)

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
class TimingShape:
    """One timing point: dense ``rows x cols`` or ``rows`` seeded lengths."""

    rows: int
    cols: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        require_count(self.rows, "rows", 1, 1 << 16)
        if (self.cols is None) == (self.max_length is None):
            raise MetalFusedError("a timing shape has cols or max_length, not both")
        width = self.cols if self.cols is not None else self.max_length
        dense_layout(self.rows, width)

    @property
    def ragged(self) -> bool:
        """Whether the point draws its row lengths from the seed."""

        return self.max_length is not None

    def layout(self, seed: int, index: int) -> RowLayout:
        """The packed layout of the point; a ragged one draws from the seed."""

        if self.cols is not None:
            return dense_layout(self.rows, self.cols)
        return ragged_layout(seed, index, self.rows, self.max_length)

    @property
    def padded_elements(self) -> int:
        """The elements of the dense matrix that ``torch.softmax`` receives."""

        return self.rows * (self.cols if self.cols is not None else self.max_length)

    def to_mapping(self) -> dict[str, object]:
        """Return the shape as the config file spells it."""

        if self.cols is not None:
            return {"rows": self.rows, "cols": self.cols}
        return {"rows": self.rows, "max_length": self.max_length}


@dataclass(frozen=True)
class CorrectnessControls:
    """The cases and how many runs of each kernel measure determinism."""

    cases: tuple[CorrectnessCase, ...]
    repeats: int

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= MAX_CASES:
            raise MetalFusedError(f"correctness holds 1 to {MAX_CASES} cases")
        if not all(isinstance(case, CorrectnessCase) for case in self.cases):
            raise TypeError("correctness cases must be CorrectnessCase values")
        require_count(self.repeats, "repeats", MIN_CHECK_REPEATS, MAX_CHECK_REPEATS)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "cases": [case.to_mapping() for case in self.cases],
            "repeats": self.repeats,
        }


@dataclass(frozen=True)
class TimingControls:
    """The shapes, input distribution, warmups, repeats, batch target and budget."""

    inputs: str
    shapes: tuple[TimingShape, ...]
    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        _distribution(self.inputs)
        if type(self.shapes) is not tuple or not 1 <= len(self.shapes) <= MAX_POINTS:
            raise MetalFusedError(f"the benchmark holds 1 to {MAX_POINTS} shapes")
        if not all(isinstance(shape, TimingShape) for shape in self.shapes):
            raise TypeError("benchmark shapes must be TimingShape values")
        require_count(self.warmup, "warmup", 0, MAX_WARMUP)
        require_count(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        if self.repeats % SIDES_PER_POINT:
            raise MetalFusedError(
                f"repeats must be a multiple of the {SIDES_PER_POINT} timed sides, so "
                "each side takes each position of the rotation equally often"
            )
        require_count(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )
        require_count(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "inputs": self.inputs,
            "shapes": [shape.to_mapping() for shape in self.shapes],
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class FusedPlannedWork:
    """What a config asks of the host at most, counted before anything runs."""

    correctness_cases: int
    correctness_elements: int
    correctness_launches: int
    exp_probe_arguments: int
    benchmark_points: int
    sides_per_point: int
    timed_samples: int
    calls: int
    kernel_launches: int
    compulsory_bytes: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts as a report records them."""

        return {
            "correctness_cases": self.correctness_cases,
            "correctness_elements": self.correctness_elements,
            "correctness_launches": self.correctness_launches,
            "exp_probe_arguments": self.exp_probe_arguments,
            "benchmark_points": self.benchmark_points,
            "sides_per_point": self.sides_per_point,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "kernel_launches": self.kernel_launches,
            "compulsory_bytes": self.compulsory_bytes,
        }


def compulsory_bytes(layout: RowLayout) -> int:
    """Bytes that any softmax of the layout must move: read once, write once."""

    return 2 * FLOAT32_BYTES * layout.elements


def _require_coverage(cases: tuple[CorrectnessCase, ...]) -> None:
    lengths = [length for case in cases for length in case.lengths]
    checks = (
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
        (any(case.inputs == UNIFORM_80 for case in cases), f"a {UNIFORM_80} case"),
        (
            any(case.inputs in (OFFSET_PLUS_1000, OFFSET_MINUS_1000) for case in cases),
            "an offset case, where softmax with no max subtraction fails",
        ),
    )
    for covered, noun in checks:
        if not covered:
            raise MetalFusedError(f"correctness cases need {noun}")


@dataclass(frozen=True)
class MetalFusedConfig:
    """Every control one fused softmax run reads, validated as a whole."""

    name: str
    seed: int
    plan_sha256: str
    correctness: CorrectnessControls
    benchmark: TimingControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MetalFusedError("config name must be a lowercase slug")
        require_count(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.plan_sha256, str)
            or _SHA256.fullmatch(self.plan_sha256) is None
        ):
            raise MetalFusedError("plan_sha256 must be a SHA-256 digest")
        if self.plan_sha256 != plan_sha256():
            raise MetalFusedError(
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
        _require_coverage(self.correctness.cases)
        if len(self.benchmark.shapes) > MAX_CASE_INDEX + 1:
            raise MetalFusedError("too many benchmark shapes for the case streams")
        work = self.planned_work()
        for value, limit, label in (
            (
                work.correctness_elements,
                MAX_CORRECTNESS_ELEMENTS,
                "correctness elements",
            ),
            (work.calls, MAX_RUN_CALLS, "side calls"),
            (work.compulsory_bytes, MAX_RUN_BYTES, "compulsory bytes"),
        ):
            if value > limit:
                raise MetalFusedError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def timing_layouts(self) -> tuple[RowLayout, ...]:
        """The packed layout of each timing point, ragged lengths from the seed."""

        return tuple(
            shape.layout(self.seed, index)
            for index, shape in enumerate(self.benchmark.shapes)
        )

    def largest_group(self) -> int:
        """The largest group that any launch of the run needs."""

        layouts = [case.layout() for case in self.correctness.cases]
        layouts += self.timing_layouts()
        return max(layout.group_size for layout in layouts)

    def planned_work(self) -> FusedPlannedWork:
        """Count the cases, launches, samples, calls and bytes of the run up front.

        Every registered step counts, so the counts are an upper bound when a
        pipeline refuses a step. The batch sizes are the run's own.
        """

        timing = self.benchmark
        cases = self.correctness.cases
        per_call = sum(LAUNCHES_PER_CALL.values())
        samples = timing.warmup + timing.repeats
        calls = launches = moved = 0
        for layout in self.timing_layouts():
            work = compulsory_bytes(layout)
            batch = calls_per_sample(work, timing.target_bytes_per_sample)
            calls += SIDES_PER_POINT * samples * batch
            launches += (per_call + 1) * samples * batch
            moved += SIDES_PER_POINT * samples * batch * work
        return FusedPlannedWork(
            correctness_cases=len(cases),
            correctness_elements=sum(case.layout().elements for case in cases),
            correctness_launches=len(cases) * self.correctness.repeats * per_call + 1,
            exp_probe_arguments=EXP_PROBE_COUNT,
            benchmark_points=len(timing.shapes),
            sides_per_point=SIDES_PER_POINT,
            timed_samples=SIDES_PER_POINT * len(timing.shapes) * timing.repeats,
            calls=calls,
            kernel_launches=launches,
            compulsory_bytes=moved,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

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
        raise MetalFusedError(f"{label} must be a JSON array")
    return value


def _case(payload: object) -> CorrectnessCase:
    if type(payload) is dict and set(payload) == {*_DENSE, "inputs"}:
        rows, cols = payload["rows"], payload["cols"]
        require_count(rows, "rows", 1, 1 << 16)
        return CorrectnessCase(
            dense_layout(rows, cols).lengths, payload["inputs"], cols
        )
    case = _fields(payload, {"lengths", "inputs"}, "correctness case")
    lengths = _list(case["lengths"], "lengths")
    return CorrectnessCase(tuple(lengths), case["inputs"])


def _shape(payload: object) -> TimingShape:
    if type(payload) is dict and set(payload) == _DENSE:
        return TimingShape(rows=payload["rows"], cols=payload["cols"])
    shape = _fields(payload, {"rows", "max_length"}, "benchmark shape")
    return TimingShape(rows=shape["rows"], max_length=shape["max_length"])


def _config(payload: object) -> MetalFusedConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MetalFusedError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], _COMPARISON_FIELDS, "comparison")
    return MetalFusedConfig(
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
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_bytes_per_sample=benchmark["target_bytes_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MetalFusedConfig:
    """Parse controls, reporting every contract violation as ``MetalFusedError``.

    The Day 73 and Day 76 controls raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except MetalFusedError:
        raise
    except (TypeError, ValueError) as error:
        raise MetalFusedError(f"invalid metal-fused config: {error}") from error


def load_config(path: Path) -> MetalFusedConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MetalFusedError(str(error)) from error
    return config_from_mapping(payload)
