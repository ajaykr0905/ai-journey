"""Strict, bounded controls for one checklist correctness and timing run.

A config names the seed, the group of the two Day 74 kernels, the digest of
the declared checklist, the correctness shapes, the timing sweep with the
Day 74 controls and the smallest ratio of median rates that calls one side
faster. A run times every kernel of the checklist, so the config pins the
digest of that plan, and a config made for other hypotheses is refused.

Before anything runs, the config is checked as a whole:

- Each Day 74 kernel pads both axes of its group in some case.
- For each block kernel, some case has an M, some an N and some a K that is
  not a multiple of the block, and some K above the K phase is not a
  multiple of it. Every kernel thus pads both axes and ends a later phase
  early in some case.
- For each kernel with vector loads, some case stages A and some stages B
  on the scalar path, and some case stages both on the vector path with an
  edge on M or N and a short K phase.
- For the interior kernel, some case of the vector path holds an interior
  group, an edge group and a short K phase, so all three paths run.
- Each output with its guard tail fits one sentinel buffer, the CPU replay
  of all cases fits its step bound, and the calls and FLOPs fit the Day 74
  work bounds.

Files go through the bounded strict reader of ``memory_access_report_io``,
which also rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_geometry import require_integer
from .kernel_metal_reference import MAX_SEED, MAX_VECTOR_BUFFER
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_steps import checklist_sha256
from .matmul_checklist_variants import (
    NAIVE_VARIANTS,
    VARIANTS,
    ChecklistVariant,
    guard_elements,
    variant_launch,
)
from .matmul_metal_benchmark import calls_per_sample
from .matmul_metal_config import (
    MAX_REPLAY_STEPS,
    MAX_RUN_CALLS,
    MAX_RUN_FLOPS,
    BenchmarkControls,
    CorrectnessControls,
    LaunchControls,
)
from .matmul_metal_mapping import matmul_shape
from .matmul_tiled_config import ComparisonControls, PlannedWork
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
# torch.matmul runs beside the registered kernels at every point.
SIDES_PER_POINT = len(VARIANTS) + 1

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "checklist_sha256",
    "naive",
    "correctness",
    "benchmark",
    "comparison",
}
_NAIVE_FIELDS = {"group_size"}
_CORRECTNESS_FIELDS = {"shapes"}
_BENCHMARK_FIELDS = {
    "shapes",
    "warmup",
    "repeats",
    "target_flops_per_sample",
    "max_runtime_seconds",
}
_COMPARISON_FIELDS = {"min_faster_ratio"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MatmulChecklistError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MatmulChecklistError(f"{label} must be a JSON array")
    return tuple(value)


def _shapes(value: object, label: str) -> tuple[MatmulShape, ...]:
    shapes = []
    for item in _list(value, label):
        triple = _list(item, f"{label} item")
        if len(triple) != 3:
            raise MatmulChecklistError(f"{label} must be [m, n, k] triples")
        shapes.append(matmul_shape(*triple))
    return tuple(shapes)


def _pads(shape: MatmulShape, block: tuple[int, int, int]) -> dict[str, bool]:
    rows, columns, depth = block
    return {
        "m": shape.m % rows != 0,
        "n": shape.n % columns != 0,
        "k": shape.k % depth != 0,
        "short_phase": shape.k > depth and shape.k % depth != 0,
    }


def _require_block_paths(
    shapes: tuple[MatmulShape, ...], variant: ChecklistVariant
) -> None:
    """Require every path of one tiled or block kernel to run in some case."""

    design = variant.design
    if design is None:
        tile = variant.group_size[0]
        block = (tile, tile, tile)
    else:
        block = (design.block_m, design.block_n, design.block_k)
    for key, noun in (
        ("m", "an M"),
        ("n", "an N"),
        ("k", "a K"),
        ("short_phase", "a K above the K phase"),
    ):
        if not any(_pads(shape, block)[key] for shape in shapes):
            raise MatmulChecklistError(
                f"correctness shapes need {noun} that is not a multiple of the "
                f"{variant.name} block {list(block)}"
            )
    if design is None or not design.vector_loads:
        return
    vector = [shape for shape in shapes if shape.k % 4 == 0 and shape.n % 4 == 0]
    if not any(shape.k % 4 for shape in shapes) or not any(
        shape.n % 4 for shape in shapes
    ):
        raise MatmulChecklistError(
            f"correctness shapes need a K and an N that are not multiples of 4, "
            f"so that {variant.name} stages A and B on the scalar path"
        )
    if not any(
        (pads := _pads(shape, block))["short_phase"] and (pads["m"] or pads["n"])
        for shape in vector
    ):
        raise MatmulChecklistError(
            f"correctness shapes need a vector-path case of {variant.name} with an "
            "edge on M or N and a short K phase"
        )
    if design.interior_path and not any(
        _pads(shape, block)["short_phase"]
        and shape.m >= design.block_m
        and shape.n >= design.block_n
        and (shape.m % design.block_m or shape.n % design.block_n)
        for shape in vector
    ):
        raise MatmulChecklistError(
            f"correctness shapes need a vector-path case of {variant.name} with an "
            "interior group, an edge group and a short K phase"
        )


@dataclass(frozen=True)
class MatmulChecklistConfig:
    """Every control one checklist run reads, validated as a whole."""

    name: str
    seed: int
    checklist_sha256: str
    naive: LaunchControls
    correctness: CorrectnessControls
    benchmark: BenchmarkControls
    comparison: ComparisonControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MatmulChecklistError("config name must be a lowercase slug")
        require_integer(self.seed, "seed", 0, MAX_SEED)
        if (
            not isinstance(self.checklist_sha256, str)
            or _SHA256.fullmatch(self.checklist_sha256) is None
        ):
            raise MatmulChecklistError("checklist_sha256 must be a SHA-256 digest")
        if self.checklist_sha256 != checklist_sha256():
            raise MatmulChecklistError(
                "checklist_sha256 does not name the declared checklist; a config "
                "for other hypotheses cannot run this plan"
            )
        for value, kind in (
            (self.naive, LaunchControls),
            (self.correctness, CorrectnessControls),
            (self.benchmark, BenchmarkControls),
            (self.comparison, ComparisonControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        shapes = self.correctness.shapes
        group_x, group_y = self.naive.group_size
        for name, axes in (("matmul_x_columns", "nm"), ("matmul_x_rows", "mn")):
            for axis, group, label in zip(axes, (group_x, group_y), "xy"):
                if all(getattr(shape, axis) % group == 0 for shape in shapes):
                    raise MatmulChecklistError(
                        f"correctness shapes need an {axis.upper()} that is not a "
                        f"multiple of the naive group {label} size {group}, so that "
                        f"{name} pads its {label} axis"
                    )
        for name, variant in VARIANTS.items():
            if name not in NAIVE_VARIANTS:
                _require_block_paths(shapes, variant)
            for shape in shapes:
                launch = variant_launch(variant, shape, self.naive.group_size)
                elements = shape.m * shape.n + guard_elements(variant, shape, launch)
                if elements > MAX_VECTOR_BUFFER:
                    raise MatmulChecklistError(
                        f"correctness shape {[shape.m, shape.n, shape.k]} needs "
                        f"{elements} output elements for {name} with its guard "
                        f"tail; the bound is {MAX_VECTOR_BUFFER}"
                    )
        work = self.planned_work()
        for value, limit, label in (
            (work.replay_steps, MAX_REPLAY_STEPS, "replay steps"),
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.flops, MAX_RUN_FLOPS, "FLOPs"),
        ):
            if value > limit:
                raise MatmulChecklistError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the cases, launches, samples, calls and FLOPs of the run up front.

        Every registered kernel counts, so the counts are an upper bound when
        a pipeline refuses a kernel. The batch sizes are the run's own.
        """

        timing = self.benchmark
        samples = SIDES_PER_POINT * (timing.warmup + timing.repeats)
        calls = flops = 0
        for shape in timing.shapes:
            batch = calls_per_sample(shape.flops, timing.target_flops_per_sample)
            calls += samples * batch
            flops += samples * batch * shape.flops
        cases = self.correctness.shapes
        return PlannedWork(
            correctness_cases=len(cases),
            replay_steps=sum(shape.m * shape.n * shape.k for shape in cases),
            correctness_launches=len(cases) * len(VARIANTS),
            benchmark_points=len(timing.shapes),
            sides_per_point=SIDES_PER_POINT,
            timed_samples=SIDES_PER_POINT * len(timing.shapes) * timing.repeats,
            calls=calls,
            flops=flops,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "checklist_sha256": self.checklist_sha256,
            "naive": self.naive.to_mapping(),
            "correctness": self.correctness.to_mapping(),
            "benchmark": self.benchmark.to_mapping(),
            "comparison": self.comparison.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> MatmulChecklistConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MatmulChecklistError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    naive = _fields(config["naive"], _NAIVE_FIELDS, "naive")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    benchmark = _fields(config["benchmark"], _BENCHMARK_FIELDS, "benchmark")
    comparison = _fields(config["comparison"], _COMPARISON_FIELDS, "comparison")
    return MatmulChecklistConfig(
        name=config["name"],
        seed=config["seed"],
        checklist_sha256=config["checklist_sha256"],
        naive=LaunchControls(group_size=_list(naive["group_size"], "group_size")),
        correctness=CorrectnessControls(
            shapes=_shapes(correctness["shapes"], "correctness shapes")
        ),
        benchmark=BenchmarkControls(
            shapes=_shapes(benchmark["shapes"], "benchmark shapes"),
            warmup=benchmark["warmup"],
            repeats=benchmark["repeats"],
            target_flops_per_sample=benchmark["target_flops_per_sample"],
            max_runtime_seconds=benchmark["max_runtime_seconds"],
        ),
        comparison=ComparisonControls(min_faster_ratio=comparison["min_faster_ratio"]),
    )


def config_from_mapping(payload: object) -> MatmulChecklistConfig:
    """Parse controls, reporting every contract violation as ``MatmulChecklistError``.

    The Day 74 and Day 76 sections raise their own labs' errors; they are
    rewrapped, so a caller catches one type.
    """

    try:
        return _config(payload)
    except MatmulChecklistError:
        raise
    except (TypeError, ValueError) as error:
        raise MatmulChecklistError(
            f"invalid matmul-checklist config: {error}"
        ) from error


def load_config(path: Path) -> MatmulChecklistConfig:
    """Read one bounded, duplicate-free JSON config file and validate it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MatmulChecklistError(str(error)) from error
    return config_from_mapping(payload)
