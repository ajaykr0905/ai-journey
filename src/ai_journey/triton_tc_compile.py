"""Compile the four variants of the pinned text offline for sm_80 and sm_90.

``CompilerBackend`` imports triton only when a command makes it, never when
this module is imported. Triton has no macOS wheel, so on the development
host it stops with ``TritonTcUnavailableError`` before anything is compiled
or written, and a compile command exits 3. On a Linux host with the triton
3.9.0 wheel it compiles ``TRITON_SOURCE`` the way the Day 80 dump did:
``ASTSource`` with the variant's signature and constexprs,
``GPUTarget("cuda", 80 or 90, 32)``, and the warps and stages of
``TC_CONFIG``. A compile for a declared target needs no GPU, and none runs.

Each compile keeps the two texts that the lab reads, TTGIR and PTX, and the
shared memory bytes, warps and stages of the compile metadata. Only the
metadata states the shared memory of a launch: Triton declares one
``extern`` shared array with no size in the PTX.

``write_dump`` writes each text to ``<variant>/<target>/`` with an
``index.json`` of digests and metadata, so a CI job can keep the compiler
output as an artifact. ``load_dump`` reads such a folder back and refuses a
text whose digest or line count differs from the index.
"""

from __future__ import annotations

import importlib
import linecache
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import (
    canonical_json_bytes,
    loads_strict,
    read_bounded,
    write_atomic,
)
from .triton_tc_kernel import (
    KERNEL_NAME,
    TC_CONFIG,
    TRITON_SOURCE,
    VARIANTS,
    TcVariant,
    TritonTcError,
    require_pinned,
    require_variant,
    source_sha256,
)

COMPILER_BACKEND = "triton_compiler"
INJECTED_BACKEND = "injected"
JIT_TYPE = "JITFunction"
TARGETS = {"sm_80": 80, "sm_90": 90}
THREADS_PER_WARP = 32
REASONS = ("triton_not_importable", "compiler_not_importable", "interpreter_enabled")
ENVIRONMENT_FIELDS = ("python", "system", "machine", "triton")
COMPILE_ORDER = tuple(
    (variant.name, target) for variant in VARIANTS for target in TARGETS
)
STAGES = ("ttgir", "ptx")
STAGE_FILES = {"ttgir": "ttgir.mlir", "ptx": "ptx.ptx"}
INDEX_NAME = "index.json"
SUMMARY_NAME = "summary.json"
SOURCE_PREFIX = "<ai_journey.triton_tc_kernel.TRITON_SOURCE"
# The opt-in shared memory of one block on sm_90 is 227 KiB, the largest of
# both targets, so a larger compile result is not a result of these targets.
MAX_SHARED_BYTES = 227 * 1024
MAX_STAGE_BYTES = 4 * 1024 * 1024
MAX_INDEX_BYTES = 256 * 1024
_INDEX_FIELDS = {"kernel_sha256", "config", "environment", "compiles"}
_COMPILE_FIELDS = {
    "variant",
    "target",
    "shared_bytes",
    "num_warps",
    "num_stages",
    "stages",
}
_STAGE_FIELDS = {"file", "sha256", "lines", "bytes"}
_HEX = re.compile(r"[0-9a-f]{64}")


class TritonTcUnavailableError(TritonTcError):
    """Raised when this host cannot compile with triton; nothing was compiled."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        self.reason = reason
        super().__init__(f"{detail} ({reason})")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonTcError(message)


def stage_path(variant: str, target: str, stage: str) -> str:
    """Return the relative path of one stage text in a dump folder."""

    return f"{variant}/{target}/{STAGE_FILES[stage]}"


def _stage_record(variant: str, target: str, stage: str, text: str) -> dict:
    payload = text.encode("utf-8")
    return {
        "file": stage_path(variant, target, stage),
        "sha256": sha256(payload).hexdigest(),
        "lines": len(text.splitlines()),
        "bytes": len(payload),
    }


@dataclass(frozen=True)
class CompiledVariant:
    """The TTGIR and PTX texts and the metadata of one variant on one target."""

    variant: str
    target: str
    ttgir: str
    ptx: str
    shared_bytes: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        require_variant(self.variant)
        _require(self.target in TARGETS, f"target must be one of {tuple(TARGETS)}")
        for stage in STAGES:
            text = getattr(self, stage)
            _require(type(text) is str and bool(text), f"{stage} must be a text")
            _require(
                len(text.encode("utf-8")) <= MAX_STAGE_BYTES,
                f"{stage} exceeds {MAX_STAGE_BYTES} bytes",
            )
        _require(
            type(self.shared_bytes) is int
            and 0 <= self.shared_bytes <= MAX_SHARED_BYTES,
            f"shared_bytes must be an integer in [0, {MAX_SHARED_BYTES}]",
        )
        _require(
            type(self.num_warps) is int and self.num_warps == TC_CONFIG.num_warps,
            "compiled num_warps must equal the declared config",
        )
        _require(
            type(self.num_stages) is int and self.num_stages == TC_CONFIG.num_stages,
            "compiled num_stages must equal the declared config",
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.variant, self.target

    def text(self, stage: str) -> str:
        _require(stage in STAGES, f"stage must be one of {STAGES}")
        return getattr(self, stage)

    def to_mapping(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "target": self.target,
            "shared_bytes": self.shared_bytes,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "stages": {
                stage: _stage_record(self.variant, self.target, stage, self.text(stage))
                for stage in STAGES
            },
        }


def _import(name: str, reason: str) -> object:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise TritonTcUnavailableError(
            reason, f"{name} cannot be imported: {error}"
        ) from error


def source_filename(text: str) -> str:
    """Name one text in ``linecache`` by its digest, so no two texts share it."""

    return f"{SOURCE_PREFIX} sha256={source_sha256(text)}>"


def compile_text() -> tuple[str, dict[str, object]]:
    """Execute ``TRITON_SOURCE`` in a fresh namespace; return the text and it.

    ``triton.jit`` reads the source of the function that it wraps, so the
    text is first registered in ``linecache`` under its digest, as the Day 79
    runtime does.
    """

    text = require_pinned(TRITON_SOURCE)
    filename = source_filename(text)
    lines = text.splitlines(keepends=True)
    linecache.cache[filename] = (len(text), None, lines, filename)
    namespace: dict[str, object] = {"__name__": "ai_journey_triton_tc_text"}
    exec(compile(text, filename, "exec"), namespace)
    return text, namespace


class CompilerBackend:
    """Compile the pinned text with the installed triton for a declared target."""

    name = COMPILER_BACKEND

    def __init__(self) -> None:
        triton = _import("triton", "triton_not_importable")
        compiler = _import("triton.compiler", "compiler_not_importable")
        backends = _import("triton.backends.compiler", "compiler_not_importable")
        text, namespace = compile_text()
        kernel = namespace[KERNEL_NAME]
        if type(kernel).__name__ != JIT_TYPE:
            raise TritonTcUnavailableError(
                "interpreter_enabled",
                f"triton.jit gives {type(kernel).__name__}, so TRITON_INTERPRET "
                "turns the compiler off",
            )
        self.source_sha256 = source_sha256(text)
        self._triton = triton
        self._source = compiler.ASTSource
        self._target = backends.GPUTarget
        self._kernel = kernel
        self.environment = {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "triton": str(triton.__version__),
        }

    def compile(self, variant: TcVariant, target: str) -> CompiledVariant:
        """Compile one variant for one declared target; no GPU is needed."""

        if type(variant) is not TcVariant:
            raise TypeError("variant must be TcVariant")
        _require(target in TARGETS, f"target must be one of {tuple(TARGETS)}")
        source = self._source(
            fn=self._kernel,
            signature=variant.signature(),
            constexprs=variant.constexprs(TC_CONFIG),
        )
        try:
            compiled = self._triton.compile(
                source,
                target=self._target("cuda", TARGETS[target], THREADS_PER_WARP),
                options=TC_CONFIG.options(),
            )
        except Exception as error:
            raise TritonTcError(
                f"triton could not compile {variant.name} for {target}: {error}"
            ) from error
        return CompiledVariant(
            variant=variant.name,
            target=target,
            ttgir=compiled.asm["ttgir"],
            ptx=compiled.asm["ptx"],
            shared_bytes=int(compiled.metadata.shared),
            num_warps=int(compiled.metadata.num_warps),
            num_stages=int(compiled.metadata.num_stages),
        )


def environment_of(backend: object) -> dict[str, str]:
    """Return the environment that a backend names, or refuse the backend."""

    environment = getattr(backend, "environment", None)
    _require(
        isinstance(environment, Mapping)
        and tuple(environment) == ENVIRONMENT_FIELDS
        and all(type(value) is str for value in environment.values()),
        f"a backend must name its environment: {', '.join(ENVIRONMENT_FIELDS)}",
    )
    return dict(environment)


def compile_all(backend: object) -> tuple[CompiledVariant, ...]:
    """Compile every variant for every target, in ``COMPILE_ORDER``."""

    environment_of(backend)
    results = []
    for name, target in COMPILE_ORDER:
        compiled = backend.compile(require_variant(name), target)
        _require(
            type(compiled) is CompiledVariant,
            "a backend must return CompiledVariant",
        )
        _require(
            compiled.key == (name, target),
            f"a backend returned {compiled.key} for {(name, target)}",
        )
        results.append(compiled)
    return tuple(results)


def index_mapping(
    compiled: Sequence[CompiledVariant], environment: Mapping[str, str]
) -> dict[str, object]:
    """Return the ``index.json`` mapping of one dump of every compile."""

    _require(
        tuple(item.key for item in compiled) == COMPILE_ORDER,
        "a dump must hold every variant and target once, in COMPILE_ORDER",
    )
    return {
        "kernel_sha256": source_sha256(),
        "config": TC_CONFIG.to_mapping(),
        "environment": dict(environment),
        "compiles": [item.to_mapping() for item in compiled],
    }


def write_dump(
    directory: Path,
    compiled: Sequence[CompiledVariant],
    environment: Mapping[str, str],
    summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write every text, the index and an optional summary; return the index."""

    if not isinstance(directory, Path):
        raise TypeError("dump folder must be pathlib.Path")
    index = index_mapping(compiled, environment)
    for item in compiled:
        for stage in STAGES:
            path = directory / stage_path(item.variant, item.target, stage)
            write_atomic(path, item.text(stage).encode("utf-8"))
    write_atomic(directory / INDEX_NAME, canonical_json_bytes(index) + b"\n")
    if summary is not None:
        write_atomic(directory / SUMMARY_NAME, canonical_json_bytes(summary) + b"\n")
    return index


def _read(path: Path, limit: int, label: str) -> bytes:
    try:
        return read_bounded(path, limit=limit, label=label)
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error


def _fields(value: object, expected: set[str], label: str) -> Mapping:
    _require(
        isinstance(value, Mapping) and set(value) == expected,
        f"{label} fields do not match a dump index",
    )
    return value


def load_dump(directory: Path) -> tuple[dict[str, object], tuple[CompiledVariant, ...]]:
    """Read a dump folder; return its index and the compiles it holds."""

    if not isinstance(directory, Path):
        raise TypeError("dump folder must be pathlib.Path")
    payload = _read(directory / INDEX_NAME, MAX_INDEX_BYTES, "dump index")
    try:
        index = loads_strict(payload, label="dump index")
    except KernelLaunchError as error:
        raise TritonTcError(str(error)) from error
    _fields(index, _INDEX_FIELDS, "dump index")
    _require(index["kernel_sha256"] == source_sha256(), "the dump names another text")
    _require(index["config"] == TC_CONFIG.to_mapping(), "the dump names another config")
    environment = index["environment"]
    _require(
        isinstance(environment, Mapping)
        and set(environment) == set(ENVIRONMENT_FIELDS)
        and all(type(value) is str for value in environment.values()),
        "the dump must name its environment",
    )
    index["environment"] = {name: environment[name] for name in ENVIRONMENT_FIELDS}
    rows = index["compiles"]
    _require(
        type(rows) is list and len(rows) == len(COMPILE_ORDER),
        "the dump must hold every variant and target once",
    )
    compiled = []
    for row, (name, target) in zip(rows, COMPILE_ORDER):
        _fields(row, _COMPILE_FIELDS, "dump compile")
        _require(
            (row["variant"], row["target"]) == (name, target),
            "dump compiles must follow COMPILE_ORDER",
        )
        stages = _fields(row["stages"], set(STAGES), "dump stages")
        texts = {}
        for stage in STAGES:
            record = _fields(stages[stage], _STAGE_FIELDS, "dump stage")
            _require(
                record["file"] == stage_path(name, target, stage),
                f"{record['file']} is not the dump path of its stage",
            )
            _require(
                type(record["sha256"]) is str
                and _HEX.fullmatch(record["sha256"]) is not None,
                "a stage digest must be a lowercase SHA-256 hex digest",
            )
            data = _read(directory / record["file"], MAX_STAGE_BYTES, stage)
            _require(
                sha256(data).hexdigest() == record["sha256"],
                f"{record['file']} does not match its index digest",
            )
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TritonTcError(f"{record['file']} must be UTF-8") from error
            _require(
                record["lines"] == len(text.splitlines())
                and record["bytes"] == len(data),
                f"{record['file']} does not have its index size",
            )
            texts[stage] = text
        compiled.append(
            CompiledVariant(
                variant=name,
                target=target,
                ttgir=texts["ttgir"],
                ptx=texts["ptx"],
                shared_bytes=row["shared_bytes"],
                num_warps=row["num_warps"],
                num_stages=row["num_stages"],
            )
        )
    return dict(index), tuple(compiled)
