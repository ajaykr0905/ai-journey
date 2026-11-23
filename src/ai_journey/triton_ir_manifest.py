"""Strict model of the recorded Triton stage dump of the Day 79 matmul kernel.

The session owner compiled ``ai_journey.triton_matmul_kernel.TRITON_SOURCE``
with triton 3.9.0 in a Linux aarch64 container for two declared NVIDIA
targets, ``sm_80`` and ``sm_90``, once for each of the six
``AUTOTUNE_CONFIGS``. The dump script wrote the TTIR, TTGIR, LLVM IR and PTX
text of each compile and an ``index.json`` with their SHA-256 digests and
line counts, the shared memory bytes, warps and stages from the compiled
metadata, and the digest of each cubin. No NVIDIA GPU ran any of it.

``load_manifest`` parses that ``index.json`` and refuses any field, config or
path that the dump script could not have written, so a later reader can trust
each number it takes from the model. The repository keeps only three stage
files, ``COMMITTED``: the TTIR, TTGIR and PTX of the smallest ``sm_80``
config. ``check_fixtures`` checks their bytes against the manifest, and
``check_source`` checks that the recorded source digest is the digest of the
kernel text of today. All other stages, and all LLVM IR, stay digest only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_launch_config import KernelLaunchError
from .kernel_launch_io import loads_strict, read_bounded
from .triton_matmul_kernel import AUTOTUNE_CONFIGS, TritonConfig, source_sha256

TARGETS = ("sm_80", "sm_90")
STAGES = ("ttir", "ttgir", "llir", "ptx")
STAGE_FILES = {
    "ttir": "ttir.mlir",
    "ttgir": "ttgir.mlir",
    "llir": "llir.ll",
    "ptx": "ptx.ptx",
}
COMMITTED_TARGET = "sm_80"
COMMITTED_CONFIG = "m32_n64_k32_g8_w2_s5"
COMMITTED_STAGES = ("ttir", "ttgir", "ptx")
MANIFEST_NAME = "index.json"
# The raw bytes of the recorded index.json, so a report can name its input.
RECORDED_MANIFEST_SHA256 = (
    "95c187abaae39edda3262eec53588855360b4d9b933f7cbe4e3536b3696b6473"
)
RECORDED_TRITON = "3.9.0"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_STAGE_BYTES = 1024 * 1024
# The opt-in shared memory of one block on sm_90 is 227 KiB, the largest of
# both targets, so a larger compile result is not a result of these targets.
MAX_SHARED_BYTES = 227 * 1024
MAX_CUBIN_BYTES = 16 * 1024 * 1024
MAX_STAGE_LINES = 1_000_000
_TOP_FIELDS = {
    "TRITON_DEFAULT_FP_FUSION",
    "machine",
    "python",
    "source_sha256",
    "targets",
    "triton",
}
_RECORD_FIELDS = {
    "config",
    "cubin_bytes",
    "cubin_sha256",
    "num_stages",
    "num_warps",
    "shared_bytes",
    "stages",
}
_STAGE_FIELDS = {"file", "lines", "sha256"}
_HEX = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


class TritonIrError(ValueError):
    """Raised when a Day 80 Triton IR input, fixture or report breaks its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TritonIrError(message)


def _digest(value: object, name: str) -> str:
    _require(
        type(value) is str and _HEX.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 hex digest",
    )
    return value


def _count(value: object, name: str, low: int, high: int) -> int:
    _require(
        type(value) is int and low <= value <= high,
        f"{name} must be an integer in [{low}, {high}]",
    )
    return value


def stage_path(target: str, config: str, stage: str) -> str:
    """Return the relative path that the dump script gives one stage file."""

    return f"{target}/{config}/{STAGE_FILES[stage]}"


@dataclass(frozen=True)
class StageRecord:
    """The digest and line count of one stage text, as ``index.json`` names it."""

    stage: str
    file: str
    sha256: str
    lines: int

    def __post_init__(self) -> None:
        _require(self.stage in STAGES, f"stage must be one of {STAGES}")
        _require(type(self.file) is str, "stage file must be a string")
        _digest(self.sha256, f"{self.stage} sha256")
        _count(self.lines, f"{self.stage} lines", 1, MAX_STAGE_LINES)

    def to_mapping(self) -> dict[str, object]:
        return {"file": self.file, "lines": self.lines, "sha256": self.sha256}


@dataclass(frozen=True)
class CompiledConfig:
    """One compile of one config for one target: metadata and stage digests."""

    target: str
    config: TritonConfig
    shared_bytes: int
    num_warps: int
    num_stages: int
    cubin_sha256: str
    cubin_bytes: int
    stages: tuple[StageRecord, ...]

    def __post_init__(self) -> None:
        _require(self.target in TARGETS, f"target must be one of {TARGETS}")
        _require(type(self.config) is TritonConfig, "config must be TritonConfig")
        _count(self.shared_bytes, "shared_bytes", 0, MAX_SHARED_BYTES)
        _require(
            self.num_warps == self.config.num_warps and type(self.num_warps) is int,
            f"{self.config.name}: compiled num_warps must equal the config",
        )
        _require(
            self.num_stages == self.config.num_stages and type(self.num_stages) is int,
            f"{self.config.name}: compiled num_stages must equal the config",
        )
        _digest(self.cubin_sha256, "cubin_sha256")
        _count(self.cubin_bytes, "cubin_bytes", 1, MAX_CUBIN_BYTES)
        _require(
            type(self.stages) is tuple
            and tuple(record.stage for record in self.stages) == STAGES,
            f"stages must be {STAGES} in order",
        )
        for record in self.stages:
            _require(
                record.file == stage_path(self.target, self.config.name, record.stage),
                f"{record.file} is not the dump path of its stage",
            )

    @property
    def name(self) -> str:
        return self.config.name

    def stage(self, stage: str) -> StageRecord:
        return self.stages[STAGES.index(stage)]

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config.to_mapping(),
            "cubin_bytes": self.cubin_bytes,
            "cubin_sha256": self.cubin_sha256,
            "num_stages": self.num_stages,
            "num_warps": self.num_warps,
            "shared_bytes": self.shared_bytes,
            "stages": {record.stage: record.to_mapping() for record in self.stages},
        }


@dataclass(frozen=True)
class TritonIrManifest:
    """The whole recorded dump: versions, source digest and every compile."""

    triton: str
    python: str
    machine: str
    fp_fusion: str
    source_sha256: str
    records: tuple[CompiledConfig, ...]

    def __post_init__(self) -> None:
        for name in ("triton", "python"):
            value = getattr(self, name)
            _require(
                type(value) is str and _VERSION.fullmatch(value) is not None,
                f"{name} must be a version such as 3.9.0",
            )
        _require(
            type(self.machine) is str and self.machine.isidentifier(),
            "machine must be a platform.machine() name",
        )
        _require(
            self.fp_fusion in ("0", "1"), "TRITON_DEFAULT_FP_FUSION must be 0 or 1"
        )
        _digest(self.source_sha256, "source_sha256")
        expected = [
            (target, config.name) for target in TARGETS for config in AUTOTUNE_CONFIGS
        ]
        _require(
            type(self.records) is tuple
            and [(record.target, record.name) for record in self.records] == expected,
            "records must hold each AUTOTUNE_CONFIGS entry once per target, in order",
        )

    def record(self, target: str, config: str) -> CompiledConfig:
        for record in self.records:
            if record.target == target and record.name == config:
                return record
        raise TritonIrError(f"the manifest has no {target} {config}")

    def to_mapping(self) -> dict[str, object]:
        """Return the mapping that ``index.json`` holds, key for key."""

        return {
            "TRITON_DEFAULT_FP_FUSION": self.fp_fusion,
            "machine": self.machine,
            "python": self.python,
            "source_sha256": self.source_sha256,
            "targets": {
                target: [
                    record.to_mapping()
                    for record in self.records
                    if record.target == target
                ]
                for target in TARGETS
            },
            "triton": self.triton,
        }


def _stage(target: str, name: str, stage: str, value: object) -> StageRecord:
    _require(
        type(value) is dict and set(value) == _STAGE_FIELDS,
        f"{target} {name} {stage} fields do not match the dump",
    )
    return StageRecord(stage, value["file"], value["sha256"], value["lines"])


def _record(target: str, config: TritonConfig, value: object) -> CompiledConfig:
    _require(
        type(value) is dict and set(value) == _RECORD_FIELDS,
        f"{target} record fields do not match the dump",
    )
    _require(
        value["config"] == config.to_mapping(),
        f"{target} config must be {config.name} as AUTOTUNE_CONFIGS declares it",
    )
    stages = value["stages"]
    _require(
        type(stages) is dict and set(stages) == set(STAGES),
        f"{target} {config.name} must name the stages {STAGES}",
    )
    return CompiledConfig(
        target=target,
        config=config,
        shared_bytes=value["shared_bytes"],
        num_warps=value["num_warps"],
        num_stages=value["num_stages"],
        cubin_sha256=value["cubin_sha256"],
        cubin_bytes=value["cubin_bytes"],
        stages=tuple(
            _stage(target, config.name, stage, stages[stage]) for stage in STAGES
        ),
    )


def parse_manifest(value: object) -> TritonIrManifest:
    """Build the model from the decoded JSON of ``index.json``."""

    _require(
        type(value) is dict and set(value) == _TOP_FIELDS,
        "manifest fields do not match the dump",
    )
    targets = value["targets"]
    _require(
        type(targets) is dict and set(targets) == set(TARGETS),
        f"manifest targets must be {TARGETS}",
    )
    records = []
    for target in TARGETS:
        rows = targets[target]
        _require(
            type(rows) is list and len(rows) == len(AUTOTUNE_CONFIGS),
            f"{target} must hold one record for each of the six configs",
        )
        records += [
            _record(target, config, row) for config, row in zip(AUTOTUNE_CONFIGS, rows)
        ]
    return TritonIrManifest(
        triton=value["triton"],
        python=value["python"],
        machine=value["machine"],
        fp_fusion=value["TRITON_DEFAULT_FP_FUSION"],
        source_sha256=value["source_sha256"],
        records=tuple(records),
    )


def _read(path: Path, limit: int, label: str) -> bytes:
    try:
        return read_bounded(path, limit=limit, label=label)
    except KernelLaunchError as error:
        raise TritonIrError(str(error)) from error


def read_manifest(directory: Path) -> tuple[bytes, TritonIrManifest]:
    """Read ``index.json`` of a fixture folder; return its bytes and its model."""

    payload = _read(directory / MANIFEST_NAME, MAX_MANIFEST_BYTES, "manifest")
    try:
        decoded = loads_strict(payload, label="manifest")
    except KernelLaunchError as error:
        raise TritonIrError(str(error)) from error
    return payload, parse_manifest(decoded)


def load_manifest(directory: Path) -> TritonIrManifest:
    return read_manifest(directory)[1]


def require_recorded(payload: bytes) -> None:
    """Refuse a manifest other than the one the session owner recorded."""

    _require(
        sha256(payload).hexdigest() == RECORDED_MANIFEST_SHA256,
        "the manifest is not the recorded Day 80 dump",
    )


def check_source(manifest: TritonIrManifest) -> str:
    """Return the source digest when it is the digest of today's kernel text."""

    _require(type(manifest) is TritonIrManifest, "manifest must be TritonIrManifest")
    _require(
        manifest.source_sha256 == source_sha256(),
        "the dump compiled another kernel text than TRITON_SOURCE",
    )
    return manifest.source_sha256


@dataclass(frozen=True)
class FixtureText:
    """One committed stage text whose bytes match its manifest record."""

    stage: str
    record: StageRecord
    text: str

    @property
    def byte_count(self) -> int:
        return len(self.text.encode("ascii"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": COMMITTED_TARGET,
            "config": COMMITTED_CONFIG,
            "stage": self.stage,
            "file": self.record.file,
            "sha256": self.record.sha256,
            "lines": self.record.lines,
            "bytes": self.byte_count,
        }


def read_stage(directory: Path, record: StageRecord) -> str:
    """Read one stage file and return its text only if it matches its record."""

    payload = _read(directory / record.file, MAX_STAGE_BYTES, record.stage)
    _require(
        sha256(payload).hexdigest() == record.sha256,
        f"{record.file} does not match its manifest digest",
    )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise TritonIrError(f"{record.file} must be ASCII") from error
    _require(
        len(text.splitlines()) == record.lines,
        f"{record.file} does not have its manifest line count",
    )
    return text


def check_fixtures(
    manifest: TritonIrManifest, directory: Path
) -> dict[str, FixtureText]:
    """Read the committed stage texts, each checked against the manifest."""

    record = manifest.record(COMMITTED_TARGET, COMMITTED_CONFIG)
    return {
        stage: FixtureText(
            stage, record.stage(stage), read_stage(directory, record.stage(stage))
        )
        for stage in COMMITTED_STAGES
    }
