"""Strict, bounded controls for one CPU memory-hierarchy measurement run.

A config names the dtype and seed shared by every section, the timing budget
(warmups, repeats, bytes per sample and a wall-clock limit), the STREAM
kernels and working-set sweep points, the access-pattern source and stride,
the matmul shape behind the compute ceiling, and the stated latency the
Little's-law rows multiply by. Parsing rejects unknown and missing fields at
every level, booleans posing as integers and out-of-range values. Before any
array is allocated, the config is also checked as a whole: the largest
section must fit the memory bound, every sample must fit the batch limit, and
the run's total accounted bytes, calls and FLOPs must fit the work bounds,
so a valid config cannot ask for an unbounded run. Files are read through
the bounded strict reader of ``memory_access_report_io``, which also rejects
duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_hierarchy_access import (
    ACCESS_PATTERNS,
    INDEX_BYTES,
    PATTERN_NAMES,
    validate_access,
)
from .memory_hierarchy_matmul import matmul_allocation_bytes, validate_matmul
from .memory_hierarchy_stream import (
    ARRAY_NAMES,
    CHECK_CHUNK_ELEMENTS,
    MAX_SEED,
    MemoryHierarchyError,
    element_bytes,
    require_integer,
    stream_kernel,
)
from .memory_hierarchy_sweep import validate_sweep
from .memory_hierarchy_timing import (
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_ALLOCATION_BYTES = 1 << 28
MAX_RUN_BYTES = 1 << 38
MAX_RUN_CALLS = 1 << 22
MAX_RUN_FLOPS = 1 << 38
MAX_LATENCY_NS = 100_000
MAX_NOTE_CHARACTERS = 400

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {
    "schema_version",
    "name",
    "dtype",
    "seed",
    "timing",
    "stream",
    "access",
    "matmul",
    "littles_law",
}
_TIMING_FIELDS = {
    "warmup",
    "repeats",
    "target_bytes_per_sample",
    "max_runtime_seconds",
}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise MemoryHierarchyError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise MemoryHierarchyError(f"{label} must be a JSON array")
    return tuple(value)


def _check_bytes(elements: int, width: int) -> int:
    # A chunked output check holds at most two chunk-sized temporaries.
    return 2 * min(CHECK_CHUNK_ELEMENTS, elements) * width


@dataclass(frozen=True)
class TimingControls:
    """Warmups, repeats, bytes per sample and the run's wall-clock limit."""

    warmup: int
    repeats: int
    target_bytes_per_sample: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        require_integer(self.warmup, "warmup", 0, MAX_WARMUP)
        require_integer(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_integer(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )
        require_integer(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    @property
    def samples(self) -> int:
        """Batches each measurement runs: untimed warmups plus timed repeats."""

        return self.warmup + self.repeats

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it, for the fingerprint."""

        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "target_bytes_per_sample": self.target_bytes_per_sample,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class StreamControls:
    """STREAM kernels and the per-array element counts of the sweep."""

    kernels: tuple[str, ...]
    elements: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_sweep(self.kernels, self.elements)
        if "triad" not in self.kernels:
            raise MemoryHierarchyError(
                "stream kernels must include triad, which sets the bandwidth ceiling"
            )

    def to_mapping(self) -> dict[str, list]:
        """Return the section as the config file spells it, for the fingerprint."""

        return {"kernels": list(self.kernels), "elements": list(self.elements)}


@dataclass(frozen=True)
class AccessControls:
    """Source size and stride of the access-pattern contrast."""

    elements: int
    stride_elements: int

    def __post_init__(self) -> None:
        validate_access(self.elements, self.stride_elements)

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it, for the fingerprint."""

        return {"elements": self.elements, "stride_elements": self.stride_elements}


@dataclass(frozen=True)
class MatmulControls:
    """Shape of the matmul whose FLOP rate sets the compute ceiling."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        validate_matmul(self.m, self.n, self.k)

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it, for the fingerprint."""

        return {"m": self.m, "n": self.n, "k": self.k}


@dataclass(frozen=True)
class LatencyControls:
    """A stated memory latency and where it came from; never measured here."""

    latency_ns: int
    latency_note: str

    def __post_init__(self) -> None:
        require_integer(self.latency_ns, "latency_ns", 1, MAX_LATENCY_NS)
        note = self.latency_note
        if (
            not isinstance(note, str)
            or not note
            or note != " ".join(note.split())
            or len(note) > MAX_NOTE_CHARACTERS
            or not note.isascii()
        ):
            raise MemoryHierarchyError(
                "latency_note must be single-spaced ASCII text of 1 to "
                f"{MAX_NOTE_CHARACTERS} characters"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it, for the fingerprint."""

        return {"latency_ns": self.latency_ns, "latency_note": self.latency_note}


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host, computed before anything is allocated."""

    peak_allocation_bytes: int
    accounted_bytes: int
    calls: int
    flops: int
    measurements: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "peak_allocation_bytes": self.peak_allocation_bytes,
            "accounted_bytes": self.accounted_bytes,
            "calls": self.calls,
            "flops": self.flops,
            "measurements": self.measurements,
        }


@dataclass(frozen=True)
class MemoryHierarchyConfig:
    """Every control one measurement run reads, validated as a whole."""

    name: str
    dtype: str
    seed: int
    timing: TimingControls
    stream: StreamControls
    access: AccessControls
    matmul: MatmulControls
    littles_law: LatencyControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise MemoryHierarchyError("config name must be a lowercase slug")
        element_bytes(self.dtype)
        require_integer(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.timing, TimingControls),
            (self.stream, StreamControls),
            (self.access, AccessControls),
            (self.matmul, MatmulControls),
            (self.littles_law, LatencyControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        work = self.planned_work()
        for value, limit, label in (
            (work.peak_allocation_bytes, MAX_ALLOCATION_BYTES, "allocation"),
            (work.accounted_bytes, MAX_RUN_BYTES, "accounted bytes"),
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.flops, MAX_RUN_FLOPS, "FLOPs"),
        ):
            if value > limit:
                raise MemoryHierarchyError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def planned_work(self) -> PlannedWork:
        """Count the run's peak allocation, bytes, calls and FLOPs up front.

        Sections run one after another and free their arrays, so the peak is
        the largest section. Counting uses the same batch sizes the run will
        use, so a config that would need an oversized sample fails here.
        """

        width = element_bytes(self.dtype)
        samples = self.timing.samples
        target = self.timing.target_bytes_per_sample
        accounted = calls = flops = 0
        for count in self.stream.elements:
            for name in self.stream.kernels:
                kernel = stream_kernel(name)
                per_call = kernel.bytes_per_call(count, self.dtype)
                batch = calls_per_sample(per_call, target)
                accounted += samples * batch * per_call
                calls += samples * batch
                flops += samples * batch * kernel.flops_per_call(count)
        source = self.access.elements
        for name in PATTERN_NAMES:
            per_call = ACCESS_PATTERNS[name].bytes_per_call(source, self.dtype)
            batch = calls_per_sample(per_call, target)
            accounted += samples * batch * per_call
            calls += samples * batch
        matmul = self.matmul
        calls += samples
        flops += samples * 2 * matmul.m * matmul.n * matmul.k
        largest = self.stream.elements[-1]
        peak = max(
            len(ARRAY_NAMES) * largest * width + _check_bytes(largest, width),
            source * (2 * width + INDEX_BYTES) + _check_bytes(source, width),
            matmul_allocation_bytes(matmul.m, matmul.n, matmul.k, self.dtype),
        )
        return PlannedWork(
            peak_allocation_bytes=peak,
            accounted_bytes=accounted,
            calls=calls,
            flops=flops,
            measurements=len(self.stream.elements) * len(self.stream.kernels)
            + len(PATTERN_NAMES)
            + 1,
        )

    @property
    def element_bytes(self) -> int:
        """Width of the configured dtype, shared by every section."""

        return element_bytes(self.dtype)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "dtype": self.dtype,
            "seed": self.seed,
            "timing": self.timing.to_mapping(),
            "stream": self.stream.to_mapping(),
            "access": self.access.to_mapping(),
            "matmul": self.matmul.to_mapping(),
            "littles_law": self.littles_law.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> MemoryHierarchyConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise MemoryHierarchyError(
            f"config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    stream = _fields(config["stream"], {"kernels", "elements"}, "stream")
    return MemoryHierarchyConfig(
        name=config["name"],
        dtype=config["dtype"],
        seed=config["seed"],
        timing=TimingControls(**_fields(config["timing"], _TIMING_FIELDS, "timing")),
        stream=StreamControls(
            kernels=_list(stream["kernels"], "stream kernels"),
            elements=_list(stream["elements"], "stream elements"),
        ),
        access=AccessControls(
            **_fields(config["access"], {"elements", "stride_elements"}, "access")
        ),
        matmul=MatmulControls(**_fields(config["matmul"], {"m", "n", "k"}, "matmul")),
        littles_law=LatencyControls(
            **_fields(
                config["littles_law"], {"latency_ns", "latency_note"}, "littles_law"
            )
        ),
    )


def config_from_mapping(payload: object) -> MemoryHierarchyConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except TypeError as error:
        raise MemoryHierarchyError(
            f"invalid memory-hierarchy config: {error}"
        ) from error


def load_config(path: Path) -> MemoryHierarchyConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``MemoryHierarchyError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except ValueError as error:
        raise MemoryHierarchyError(str(error)) from error
    return config_from_mapping(payload)
