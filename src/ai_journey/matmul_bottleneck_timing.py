"""The small declared schema of one-call Metal matmul timings.

The Day 74 matmul-metal report times batches of back-to-back calls, so its
median per call is a steady-state time that hides the fixed cost of one call.
The bottleneck rule also needs that fixed cost, so it reads one more input:
for each shape and kernel, the medians of three one-call times.

* ``median_gpu_ns``: the GPU time of one call, for example between two
  ``torch.mps.Event`` records around it. Such an event keeps a completion
  time that the host records, so the interval holds the signal latency too;
* ``median_host_ns``: the host wall-clock time of one call, from the moment
  the queue is idle until the host sees the call complete;
* ``median_dispatch_ns``: the host time to issue that call, before it waits.

A sealed gpu-timing report is the other real input, through
``matmul_bottleneck_gpu_timing``; the recorded input here came from a
scratch capture. An input is a measurement only with ``measured: true``, the
Metal measurement label and macOS in its environment, and it records the
load averages of its run. It need not be canonical JSON, because another
tool writes it, so its digest covers the canonical form of the parsed
mapping. Nothing here imports torch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .kernel_metal_geometry import KernelMetalError
from .kernel_metal_report import ENVIRONMENT_FIELDS, MEASUREMENT_LABEL, SYNTHETIC_LABEL
from .matmul_bottleneck_config import MatmulBottleneckError
from .matmul_metal_mapping import matmul_shape
from .matmul_metal_runtime import SIDES
from .memory_access_benchmark import MAX_REPEATS
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object
from .memory_access_tiling import MatmulShape

TIMING_SCHEMA_VERSION = 1
TIMING_KIND = "matmul_gpu_timing"
MAX_TIMING_BYTES = 256 * 1024
MAX_TIMING_POINTS = 96
MAX_PRODUCER_CHARS = 300
# About 18 minutes in nanoseconds: far above any one call of the lab.
MAX_CALL_NS = 1 << 40
TIMES = ("median_gpu_ns", "median_host_ns", "median_dispatch_ns")

_TOKEN = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_PRODUCER = re.compile(r"[ -~]+")
_FIELDS = {
    "schema_version",
    "kind",
    "measured",
    "label",
    "producer",
    "environment",
    "repeats",
    "load_average_before",
    "load_average_after",
    "points",
}
_POINT_FIELDS = {"m", "n", "k", "kernel", *TIMES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatmulBottleneckError(message)


def _loads(value: object, name: str) -> tuple[float, float, float]:
    _require(
        type(value) is list
        and len(value) == 3
        and all(
            type(item) is float and math.isfinite(item) and item >= 0 for item in value
        ),
        f"timing {name} must be three nonnegative floats",
    )
    return tuple(value)


@dataclass(frozen=True)
class TimingEntry:
    """The three one-call medians of one kernel at one shape, in nanoseconds."""

    shape: MatmulShape
    kernel: str
    gpu_ns: int
    host_ns: int
    dispatch_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, MatmulShape):
            raise TypeError("shape must be a MatmulShape")
        _require(self.kernel in SIDES, f"timing kernel must be one of {SIDES}")
        for name, value in zip(TIMES, (self.gpu_ns, self.host_ns, self.dispatch_ns)):
            _require(
                type(value) is int and 1 <= value <= MAX_CALL_NS,
                f"timing {name} must be an integer from 1 to {MAX_CALL_NS}",
            )

    @property
    def key(self) -> tuple[int, int, int, str]:
        return self.shape.m, self.shape.n, self.shape.k, self.kernel

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.shape.to_mapping(),
            "kernel": self.kernel,
            "median_gpu_ns": self.gpu_ns,
            "median_host_ns": self.host_ns,
            "median_dispatch_ns": self.dispatch_ns,
        }


@dataclass(frozen=True)
class TimingInput:
    """One parsed timing input: its labels, environment, loads and entries."""

    measured: bool
    producer: str
    environment: tuple[tuple[str, str], ...]
    repeats: int
    load_average_before: tuple[float, float, float]
    load_average_after: tuple[float, float, float]
    entries: tuple[TimingEntry, ...]

    def __post_init__(self) -> None:
        _require(type(self.measured) is bool, "timing measured must be a boolean")
        _require(
            type(self.producer) is str
            and len(self.producer) <= MAX_PRODUCER_CHARS
            and _PRODUCER.fullmatch(self.producer) is not None,
            f"timing producer must be 1 to {MAX_PRODUCER_CHARS} printable ASCII "
            "characters",
        )
        environment = dict(self.environment)
        _require(
            tuple(name for name, _ in self.environment) == ENVIRONMENT_FIELDS
            and all(
                type(value) is str and _TOKEN.fullmatch(value) is not None
                for value in environment.values()
            ),
            "timing environment must hold the Metal labs' tokens",
        )
        _require(
            not self.measured or environment["platform_system"] == "Darwin",
            "a timing measurement must come from macOS",
        )
        _require(
            type(self.repeats) is int and 1 <= self.repeats <= MAX_REPEATS,
            f"timing repeats must be an integer from 1 to {MAX_REPEATS}",
        )
        _require(
            type(self.entries) is tuple
            and 1 <= len(self.entries) <= MAX_TIMING_POINTS
            and all(type(entry) is TimingEntry for entry in self.entries),
            f"timing points must hold 1 to {MAX_TIMING_POINTS} entries",
        )
        keys = [entry.key for entry in self.entries]
        _require(len(set(keys)) == len(keys), "timing points repeat a kernel and shape")

    @property
    def label(self) -> str:
        return MEASUREMENT_LABEL if self.measured else SYNTHETIC_LABEL

    def entry(self, shape: MatmulShape, kernel: str) -> TimingEntry | None:
        """Return the entry of one kernel at one shape, or ``None``."""

        for entry in self.entries:
            if entry.shape == shape and entry.kernel == kernel:
                return entry
        return None

    def fastest_call(self, kernel: str) -> TimingEntry | None:
        """Return the entry with the smallest one-call host time of a kernel.

        A cost that every call of the kernel pays, whatever its shape, cannot
        exceed that time. The first entry wins a tie.
        """

        entries = [entry for entry in self.entries if entry.kernel == kernel]
        return min(entries, key=lambda entry: entry.host_ns) if entries else None

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": TIMING_SCHEMA_VERSION,
            "kind": TIMING_KIND,
            "measured": self.measured,
            "label": self.label,
            "producer": self.producer,
            "environment": dict(self.environment),
            "repeats": self.repeats,
            "load_average_before": list(self.load_average_before),
            "load_average_after": list(self.load_average_after),
            "points": [entry.to_mapping() for entry in self.entries],
        }

    def digest(self) -> str:
        """Hash the canonical JSON of the parsed input, whatever its file layout."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _entry(point: object, index: int) -> TimingEntry:
    _require(
        type(point) is dict and set(point) == _POINT_FIELDS,
        f"timing point {index} fields do not match the schema",
    )
    try:
        shape = matmul_shape(point["m"], point["n"], point["k"])
    except KernelMetalError as error:
        raise MatmulBottleneckError(f"timing point {index}: {error}") from error
    return TimingEntry(
        shape=shape,
        kernel=point["kernel"],
        gpu_ns=point["median_gpu_ns"],
        host_ns=point["median_host_ns"],
        dispatch_ns=point["median_dispatch_ns"],
    )


def _timing(payload: object) -> TimingInput:
    _require(
        type(payload) is dict and set(payload) == _FIELDS,
        "timing fields do not match the schema",
    )
    _require(
        type(payload["schema_version"]) is int
        and payload["schema_version"] == TIMING_SCHEMA_VERSION,
        f"timing schema_version must be {TIMING_SCHEMA_VERSION}",
    )
    _require(payload["kind"] == TIMING_KIND, f"timing kind must be {TIMING_KIND}")
    measured = payload["measured"]
    _require(type(measured) is bool, "timing measured must be a boolean")
    _require(
        payload["label"] == (MEASUREMENT_LABEL if measured else SYNTHETIC_LABEL),
        "timing label does not match whether it measured",
    )
    environment = payload["environment"]
    _require(
        type(environment) is dict and set(environment) == set(ENVIRONMENT_FIELDS),
        "timing environment fields do not match the schema",
    )
    points = payload["points"]
    _require(
        type(points) is list and 1 <= len(points) <= MAX_TIMING_POINTS,
        f"timing points must hold 1 to {MAX_TIMING_POINTS} entries",
    )
    return TimingInput(
        measured=measured,
        producer=payload["producer"],
        environment=tuple((name, environment[name]) for name in ENVIRONMENT_FIELDS),
        repeats=payload["repeats"],
        load_average_before=_loads(
            payload["load_average_before"], "load_average_before"
        ),
        load_average_after=_loads(payload["load_average_after"], "load_average_after"),
        entries=tuple(_entry(point, index) for index, point in enumerate(points)),
    )


def timing_from_mapping(payload: object) -> TimingInput:
    """Parse one timing input, reporting every violation as one error type."""

    try:
        return _timing(payload)
    except TypeError as error:
        raise MatmulBottleneckError(f"invalid timing input: {error}") from error


def load_timing(path: Path) -> TimingInput:
    """Read one bounded, duplicate-free timing file and parse it."""

    try:
        payload = read_json_object(path, max_bytes=MAX_TIMING_BYTES, label="timing")
    except ValueError as error:
        raise MatmulBottleneckError(str(error)) from error
    return timing_from_mapping(payload)
