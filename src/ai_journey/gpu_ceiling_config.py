"""Strict, bounded controls for one ceiling run on this host's Apple GPU.

A config names the seed, the correctness cases, the three sweeps and the
timing controls. Parsing rejects unknown and missing fields at every level,
booleans posing as integers, and out-of-range values. The config is also
checked as a whole before anything runs:

- every correctness launch has threads past the data, so each bounds guard
  runs before anything is timed;
- FMA step counts are multiples of the eight steps of one loop trip, and
  there are at least three, so the analysis can fit the fixed cost of a call;
- bank strides start at 1, the baseline of every ratio, and bank launches
  are whole 32-lane SIMD groups with no idle padding threads;
- the planned calls, FLOPs, bytes and loads fit the work bounds.

Files are read through the bounded strict reader of
``memory_access_report_io``, which also rejects duplicate keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .gpu_ceiling_benchmark import (
    bank_accesses_per_call,
    fma_flops_per_call,
    stream_bytes_per_call,
)
from .gpu_ceiling_reference import (
    MAX_BANK_TRIPS,
    MAX_FLOAT4_PER_THREAD,
    MAX_FMA_STEPS,
    MAX_STREAM_FLOAT4,
    stream_threads,
)
from .gpu_ceiling_source import (
    BANK_LANES,
    BANK_MAX_STRIDE,
    BANK_ROW_WORDS,
    FLOAT4_BYTES,
    FMA_CHAIN_COUNTS,
    FMA_STEPS_PER_TRIP,
    STREAM_KERNELS,
    STREAM_READ_REDUCE,
    GpuCeilingError,
    require_count,
)
from .kernel_metal_benchmark import (
    MAX_RUNTIME_SECONDS,
    MAX_TARGET_BYTES,
    MIN_REPEATS,
    MIN_TARGET_BYTES,
    calls_per_sample,
)
from .kernel_metal_geometry import MAX_THREADGROUP_THREADS, KernelMetalError
from .kernel_metal_reference import MAX_SEED
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP
from .memory_access_plan import canonical_json_bytes
from .memory_access_report_io import read_json_object

CONFIG_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 64 * 1024
MAX_THREADS = 1 << 24
MAX_CORRECTNESS_THREADS = 1 << 16
MAX_CORRECTNESS_FLOAT4 = 1 << 20
MIN_SCALING_POINTS = 3
MAX_STEP_POINTS = 8
MAX_BUFFER_SIZES = 8
MAX_RUN_CALLS = 1 << 22
MAX_RUN_FLOPS = 1 << 46
MAX_RUN_BYTES = 1 << 38
MAX_RUN_ACCESSES = 1 << 40

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIELDS = {
    "schema_version",
    "name",
    "seed",
    "correctness",
    "fma",
    "bandwidth",
    "bank",
    "timing",
}
_CORRECTNESS_FIELDS = {"threads", "fma_steps", "stream_float4", "bank_trips"}
_FMA_FIELDS = {"group_size", "threads", "chain_counts", "fma_steps"}
_BANDWIDTH_FIELDS = {
    "group_size",
    "buffer_bytes",
    "copy_float4_per_thread",
    "reduce_float4_per_thread",
    "target_bytes_per_sample",
}
_BANK_FIELDS = {"group_size", "threads", "strides", "trips"}
_TIMING_FIELDS = {"warmup", "repeats", "max_runtime_seconds"}


def _fields(payload: object, expected: set[str], label: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != expected:
        raise GpuCeilingError(f"{label} fields do not match the schema")
    return payload


def _list(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise GpuCeilingError(f"{label} must be a JSON array")
    return tuple(value)


def _increasing(
    values: object, label: str, minimum: int, maximum: int, most: int, least: int = 1
) -> tuple[int, ...]:
    """Require ``least`` to ``most`` strictly increasing integers in bounds."""

    if type(values) is not tuple or not least <= len(values) <= most:
        raise GpuCeilingError(f"{label} must hold {least} to {most} values")
    for value in values:
        require_count(value, label, minimum, maximum)
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise GpuCeilingError(f"{label} must be strictly increasing")
    return values


def _group(value: object, label: str) -> int:
    return require_count(value, label, 1, MAX_THREADGROUP_THREADS)


@dataclass(frozen=True)
class CorrectnessControls:
    """The small guarded launch every kernel passes before anything is timed."""

    threads: int
    fma_steps: int
    stream_float4: int
    bank_trips: int

    def __post_init__(self) -> None:
        require_count(self.threads, "correctness threads", 1, MAX_CORRECTNESS_THREADS)
        require_count(self.fma_steps, "correctness fma_steps", 1, MAX_FMA_STEPS)
        if self.fma_steps % FMA_STEPS_PER_TRIP:
            raise GpuCeilingError(
                f"correctness fma_steps must be a multiple of {FMA_STEPS_PER_TRIP}"
            )
        require_count(
            self.stream_float4, "correctness stream_float4", 1, MAX_CORRECTNESS_FLOAT4
        )
        require_count(self.bank_trips, "correctness bank_trips", 1, MAX_BANK_TRIPS)

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it."""

        return {
            "threads": self.threads,
            "fma_steps": self.fma_steps,
            "stream_float4": self.stream_float4,
            "bank_trips": self.bank_trips,
        }


@dataclass(frozen=True)
class FmaControls:
    """The compute sweep: every chain count at every step count per chain."""

    group_size: int
    threads: int
    chain_counts: tuple[int, ...]
    fma_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        _group(self.group_size, "fma group_size")
        require_count(self.threads, "fma threads", 1, MAX_THREADS)
        chains = _increasing(
            self.chain_counts, "fma chain_counts", 1, 32, len(FMA_CHAIN_COUNTS)
        )
        if not set(chains) <= set(FMA_CHAIN_COUNTS):
            raise GpuCeilingError(
                "fma chain_counts must come from "
                + ", ".join(map(str, FMA_CHAIN_COUNTS))
            )
        steps = _increasing(
            self.fma_steps,
            "fma fma_steps",
            1,
            MAX_FMA_STEPS,
            MAX_STEP_POINTS,
            MIN_SCALING_POINTS,
        )
        if any(step % FMA_STEPS_PER_TRIP for step in steps):
            raise GpuCeilingError(
                f"fma fma_steps must be multiples of {FMA_STEPS_PER_TRIP}"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "group_size": self.group_size,
            "threads": self.threads,
            "chain_counts": list(self.chain_counts),
            "fma_steps": list(self.fma_steps),
        }


@dataclass(frozen=True)
class BandwidthControls:
    """The stream sweep: buffer sizes, the walk of each kernel, the batch target."""

    group_size: int
    buffer_bytes: tuple[int, ...]
    copy_float4_per_thread: int
    reduce_float4_per_thread: int
    target_bytes_per_sample: int

    def __post_init__(self) -> None:
        _group(self.group_size, "bandwidth group_size")
        sizes = _increasing(
            self.buffer_bytes,
            "bandwidth buffer_bytes",
            FLOAT4_BYTES,
            FLOAT4_BYTES * MAX_STREAM_FLOAT4,
            MAX_BUFFER_SIZES,
        )
        if any(size % FLOAT4_BYTES for size in sizes):
            raise GpuCeilingError(
                "bandwidth buffer_bytes must be whole float4 values, multiples of 16"
            )
        for label in ("copy_float4_per_thread", "reduce_float4_per_thread"):
            require_count(getattr(self, label), label, 1, MAX_FLOAT4_PER_THREAD)
        require_count(
            self.target_bytes_per_sample,
            "target_bytes_per_sample",
            MIN_TARGET_BYTES,
            MAX_TARGET_BYTES,
        )

    @property
    def float4_counts(self) -> tuple[int, ...]:
        """Each buffer size in float4 values."""

        return tuple(size // FLOAT4_BYTES for size in self.buffer_bytes)

    def per_thread(self, kernel: str) -> int:
        """Float4 values in one thread's walk for one stream kernel."""

        if kernel not in STREAM_KERNELS:
            raise GpuCeilingError(
                "a stream kernel is stream_copy or stream_read_reduce"
            )
        if kernel == STREAM_READ_REDUCE:
            return self.reduce_float4_per_thread
        return self.copy_float4_per_thread

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "group_size": self.group_size,
            "buffer_bytes": list(self.buffer_bytes),
            "copy_float4_per_thread": self.copy_float4_per_thread,
            "reduce_float4_per_thread": self.reduce_float4_per_thread,
            "target_bytes_per_sample": self.target_bytes_per_sample,
        }


@dataclass(frozen=True)
class BankControls:
    """The bank sweep: lane strides from 1, each with and without padding."""

    group_size: int
    threads: int
    strides: tuple[int, ...]
    trips: int

    def __post_init__(self) -> None:
        group = _group(self.group_size, "bank group_size")
        if group % BANK_LANES:
            raise GpuCeilingError(
                f"bank group_size must be a multiple of {BANK_LANES}, whole SIMD groups"
            )
        threads = require_count(self.threads, "bank threads", 1, MAX_THREADS)
        if threads % group:
            raise GpuCeilingError(
                "bank threads must be whole threadgroups, so no padding thread "
                "does uncounted loads"
            )
        strides = _increasing(
            self.strides, "bank strides", 1, BANK_MAX_STRIDE, BANK_MAX_STRIDE, 2
        )
        if strides[0] != 1:
            raise GpuCeilingError("bank strides must start at 1, the baseline")
        require_count(self.trips, "bank trips", 1, MAX_BANK_TRIPS)

    def to_mapping(self) -> dict[str, object]:
        """Return the section as the config file spells it."""

        return {
            "group_size": self.group_size,
            "threads": self.threads,
            "strides": list(self.strides),
            "trips": self.trips,
        }


@dataclass(frozen=True)
class TimingControls:
    """Untimed warmups, timed repeats per point, and the run's time budget."""

    warmup: int
    repeats: int
    max_runtime_seconds: int

    def __post_init__(self) -> None:
        require_count(self.warmup, "warmup", 0, MAX_WARMUP)
        require_count(self.repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
        require_count(
            self.max_runtime_seconds, "max_runtime_seconds", 1, MAX_RUNTIME_SECONDS
        )

    def to_mapping(self) -> dict[str, int]:
        """Return the section as the config file spells it."""

        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass(frozen=True)
class PlannedWork:
    """What a config asks of the host, counted before anything runs."""

    correctness_cases: int
    fma_points: int
    stream_points: int
    bank_points: int
    timed_samples: int
    calls: int
    flops: int
    stream_bytes: int
    bank_accesses: int

    def to_mapping(self) -> dict[str, int]:
        """Return the counts a report records beside its measurements."""

        return {
            "correctness_cases": self.correctness_cases,
            "fma_points": self.fma_points,
            "stream_points": self.stream_points,
            "bank_points": self.bank_points,
            "timed_samples": self.timed_samples,
            "calls": self.calls,
            "flops": self.flops,
            "stream_bytes": self.stream_bytes,
            "bank_accesses": self.bank_accesses,
        }


@dataclass(frozen=True)
class GpuCeilingConfig:
    """Every control one run reads, validated as a whole."""

    name: str
    seed: int
    correctness: CorrectnessControls
    fma: FmaControls
    bandwidth: BandwidthControls
    bank: BankControls
    timing: TimingControls

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise GpuCeilingError("config name must be a lowercase slug")
        require_count(self.seed, "seed", 0, MAX_SEED)
        for value, kind in (
            (self.correctness, CorrectnessControls),
            (self.fma, FmaControls),
            (self.bandwidth, BandwidthControls),
            (self.bank, BankControls),
            (self.timing, TimingControls),
        ):
            if not isinstance(value, kind):
                raise TypeError(f"config section must be {kind.__name__}")
        threads = self.correctness.threads
        for label, group in (
            ("fma", self.fma.group_size),
            ("bank", self.bank.group_size),
        ):
            if threads % group == 0:
                raise GpuCeilingError(
                    f"correctness threads must not be a multiple of the {label} "
                    f"group size {group}, so the guard runs"
                )
        walkers = stream_threads(
            self.correctness.stream_float4, self.bandwidth.reduce_float4_per_thread
        )
        if walkers % self.bandwidth.group_size == 0:
            raise GpuCeilingError(
                "correctness stream_float4 must leave read-reduce threads past the "
                "data in the last bandwidth threadgroup"
            )
        work = self.planned_work()
        for value, limit, label in (
            (work.calls, MAX_RUN_CALLS, "kernel calls"),
            (work.flops, MAX_RUN_FLOPS, "FLOPs"),
            (work.stream_bytes, MAX_RUN_BYTES, "stream bytes"),
            (work.bank_accesses, MAX_RUN_ACCESSES, "threadgroup loads"),
        ):
            if value > limit:
                raise GpuCeilingError(
                    f"config needs {value} {label}; the bound is {limit}"
                )

    def stream_batches(self) -> tuple[tuple[str, int, int, int], ...]:
        """Return ``(kernel, float4 count, bytes per call, calls per sample)``.

        The batch sizes are the run's, so a point that needs an oversized
        sample fails here, before anything is launched.
        """

        bandwidth = self.bandwidth
        rows = []
        for kernel in STREAM_KERNELS:
            for count in bandwidth.float4_counts:
                per_call = stream_bytes_per_call(
                    kernel, count, bandwidth.per_thread(kernel)
                )
                try:
                    batch = calls_per_sample(
                        per_call, bandwidth.target_bytes_per_sample
                    )
                except KernelMetalError as error:
                    raise GpuCeilingError(
                        f"{kernel} over {count} float4: {error}"
                    ) from error
                rows.append((kernel, count, per_call, batch))
        return tuple(rows)

    def planned_work(self) -> PlannedWork:
        """Count the cases, points, samples, calls and work of the run up front."""

        fma, bank = self.fma, self.bank
        samples = self.timing.warmup + self.timing.repeats
        fma_points = len(fma.chain_counts) * len(fma.fma_steps)
        bank_points = len(BANK_ROW_WORDS) * len(bank.strides)
        batches = self.stream_batches()
        flops = sum(
            fma_flops_per_call(fma.threads, chains, steps)
            for chains in fma.chain_counts
            for steps in fma.fma_steps
        )
        return PlannedWork(
            correctness_cases=len(fma.chain_counts)
            + len(STREAM_KERNELS)
            + len(BANK_ROW_WORDS) * len(bank.strides),
            fma_points=fma_points,
            stream_points=len(batches),
            bank_points=bank_points,
            timed_samples=self.timing.repeats
            * (fma_points + len(batches) + bank_points),
            calls=samples * (fma_points + bank_points + sum(row[3] for row in batches)),
            flops=samples * flops,
            stream_bytes=samples * sum(row[2] * row[3] for row in batches),
            bank_accesses=samples
            * bank_points
            * bank_accesses_per_call(bank.threads, bank.trips),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical mapping a report embeds and its digest covers."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "correctness": self.correctness.to_mapping(),
            "fma": self.fma.to_mapping(),
            "bandwidth": self.bandwidth.to_mapping(),
            "bank": self.bank.to_mapping(),
            "timing": self.timing.to_mapping(),
        }

    def fingerprint(self) -> str:
        """Bind every control to one SHA-256 digest of its canonical JSON."""

        return sha256(canonical_json_bytes(self.to_mapping())).hexdigest()


def _config(payload: object) -> GpuCeilingConfig:
    config = _fields(payload, _FIELDS, "config")
    version = config["schema_version"]
    if type(version) is not int or version != CONFIG_SCHEMA_VERSION:
        raise GpuCeilingError(f"config schema_version must be {CONFIG_SCHEMA_VERSION}")
    correctness = _fields(config["correctness"], _CORRECTNESS_FIELDS, "correctness")
    fma = _fields(config["fma"], _FMA_FIELDS, "fma")
    bandwidth = _fields(config["bandwidth"], _BANDWIDTH_FIELDS, "bandwidth")
    bank = _fields(config["bank"], _BANK_FIELDS, "bank")
    timing = _fields(config["timing"], _TIMING_FIELDS, "timing")
    return GpuCeilingConfig(
        name=config["name"],
        seed=config["seed"],
        correctness=CorrectnessControls(**correctness),
        fma=FmaControls(
            group_size=fma["group_size"],
            threads=fma["threads"],
            chain_counts=_list(fma["chain_counts"], "fma chain_counts"),
            fma_steps=_list(fma["fma_steps"], "fma fma_steps"),
        ),
        bandwidth=BandwidthControls(
            group_size=bandwidth["group_size"],
            buffer_bytes=_list(bandwidth["buffer_bytes"], "bandwidth buffer_bytes"),
            copy_float4_per_thread=bandwidth["copy_float4_per_thread"],
            reduce_float4_per_thread=bandwidth["reduce_float4_per_thread"],
            target_bytes_per_sample=bandwidth["target_bytes_per_sample"],
        ),
        bank=BankControls(
            group_size=bank["group_size"],
            threads=bank["threads"],
            strides=_list(bank["strides"], "bank strides"),
            trips=bank["trips"],
        ),
        timing=TimingControls(**timing),
    )


def config_from_mapping(payload: object) -> GpuCeilingConfig:
    """Parse controls, reporting every contract violation as one error type."""

    try:
        return _config(payload)
    except GpuCeilingError:
        raise
    except (TypeError, KernelMetalError) as error:
        raise GpuCeilingError(f"invalid gpu-ceiling config: {error}") from error


def load_config(path: Path) -> GpuCeilingConfig:
    """Read one bounded, duplicate-free JSON config file and validate it.

    The shared reader opens only regular non-symlink files, stops at the byte
    bound, and rejects duplicate keys, non-finite constants and non-object
    roots; its errors are reported as ``GpuCeilingError``.
    """

    try:
        payload = read_json_object(path, max_bytes=MAX_CONFIG_BYTES, label="config")
    except GpuCeilingError:
        raise
    except ValueError as error:
        raise GpuCeilingError(str(error)) from error
    return config_from_mapping(payload)
