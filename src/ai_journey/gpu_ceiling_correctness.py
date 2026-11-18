"""Correctness of every ceiling kernel against its exact CPU rebuild.

A ceiling means something only if the kernel did the work that the rate
credits to it, so every kernel runs on a small launch before anything is
timed. Each launch has threads past the data, and each output buffer starts
as quiet-NaN sentinels with a tail that no correct thread writes; the tail
must keep its sentinels. The rest of the buffer must match the CPU rebuild
bit for bit:

- an FMA kernel's words must equal the XOR of the exact fused chains;
- a copy must equal its source;
- a read-reduce must equal the exact integer partials of its walk;
- a bank kernel's words must equal the wrapping sums of the tile words read.

Each case records the digests of its input, its expected output and its
whole buffer, so a verifier can rebuild every fact from the seed. A case
that fails stops the run, so a report never holds a failed check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .gpu_ceiling_reference import (
    bank_lane_sums,
    copy_source,
    fma_chain_values,
    fma_lane_words,
    fma_start_values,
    lane_tile,
    read_reduce_partials,
    reduce_source,
    stream_threads,
    words_sha256,
)
from .gpu_ceiling_runtime import STREAM_GUARD_FLOAT4
from .gpu_ceiling_source import (
    STREAM_COPY,
    STREAM_KERNELS,
    GpuCeilingError,
    bank_kernel_name,
    fma_kernel_name,
)
from .kernel_metal_geometry import CheckedLaunch, vector_launch
from .kernel_metal_reference import FLOAT32_SENTINEL_BITS, array_sha256

# Correctness sources use case index 0; the benchmark draws index 1.
CORRECTNESS_INDEX = 0


@dataclass(frozen=True)
class FmaCase:
    """One FMA kernel on a guarded launch, equal to the exact fused chains."""

    chains: int
    fma_steps: int
    launch: CheckedLaunch
    start_sha256: str
    expected_sha256: str
    buffer_sha256: str

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "kernel": fma_kernel_name(self.chains),
            "chains": self.chains,
            "fma_steps": self.fma_steps,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "start_sha256": self.start_sha256,
            "expected_sha256": self.expected_sha256,
            "buffer_sha256": self.buffer_sha256,
            "guard_words": self.launch.geometry.padding_threads,
        }


@dataclass(frozen=True)
class StreamCase:
    """One copy or read-reduce on a guarded launch, equal to its rebuild."""

    kernel: str
    float4_count: int
    float4_per_thread: int
    launch: CheckedLaunch
    source_sha256: str
    expected_sha256: str
    buffer_sha256: str
    guard_words: int

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "kernel": self.kernel,
            "float4_count": self.float4_count,
            "float4_per_thread": self.float4_per_thread,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "source_sha256": self.source_sha256,
            "expected_sha256": self.expected_sha256,
            "buffer_sha256": self.buffer_sha256,
            "guard_words": self.guard_words,
        }


@dataclass(frozen=True)
class BankCase:
    """One bank kernel at one stride on a guarded launch, equal to the lane sums."""

    row_words: int
    stride: int
    trips: int
    launch: CheckedLaunch
    expected_sha256: str
    buffer_sha256: str

    def to_mapping(self) -> dict[str, object]:
        """Return the case as a report stores it."""

        return {
            "kernel": bank_kernel_name(self.row_words),
            "row_words": self.row_words,
            "stride": self.stride,
            "trips": self.trips,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "expected_sha256": self.expected_sha256,
            "buffer_sha256": self.buffer_sha256,
            "guard_words": self.launch.geometry.padding_threads,
        }


def _buffer(buffer: object, dtype: type, elements: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(buffer, numpy.ndarray)
        or buffer.dtype != dtype
        or buffer.shape != (elements,)
    ):
        raise GpuCeilingError(f"{label}: the kernel returned no {elements}-word buffer")
    return buffer


def _guarded(buffer: numpy.ndarray, used: int, label: str) -> None:
    """Require every word past ``used`` to keep its sentinel."""

    if numpy.any(buffer[used:].view(numpy.uint32) != FLOAT32_SENTINEL_BITS):
        raise GpuCeilingError(f"{label}: a thread wrote into the guard tail")


def check_fma(
    backend: object,
    *,
    seed: int,
    chains: int,
    threads: int,
    group_size: int,
    fma_steps: int,
) -> FmaCase:
    """Run one FMA kernel on a guarded launch and require the exact chains."""

    name = fma_kernel_name(chains)
    start = fma_start_values(seed)
    (values,) = fma_chain_values(start, (fma_steps,))
    expected = lane_tile(fma_lane_words(values, chains), threads)
    launch = CheckedLaunch(vector_launch(threads, group_size), backend.limits[name])
    label = f"{name} case ({threads} threads, {fma_steps} steps per chain)"
    length = threads + launch.geometry.padding_threads
    buffer = _buffer(
        backend.fma(chains, start, launch, fma_steps), numpy.uint32, length, label
    )
    if not numpy.array_equal(buffer[:threads], expected):
        raise GpuCeilingError(f"{label}: the kernel differs from the exact FMA chains")
    _guarded(buffer, threads, label)
    return FmaCase(
        chains=chains,
        fma_steps=fma_steps,
        launch=launch,
        start_sha256=array_sha256(start),
        expected_sha256=words_sha256(expected),
        buffer_sha256=words_sha256(buffer),
    )


def check_stream(
    backend: object,
    *,
    seed: int,
    kernel: str,
    float4_count: int,
    float4_per_thread: int,
    group_size: int,
) -> StreamCase:
    """Run one stream kernel on a guarded launch and require its exact rebuild."""

    if kernel not in STREAM_KERNELS:
        raise GpuCeilingError("a stream kernel is stream_copy or stream_read_reduce")
    copy = kernel == STREAM_COPY
    draw = copy_source if copy else reduce_source
    source = draw(seed, CORRECTNESS_INDEX, float4_count)
    walkers = stream_threads(float4_count, float4_per_thread)
    launch = CheckedLaunch(vector_launch(walkers, group_size), backend.limits[kernel])
    if copy:
        expected = source
        guard = 4 * STREAM_GUARD_FLOAT4
    else:
        expected = read_reduce_partials(source, walkers)
        guard = launch.geometry.padding_threads
    label = f"{kernel} case ({float4_count} float4 values)"
    buffer = _buffer(
        backend.stream(
            kernel,
            source,
            launch,
            float4_count=float4_count,
            per_thread=float4_per_thread,
        ),
        numpy.float32,
        expected.size + guard,
        label,
    )
    if not numpy.array_equal(
        buffer[: expected.size].view(numpy.uint32), expected.view(numpy.uint32)
    ):
        raise GpuCeilingError(f"{label}: the kernel differs from its CPU rebuild")
    _guarded(buffer, expected.size, label)
    return StreamCase(
        kernel=kernel,
        float4_count=float4_count,
        float4_per_thread=float4_per_thread,
        launch=launch,
        source_sha256=array_sha256(source),
        expected_sha256=array_sha256(expected),
        buffer_sha256=array_sha256(buffer),
        guard_words=guard,
    )


def check_bank(
    backend: object,
    *,
    row_words: int,
    stride: int,
    threads: int,
    group_size: int,
    trips: int,
) -> BankCase:
    """Run one bank kernel on a guarded launch and require the lane sums."""

    name = bank_kernel_name(row_words)
    expected = lane_tile(bank_lane_sums(stride, row_words, trips), threads)
    launch = CheckedLaunch(vector_launch(threads, group_size), backend.limits[name])
    label = f"{name} case (stride {stride}, {threads} threads)"
    length = threads + launch.geometry.padding_threads
    buffer = _buffer(
        backend.bank(row_words, stride, trips, launch), numpy.uint32, length, label
    )
    if not numpy.array_equal(buffer[:threads], expected):
        raise GpuCeilingError(f"{label}: the kernel differs from the lane sums")
    _guarded(buffer, threads, label)
    return BankCase(
        row_words=row_words,
        stride=stride,
        trips=trips,
        launch=launch,
        expected_sha256=words_sha256(expected),
        buffer_sha256=words_sha256(buffer),
    )
