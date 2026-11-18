"""Synchronized, rotated wall-clock timing of the three ceiling sweeps.

A GPU launch returns once its command is queued, so each timed sample calls
``synchronize`` before its first clock reading and again before its second,
as the Day 73 Metal benchmark does, and its clock reader, budget and timing
records are reused here. Every point first runs ``warmup`` untimed samples.
Timed samples then rotate over all points of a sweep, and the order reverses
every repeat, so slow drift in clocks, heat or load spreads over the points
instead of favoring one. After the last sample, each point's latest output
must match its exact CPU rebuild, so every rate counts work that was done.

An FMA or bank sample holds one call that is large next to the fixed cost of
a launch and a synchronize; the analysis fits that cost from the FMA sweep.
A stream sample holds the fewest whole calls whose compulsory bytes reach
``target_bytes_per_sample``, with the Day 73 rule. GFLOP/s counts 2 FLOPs
per FMA. GB/s counts compulsory bytes: a copy reads and writes each byte of
its buffer, and a read-reduce reads it once and writes one float32 partial
per thread. A bank point divides its median sample by the loads of all its
active threads.

Only the host timer ``time.perf_counter_ns`` makes a measurement; any other
clock is named ``injected`` and its durations are synthetic. These are
host-specific Apple GPU (Metal) timings; nothing here measures CUDA or
NVIDIA hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat

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
from .gpu_ceiling_source import (
    BANK_LOADS_PER_TRIP,
    FLOAT4_BYTES,
    FLOAT32_BYTES,
    FMA_FLOPS_PER_STEP,
    STREAM_COPY,
    STREAM_KERNELS,
    STREAM_READ_REDUCE,
    GpuCeilingError,
    bank_kernel_name,
    fma_kernel_name,
    require_count,
)
from .kernel_metal_benchmark import (
    MAX_CALLS_PER_SAMPLE,
    MIN_REPEATS,
    DeviceTiming,
    RunBudget,
    _Reader,
    calls_per_sample,
)
from .kernel_metal_geometry import CheckedLaunch, vector_launch
from .kernel_metal_reference import array_sha256
from .memory_access_benchmark import MAX_REPEATS, MAX_WARMUP, timer_name

# The benchmark draws stream sources with case index 1; checks use 0.
BENCHMARK_INDEX = 1
MAX_POINTS = 64
PS_PER_NS = 1000


def fma_flops_per_call(threads: int, chains: int, fma_steps: int) -> int:
    """Two FLOPs for each FMA step of each chain of each active thread."""

    return FMA_FLOPS_PER_STEP * threads * chains * fma_steps


def stream_bytes_per_call(kernel: str, float4_count: int, per_thread: int) -> int:
    """Compulsory bytes: a copy reads and writes; a read-reduce reads, then
    writes one float32 partial per thread."""

    if kernel not in STREAM_KERNELS:
        raise GpuCeilingError("a stream kernel is stream_copy or stream_read_reduce")
    data = FLOAT4_BYTES * float4_count
    if kernel == STREAM_COPY:
        return 2 * data
    return data + FLOAT32_BYTES * stream_threads(float4_count, per_thread)


def bank_accesses_per_call(threads: int, trips: int) -> int:
    """Threadgroup loads of all active threads in one call."""

    return threads * trips * BANK_LOADS_PER_TRIP


def time_rotation(
    workloads: tuple[object, ...],
    *,
    labels: tuple[str, ...],
    calls: tuple[int, ...],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[DeviceTiming, ...]:
    """Time every point in rotated, synchronized samples; one timing per point."""

    if (
        type(workloads) is not tuple
        or not 1 <= len(workloads) <= MAX_POINTS
        or type(labels) is not tuple
        or type(calls) is not tuple
        or not len(workloads) == len(labels) == len(calls)
    ):
        raise GpuCeilingError(
            f"a sweep needs 1 to {MAX_POINTS} workloads, each with a label and calls"
        )
    batches = [
        require_count(count, "calls_per_sample", 1, MAX_CALLS_PER_SAMPLE)
        for count in calls
    ]
    warm = require_count(warmup, "warmup", 0, MAX_WARMUP)
    count = require_count(repeats, "repeats", MIN_REPEATS, MAX_REPEATS)
    if budget is not None and not isinstance(budget, RunBudget):
        raise TypeError("budget must be a RunBudget")
    read = _Reader(clock, "benchmark")

    def sample(index: int) -> None:
        workload = workloads[index]
        for _ in repeat(None, batches[index]):
            workload.run()

    def checkpoint() -> None:
        if budget is not None:
            budget.check()

    for index, workload in enumerate(workloads):
        for _ in range(warm):
            workload.synchronize()
            sample(index)
            workload.synchronize()
            checkpoint()
    samples: list[list[int]] = [[] for _ in workloads]
    order = range(len(workloads))
    for round_index in range(count):
        for index in order if round_index % 2 == 0 else reversed(order):
            workload = workloads[index]
            workload.synchronize()
            started = read()
            sample(index)
            workload.synchronize()
            elapsed = read() - started
            if elapsed <= 0:
                raise GpuCeilingError("benchmark clock did not advance across a sample")
            samples[index].append(elapsed)
            checkpoint()
    timer = timer_name(clock)
    return tuple(
        DeviceTiming(label, batch, warm, count, tuple(times), timer)
        for label, batch, times in zip(labels, batches, samples)
    )


def _timing_mapping(timing: DeviceTiming) -> dict[str, object]:
    """Return the raw samples beside their statistics, so both can be checked."""

    summary = timing.summary
    calls = timing.calls_per_sample
    return {
        "calls_per_sample": calls,
        "warmup": timing.warmup,
        "repeats": timing.repeats,
        "samples_ns": list(timing.samples_ns),
        "timing": summary.to_mapping(),
        "median_ns_per_call": summary.median_ns / calls,
        "iqr_ns_per_call": summary.iqr_ns / calls,
    }


def _rated(timing: DeviceTiming, work_per_call: int, unit: str) -> dict[str, object]:
    """Add the median and best rates, in work units per nanosecond."""

    work = work_per_call * timing.calls_per_sample
    summary = timing.summary
    return {
        **_timing_mapping(timing),
        f"median_{unit}": work / summary.median_ns,
        f"best_{unit}": work / summary.min_ns,
    }


@dataclass(frozen=True)
class FmaPoint:
    """One chain count and step count, timed and checked against the CPU."""

    chains: int
    fma_steps: int
    launch: CheckedLaunch
    timing: DeviceTiming
    output_sha256: str

    @property
    def flops_per_call(self) -> int:
        """FLOPs of one call, two per FMA."""

        return fma_flops_per_call(
            self.launch.geometry.active_threads, self.chains, self.fma_steps
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics and derived rates."""

        return {
            "kernel": fma_kernel_name(self.chains),
            "chains": self.chains,
            "fma_steps": self.fma_steps,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "flops_per_call": self.flops_per_call,
            **_rated(self.timing, self.flops_per_call, "gflops_per_second"),
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class StreamPoint:
    """One stream kernel at one buffer size, timed and checked against the CPU."""

    kernel: str
    float4_count: int
    float4_per_thread: int
    launch: CheckedLaunch
    timing: DeviceTiming
    output_sha256: str

    @property
    def bytes_per_call(self) -> int:
        """Compulsory bytes of one call."""

        return stream_bytes_per_call(
            self.kernel, self.float4_count, self.float4_per_thread
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics and derived rates."""

        rated = _rated(self.timing, self.bytes_per_call, "gb_per_second")
        return {
            "kernel": self.kernel,
            "buffer_bytes": FLOAT4_BYTES * self.float4_count,
            "float4_count": self.float4_count,
            "float4_per_thread": self.float4_per_thread,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "bytes_per_call": self.bytes_per_call,
            "bytes_per_sample": self.bytes_per_call * self.timing.calls_per_sample,
            **rated,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class BankPoint:
    """One row length and lane stride, timed and checked against the CPU."""

    row_words: int
    stride: int
    trips: int
    launch: CheckedLaunch
    timing: DeviceTiming
    output_sha256: str

    @property
    def accesses_per_call(self) -> int:
        """Threadgroup loads of one call."""

        return bank_accesses_per_call(self.launch.geometry.active_threads, self.trips)

    def to_mapping(self) -> dict[str, object]:
        """Return the point with raw samples, statistics and time per access."""

        accesses = self.accesses_per_call * self.timing.calls_per_sample
        return {
            "kernel": bank_kernel_name(self.row_words),
            "row_words": self.row_words,
            "stride": self.stride,
            "trips": self.trips,
            "threads": self.launch.geometry.active_threads,
            "launch": self.launch.to_mapping(),
            "accesses_per_call": self.accesses_per_call,
            **_timing_mapping(self.timing),
            "median_ps_per_access": self.timing.summary.median_ns
            * PS_PER_NS
            / accesses,
            "output_sha256": self.output_sha256,
        }


def _checked_output(workload: object, expected: numpy.ndarray, label: str) -> str:
    """Require the latest timed output to equal its rebuild; return its digest."""

    output = workload.output()
    if (
        not isinstance(output, numpy.ndarray)
        or output.dtype != expected.dtype
        or output.shape != expected.shape
        or not numpy.array_equal(output.view(numpy.uint32), expected.view(numpy.uint32))
    ):
        raise GpuCeilingError(f"{label}: the timed output differs from its CPU rebuild")
    if output.dtype == numpy.uint32:
        return words_sha256(output)
    return array_sha256(output)


def benchmark_fma(
    backend: object,
    *,
    seed: int,
    threads: int,
    group_size: int,
    chain_counts: tuple[int, ...],
    fma_steps: tuple[int, ...],
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[FmaPoint, ...]:
    """Time every chain count at every step count, one call per sample."""

    start = fma_start_values(seed)
    tables = fma_chain_values(start, fma_steps)
    plan = [(chains, steps) for chains in chain_counts for steps in fma_steps]
    launches, workloads, expected = [], [], []
    for chains, steps in plan:
        name = fma_kernel_name(chains)
        launch = CheckedLaunch(vector_launch(threads, group_size), backend.limits[name])
        launches.append(launch)
        workloads.append(backend.fma_workload(chains, start, launch, steps))
        table = tables[fma_steps.index(steps)]
        expected.append(lane_tile(fma_lane_words(table, chains), threads))
    timings = time_rotation(
        tuple(workloads),
        labels=tuple(fma_kernel_name(chains) for chains, _ in plan),
        calls=(1,) * len(plan),
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    points = []
    for (chains, steps), launch, workload, words, timing in zip(
        plan, launches, workloads, expected, timings
    ):
        label = f"{fma_kernel_name(chains)} at {steps} steps per chain"
        points.append(
            FmaPoint(
                chains=chains,
                fma_steps=steps,
                launch=launch,
                timing=timing,
                output_sha256=_checked_output(workload, words, label),
            )
        )
    return tuple(points)


def benchmark_streams(
    backend: object,
    *,
    seed: int,
    group_size: int,
    float4_counts: tuple[int, ...],
    copy_float4_per_thread: int,
    reduce_float4_per_thread: int,
    target_bytes_per_sample: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[StreamPoint, ...]:
    """Time the copy and the read-reduce at every size, on shared sources.

    Each kernel reads a prefix of one uploaded source of the largest size,
    so the sweep holds two sources on the device, not one per point.
    """

    largest = max(float4_counts)
    per_thread = {
        STREAM_COPY: copy_float4_per_thread,
        STREAM_READ_REDUCE: reduce_float4_per_thread,
    }
    sources = {
        STREAM_COPY: copy_source(seed, BENCHMARK_INDEX, largest),
        STREAM_READ_REDUCE: reduce_source(seed, BENCHMARK_INDEX, largest),
    }
    plan = [(kernel, count) for kernel in STREAM_KERNELS for count in float4_counts]
    uploaded = {kernel: backend.upload(source) for kernel, source in sources.items()}
    launches, workloads, batches = [], [], []
    for kernel, count in plan:
        walkers = stream_threads(count, per_thread[kernel])
        launch = CheckedLaunch(
            vector_launch(walkers, group_size), backend.limits[kernel]
        )
        launches.append(launch)
        workloads.append(
            backend.stream_workload(
                kernel,
                uploaded[kernel],
                launch,
                float4_count=count,
                per_thread=per_thread[kernel],
            )
        )
        bytes_per_call = stream_bytes_per_call(kernel, count, per_thread[kernel])
        batches.append(calls_per_sample(bytes_per_call, target_bytes_per_sample))
    timings = time_rotation(
        tuple(workloads),
        labels=tuple(kernel for kernel, _ in plan),
        calls=tuple(batches),
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    points = []
    for (kernel, count), launch, workload, timing in zip(
        plan, launches, workloads, timings
    ):
        prefix = sources[kernel][: 4 * count]
        if kernel == STREAM_COPY:
            expected = prefix
        else:
            expected = read_reduce_partials(prefix, launch.geometry.active_threads)
        label = f"{kernel} over {count} float4 values"
        points.append(
            StreamPoint(
                kernel=kernel,
                float4_count=count,
                float4_per_thread=per_thread[kernel],
                launch=launch,
                timing=timing,
                output_sha256=_checked_output(workload, expected, label),
            )
        )
    return tuple(points)


def benchmark_banks(
    backend: object,
    *,
    threads: int,
    group_size: int,
    row_words: tuple[int, ...],
    strides: tuple[int, ...],
    trips: int,
    warmup: int,
    repeats: int,
    clock: Callable[[], int] = time.perf_counter_ns,
    budget: RunBudget | None = None,
) -> tuple[BankPoint, ...]:
    """Time each row length at each lane stride, one call per sample."""

    plan = [(words, stride) for words in row_words for stride in strides]
    launches, workloads = [], []
    for words, stride in plan:
        name = bank_kernel_name(words)
        launch = CheckedLaunch(vector_launch(threads, group_size), backend.limits[name])
        launches.append(launch)
        workloads.append(backend.bank_workload(words, stride, trips, launch))
    timings = time_rotation(
        tuple(workloads),
        labels=tuple(bank_kernel_name(words) for words, _ in plan),
        calls=(1,) * len(plan),
        warmup=warmup,
        repeats=repeats,
        clock=clock,
        budget=budget,
    )
    points = []
    for (words, stride), launch, workload, timing in zip(
        plan, launches, workloads, timings
    ):
        expected = lane_tile(bank_lane_sums(stride, words, trips), threads)
        label = f"{bank_kernel_name(words)} at stride {stride}"
        points.append(
            BankPoint(
                row_words=words,
                stride=stride,
                trips=trips,
                launch=launch,
                timing=timing,
                output_sha256=_checked_output(workload, expected, label),
            )
        )
    return tuple(points)
