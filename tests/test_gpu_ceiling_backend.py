"""A NumPy stand-in for ``CeilingRuntime``, checked against the real GPU.

``NumpyBackend`` returns what the development host's Apple GPU produced
(torch 2.15.0, macOS 27.2, arm64): every kernel matched its exact CPU
rebuild bit for bit, so the stand-in computes those rebuilds. Its pipeline
limits are the ones the compiled pipelines reported. Reports built on it are
synthetic, never measurements; the GPU tests below keep it honest.
"""

import numpy
import pytest

from ai_journey.gpu_ceiling_reference import (
    bank_lane_sums,
    copy_source,
    fma_chain_values,
    fma_lane_words,
    fma_start_values,
    lane_tile,
    read_reduce_partials,
    reduce_source,
    stream_threads,
)
from ai_journey.gpu_ceiling_runtime import STREAM_GUARD_FLOAT4, KernelWorkload
from ai_journey.gpu_ceiling_source import (
    BANK,
    KERNELS,
    STREAM_COPY,
    bank_tile_bytes,
)
from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits
from ai_journey.kernel_metal_reference import float32_sentinels
from test_gpu_ceiling_runtime import ceiling_runtime, launch
from test_kernel_metal_runtime import requires_metal

RECORDED_TORCH_VERSION = "2.15.0"


def recorded_limits(name):
    """The limits each compiled pipeline reported on the development host."""

    memory = 0
    if KERNELS[name].family == BANK:
        memory = bank_tile_bytes(int(name.rsplit("_", 1)[1]))
    return PipelineLimits(1024, 32, memory)


RECORDED_LIMITS = {name: recorded_limits(name) for name in KERNELS}


def sentinel_tail(count):
    return float32_sentinels(count).view(numpy.uint32)


class NumpyBackend:
    """Return the exact rebuild of each kernel, as the GPU did."""

    backend = "numpy-stand-in"
    torch_version = RECORDED_TORCH_VERSION

    def __init__(self):
        self.limits = dict(RECORDED_LIMITS)
        self.launches = []
        self.calls = 0

    def synchronize(self):
        pass

    def upload(self, array):
        return array

    def _admit(self, name, launch):
        assert isinstance(launch, CheckedLaunch)
        assert launch.limits == self.limits[name]
        self.launches.append((name, launch))

    def _fma_words(self, chains, start, launch, steps):
        self._admit(f"fma_chains_{chains}", launch)
        (values,) = fma_chain_values(start, (steps,))
        return lane_tile(fma_lane_words(values, chains), launch.geometry.extent[0])

    def _stream_values(self, name, source, launch, float4_count, per_thread):
        self._admit(name, launch)
        assert launch.geometry.extent == (stream_threads(float4_count, per_thread),)
        if name == STREAM_COPY:
            return source[: 4 * float4_count].copy()
        return read_reduce_partials(
            source[: 4 * float4_count], launch.geometry.extent[0]
        )

    def _bank_words(self, row_words, stride, trips, launch):
        self._admit(f"bank_rows_{row_words}", launch)
        return lane_tile(
            bank_lane_sums(stride, row_words, trips), launch.geometry.extent[0]
        )

    def fma(self, chains, start, launch, steps):
        words = self._fma_words(chains, start, launch, steps)
        return numpy.concatenate(
            [words, sentinel_tail(launch.geometry.padding_threads)]
        )

    def stream(self, name, source, launch, *, float4_count, per_thread):
        values = self._stream_values(name, source, launch, float4_count, per_thread)
        guard = (
            4 * STREAM_GUARD_FLOAT4
            if name == STREAM_COPY
            else launch.geometry.padding_threads
        )
        return numpy.concatenate([values, float32_sentinels(guard)])

    def bank(self, row_words, stride, trips, launch):
        words = self._bank_words(row_words, stride, trips, launch)
        return numpy.concatenate(
            [words, sentinel_tail(launch.geometry.padding_threads)]
        )

    def _workload(self, output):
        def run():
            self.calls += 1

        return KernelWorkload(
            run=run, synchronize=self.synchronize, output=lambda: output
        )

    def fma_workload(self, chains, start, launch, steps):
        return self._workload(self._fma_words(chains, start, launch, steps))

    def stream_workload(self, name, source, launch, *, float4_count, per_thread):
        values = self._stream_values(name, source, launch, float4_count, per_thread)
        return self._workload(values)

    def bank_workload(self, row_words, stride, trips, launch):
        return self._workload(self._bank_words(row_words, stride, trips, launch))


@requires_metal
def test_the_stand_in_reports_the_limits_of_this_hosts_pipelines():
    assert ceiling_runtime().limits == RECORDED_LIMITS
    assert ceiling_runtime().torch_version == RECORDED_TORCH_VERSION


@requires_metal
@pytest.mark.parametrize("chains", [1, 8, 32])
def test_the_stand_in_returns_the_gpus_fma_buffers(chains):
    runtime, stand_in = ceiling_runtime(), NumpyBackend()
    start = fma_start_values(7)
    name = f"fma_chains_{chains}"

    gpu = runtime.fma(chains, start, launch(runtime, name, 777, 128), 40)
    cpu = stand_in.fma(chains, start, launch(stand_in, name, 777, 128), 40)

    assert numpy.array_equal(gpu, cpu)


@requires_metal
@pytest.mark.parametrize(
    ("name", "per_thread"), [("stream_copy", 3), ("stream_read_reduce", 5)]
)
def test_the_stand_in_returns_the_gpus_stream_buffers(name, per_thread):
    runtime, stand_in = ceiling_runtime(), NumpyBackend()
    draw = copy_source if name == "stream_copy" else reduce_source
    source = draw(7, 3, 1001)
    walkers = stream_threads(1001, per_thread)
    arguments = {"float4_count": 1001, "per_thread": per_thread}

    gpu = runtime.stream(name, source, launch(runtime, name, walkers), **arguments)
    cpu = stand_in.stream(name, source, launch(stand_in, name, walkers), **arguments)

    assert numpy.array_equal(gpu.view(numpy.uint32), cpu.view(numpy.uint32))


@requires_metal
@pytest.mark.parametrize("row_words", [32, 33])
def test_the_stand_in_returns_the_gpus_bank_buffers(row_words):
    runtime, stand_in = ceiling_runtime(), NumpyBackend()
    name = f"bank_rows_{row_words}"

    for stride in (2, 5, 16):
        gpu = runtime.bank(row_words, stride, 65, launch(runtime, name, 999, 64))
        cpu = stand_in.bank(row_words, stride, 65, launch(stand_in, name, 999, 64))
        assert numpy.array_equal(gpu, cpu), stride
