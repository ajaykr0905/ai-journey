"""Observations recorded on the development host's Apple GPU, replayed on any host.

``RecordedBackend`` stands in for ``MatmulMetalRuntime`` where MPS is missing.
Its naive kernels return the CPU replay of the fma chain with a sentinel
guard tail, which is what the GPU produced on 2026-11-17 (torch 2.15.0,
macOS 27.2, arm64): every naive output matched the replay bit for bit. Its
PyTorch outputs come from NumPy's float32 matmul, which the GPU did not
produce; they only need to stay within the bound. Reports built on it are
synthetic, never measurements.
"""

import numpy

from ai_journey.kernel_metal_geometry import CheckedLaunch, PipelineLimits
from ai_journey.kernel_metal_reference import FLOAT32_SENTINEL_BITS, float32_sentinels
from ai_journey.matmul_metal_mapping import X_ROWS, matmul_launch, matmul_shape
from ai_journey.matmul_metal_reference import (
    CORRECTNESS_STREAM,
    fma_replay,
    matmul_inputs,
)
from ai_journey.matmul_metal_runtime import SIDES, MatmulWorkload, WorkloadCheck
from ai_journey.matmul_metal_source import KERNEL_FOR_MAPPING, KERNELS
from test_kernel_metal_runtime import requires_metal
from test_matmul_metal_runtime import launches, matmul_runtime

# Both pipelines reported these limits.
RECORDED_LIMITS = PipelineLimits(
    max_threads_per_threadgroup=1024,
    thread_execution_width=32,
    static_threadgroup_memory_bytes=0,
)
RECORDED_TORCH_VERSION = "2.15.0"
RECORDED_MATMUL_PRECISION = "highest"


class RecordedBackend:
    """A NumPy stand-in for ``MatmulMetalRuntime``; its reports are synthetic."""

    backend = "recorded"
    torch_version = RECORDED_TORCH_VERSION
    float32_matmul_precision = RECORDED_MATMUL_PRECISION

    def __init__(self, limits=None):
        self.limits = {name: limits or RECORDED_LIMITS for name in KERNELS}
        self.launches = []

    def synchronize(self):
        pass

    def run(self, name, left, right, launch):
        self.launches.append((name, launch))
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([fma_replay(left, right).reshape(-1), tail])

    def torch_matmul(self, left, right, device):
        return left @ right

    def workload(self, left, right, launches):
        self.launches.extend(launches.items())

        def side():
            return left @ right

        return MatmulWorkload(
            sides={name: side for name in SIDES},
            synchronize=self.synchronize,
            check=lambda: WorkloadCheck(True, 0.0),
        )


def test_the_stand_in_returns_the_replay_with_a_sentinel_tail():
    backend = RecordedBackend()
    shape = matmul_shape(17, 33, 9)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 0, shape)
    name = KERNEL_FOR_MAPPING[X_ROWS]
    launch = CheckedLaunch(matmul_launch(X_ROWS, shape, (32, 8)), RECORDED_LIMITS)

    buffer = backend.run(name, left, right, launch)

    replay = fma_replay(left, right).reshape(-1)
    assert numpy.array_equal(buffer[: replay.size], replay)
    tail = buffer[replay.size :].view(numpy.uint32)
    assert tail.size == launch.geometry.padding_threads == 32 * 40 - 17 * 33
    assert set(tail.tolist()) == {FLOAT32_SENTINEL_BITS}
    assert backend.launches == [(name, launch)]


@requires_metal
def test_the_stand_in_matches_this_gpu_bit_for_bit():
    runtime = matmul_runtime()
    backend = RecordedBackend()
    shape = matmul_shape(31, 47, 257)
    left, right = matmul_inputs(74, CORRECTNESS_STREAM, 5, shape)

    for name, launch in launches(runtime, shape).items():
        expected = runtime.run(name, left, right, launch).view(numpy.uint32)
        recorded = backend.run(name, left, right, launch).view(numpy.uint32)
        assert numpy.array_equal(recorded, expected), name
