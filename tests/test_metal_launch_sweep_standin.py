"""A NumPy stand-in for the sweep runtime, held to what the development GPU did.

``StandInBackend`` replaces ``MetalLaunchSweepRuntime`` where MPS is missing.
It reports the pipeline limits that every sweep pipeline gave on the
development host (torch 2.15.0, macOS 27.2, arm64): 1024 threads per
threadgroup and an execution width of 32. Its probe returns the linear
layout, which the GPU matched for every swept shape. Its matmul is the
float64 product rounded once to float32, which is inside the bound that the
GPU result met, and the same for every shape, as the GPU result was. Reports
built on it are synthetic, never measurements.
"""

import numpy

from ai_journey.kernel_metal_reference import float32_sentinels, vector_add_reference
from ai_journey.metal_launch_sweep_layout import linear_layout
from ai_journey.metal_launch_sweep_runtime import PIPELINES, SweepWorkload
from ai_journey.metal_launch_sweep_source import MATMUL, SIMD_LAYOUT_PROBE
from test_kernel_metal_recorded import RECORDED_LIMITS, RecordedBackend
from test_kernel_metal_runtime import requires_metal


def rounded_product(a, b):
    """Return the float64 product rounded once to float32."""

    return (a.astype(numpy.float64) @ b.astype(numpy.float64)).astype(numpy.float32)


class StandInBackend(RecordedBackend):
    """A NumPy stand-in for ``MetalLaunchSweepRuntime`` with the host's limits."""

    def __init__(self, limits=None, *, requires_whole_simd_groups=False):
        super().__init__(limits)
        for name in (MATMUL, SIMD_LAYOUT_PROBE):
            self.limits[name] = limits or RECORDED_LIMITS
        self.requires_whole_simd_groups = requires_whole_simd_groups
        self.probed = []

    def simd_layout(self, group):
        self.probed.append(group)
        width = self.limits[SIMD_LAYOUT_PROBE].thread_execution_width
        return linear_layout(group, width)

    def matmul(self, a, b, launch):
        self.launches.append(launch)
        tail = float32_sentinels(launch.geometry.padding_threads)
        return numpy.concatenate([rounded_product(a, b).reshape(-1), tail])

    def matmul_workload(self, a, b, launches):
        self.launches.extend(launches)
        product = rounded_product(a, b).reshape(-1)
        return SweepWorkload(
            calls=tuple(lambda: None for _ in launches),
            synchronize=self.synchronize,
            outputs_agree=lambda: True,
            output=lambda: product.copy(),
        )

    def vector_workload(self, first, second, launches):
        self.launches.extend(launches)
        total = vector_add_reference(first, second)
        return SweepWorkload(
            calls=tuple(lambda: None for _ in launches),
            synchronize=self.synchronize,
            outputs_agree=lambda: True,
            output=lambda: total.copy(),
        )


def test_the_stand_in_reports_the_development_host_limits():
    backend = StandInBackend()

    assert {backend.limits[name] for name in PIPELINES} == {RECORDED_LIMITS}
    assert RECORDED_LIMITS.to_mapping() == {
        "max_threads_per_threadgroup": 1024,
        "thread_execution_width": 32,
        "static_threadgroup_memory_bytes": 0,
    }
    assert backend.requires_whole_simd_groups is False


@requires_metal
def test_the_stand_in_layout_is_the_one_this_gpu_gives():
    from test_metal_launch_sweep_runtime import SWEPT, sweep_runtime

    runtime = sweep_runtime()
    backend = StandInBackend()
    width = runtime.limits[SIMD_LAYOUT_PROBE].thread_execution_width
    assert width == backend.limits[SIMD_LAYOUT_PROBE].thread_execution_width
    for shape in (*SWEPT, (8,), (48,), (1024,)):
        expected = backend.simd_layout(shape)
        assert numpy.array_equal(runtime.simd_layout(shape), expected), shape
