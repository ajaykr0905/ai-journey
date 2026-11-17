"""Compile the sweep kernels on the MPS device beside the Day 73 Metal kernels.

``MetalLaunchSweepRuntime`` is the Day 73 ``MetalRuntime`` with one more
library. Its constructor runs the Day 73 probe and compile first, so a host
without macOS, torch, an MPS device or ``torch.mps.compile_shader`` stops
with the Day 73 reason before anything here runs, and importing this module
never imports torch. It then compiles ``SWEEP_SOURCE`` and reads the limits of
each new pipeline. The 1D sweep runs the Day 73 ``vector_add`` through the
Day 73 launch binding, so its admission check and 32-bit casts stay the Day
73 ones.

``requires_whole_simd_groups`` is ``False`` for this runtime. The PyTorch
2.15.0 library builds its pipelines with
``newComputePipelineStateWithFunction:error:``, which takes no descriptor, and
the library holds no ``threadGroupSizeIsMultipleOfThreadExecutionWidth``
selector. That option thus keeps its default, so no pipeline requires a
threadgroup of whole SIMD groups. On the development host, groups of 8, 16
and 48 threads ran and gave exact outputs.

Workloads keep their buffers on the device and bind one call per admitted
shape, so a timed sample only queues kernels. Every shape writes its own
output, so a check after the timing can compare all shapes bit for bit.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    LaunchGeometry,
    PipelineLimits,
)
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import KernelMetalUnavailableError, MetalRuntime
from .kernel_metal_source import VECTOR_ADD
from .metal_launch_sweep_shapes import group_shape
from .metal_launch_sweep_source import (
    LAYOUT_FIELDS,
    MATMUL,
    SIMD_LAYOUT_PROBE,
    SWEEP_KERNELS,
    SWEEP_SOURCE,
    matmul_arg_casts,
)

# The pipelines a sweep reads, in the order a report lists them.
PIPELINES = (MATMUL, VECTOR_ADD, SIMD_LAYOUT_PROBE)
LAYOUT_SENTINEL = 0xFFFFFFFF


def pipeline_limits(kernel: object) -> PipelineLimits:
    """Read the limits one compiled pipeline reports about itself."""

    return PipelineLimits(
        max_threads_per_threadgroup=kernel.max_threads_per_threadgroup,
        thread_execution_width=kernel.thread_execution_width,
        static_threadgroup_memory_bytes=kernel.static_thread_group_memory_length,
    )


@dataclass(frozen=True)
class SweepWorkload:
    """One sweep point's device buffers, bound as one call per admitted shape.

    Each call only queues its kernel, and ``synchronize`` waits for it.
    ``outputs_agree`` waits, then reports whether the latest output of every
    shape is bit-identical to the first. ``output`` waits, then copies the
    first shape's output to the host.
    """

    calls: tuple[Callable[[], object], ...]
    synchronize: Callable[[], None]
    outputs_agree: Callable[[], bool]
    output: Callable[[], numpy.ndarray]


def _matrix(values: object, label: str) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 2
    ):
        raise KernelMetalError(f"{label} must be a 2D float32 array")
    return numpy.ascontiguousarray(values)


class MetalLaunchSweepRuntime(MetalRuntime):
    """The Day 73 runtime plus the sweep library, with NumPy arrays in and out."""

    requires_whole_simd_groups = False

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        super().__init__(system=system, import_module=import_module)
        try:
            library = self._torch.mps.compile_shader(SWEEP_SOURCE)
        except RuntimeError as error:
            raise KernelMetalUnavailableError(
                f"the Metal compiler rejected the sweep source: {error}"
            ) from error
        self._sweep = library
        for name in SWEEP_KERNELS:
            self.limits[name] = pipeline_limits(getattr(library, name))

    def _admit_sweep(
        self, name: str, launch: object, extent: tuple[int, ...]
    ) -> CheckedLaunch:
        """Accept only a launch of this extent checked against the kernel's pipeline."""

        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[name]
            or launch.geometry.extent != extent
        ):
            raise KernelMetalError(
                f"{name} runs only with a launch of extent {list(extent)} checked "
                "against its own pipeline"
            )
        return launch

    def simd_layout(self, group: tuple[int, ...]) -> numpy.ndarray:
        """Run the probe as one threadgroup of ``group``; return a word row per thread.

        The buffer starts as sentinels, so a thread that never ran shows.
        """

        shape = group_shape(group)
        launch = CheckedLaunch(
            LaunchGeometry(extent=shape, group_size=shape),
            self.limits[SIMD_LAYOUT_PROBE],
        )
        words = len(LAYOUT_FIELDS) * launch.geometry.threads_per_group
        output = self._to(
            numpy.full(words, LAYOUT_SENTINEL, dtype=numpy.uint32).view(numpy.int32)
        )
        getattr(self._sweep, SIMD_LAYOUT_PROBE)(
            output, **launch.geometry.launch_arguments()
        )
        self.synchronize()
        return self._host(output).view(numpy.uint32).reshape(-1, len(LAYOUT_FIELDS))

    def _matmul_call(
        self,
        launch: CheckedLaunch,
        output: object,
        left: object,
        right: object,
        m: int,
        n: int,
        k: int,
    ) -> partial:
        return partial(
            getattr(self._sweep, MATMUL),
            output,
            left,
            right,
            m,
            n,
            k,
            **launch.geometry.launch_arguments(),
            arg_casts=matmul_arg_casts(),
        )

    def matmul(
        self, a: numpy.ndarray, b: numpy.ndarray, launch: CheckedLaunch
    ) -> numpy.ndarray:
        """Run the kernel once and return its whole flat output buffer.

        The buffer is longer than ``C`` by the launch's padding threads and
        starts as sentinels, so a guard failure past ``C`` shows in its tail.
        """

        left, right = _matrix(a, "a"), _matrix(b, "b")
        (m, k), (inner, n) = left.shape, right.shape
        if inner != k:
            raise KernelMetalError("a and b must be conformable")
        self._admit_sweep(MATMUL, launch, (n, m))
        output = self._to(float32_sentinels(m * n + launch.geometry.padding_threads))
        call = self._matmul_call(
            launch, output, self._to(left), self._to(right), m, n, k
        )
        call()
        self.synchronize()
        return self._host(output)

    def _workload(self, outputs: list[object], calls: list[partial]) -> SweepWorkload:
        torch = self._torch

        def outputs_agree() -> bool:
            self.synchronize()
            first = outputs[0].view(torch.int32)
            return all(torch.equal(first, other.view(torch.int32)) for other in outputs)

        def output() -> numpy.ndarray:
            self.synchronize()
            return self._host(outputs[0])

        return SweepWorkload(
            calls=tuple(calls),
            synchronize=self.synchronize,
            outputs_agree=outputs_agree,
            output=output,
        )

    def matmul_workload(
        self,
        a: numpy.ndarray,
        b: numpy.ndarray,
        launches: tuple[CheckedLaunch, ...],
    ) -> SweepWorkload:
        """Place one point's matrices on the device and bind one call per shape."""

        left, right = _matrix(a, "a"), _matrix(b, "b")
        (m, k), (inner, n) = left.shape, right.shape
        if inner != k or not launches:
            raise KernelMetalError("a matmul workload needs conformable inputs")
        for launch in launches:
            self._admit_sweep(MATMUL, launch, (n, m))
        device_a, device_b = self._to(left), self._to(right)
        outputs = [
            self._torch.empty(m * n, dtype=self._torch.float32, device="mps")
            for _ in launches
        ]
        calls = [
            self._matmul_call(launch, output, device_a, device_b, m, n, k)
            for launch, output in zip(launches, outputs)
        ]
        return self._workload(outputs, calls)

    def vector_workload(
        self,
        first: numpy.ndarray,
        second: numpy.ndarray,
        launches: tuple[CheckedLaunch, ...],
    ) -> SweepWorkload:
        """Bind the Day 73 ``vector_add`` once per group size on shared inputs."""

        if not launches:
            raise KernelMetalError("a vector workload needs at least one launch")
        length = launches[0].geometry.extent[0]
        for launch in launches:
            self._admit(VECTOR_ADD, launch)
            if launch.geometry.extent != (length,):
                raise KernelMetalError("every launch of a point covers one length")
        if not all(
            isinstance(values, numpy.ndarray)
            and values.dtype == numpy.float32
            and values.shape == (length,)
            for values in (first, second)
        ):
            raise KernelMetalError("vector inputs must be float32 of the launch length")
        left, right = self._to(first), self._to(second)
        outputs = [self._torch.empty_like(left) for _ in launches]
        calls = [
            self._kernel(VECTOR_ADD, launch, output, left, right, length)
            for launch, output in zip(launches, outputs)
        ]
        return self._workload(outputs, calls)
