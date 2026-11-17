"""Run the ceiling kernels through PyTorch's MPS backend, importing torch on demand.

Importing this module never imports torch, so a host without torch can still
verify reports. ``CeilingRuntime`` asks the Day 73 probe, ``probe_metal``,
which reads the operating system before it imports torch and names the first
missing requirement. It compiles ``SHADER_SOURCE`` once, reads each
pipeline's limits, and runs a kernel only with a one-axis launch checked
against that kernel's own pipeline. The bank pattern assumes SIMD groups of
32 lanes, so a bank pipeline that reports another width is refused.

Each check method runs one call and returns the whole output buffer as a
NumPy array: output buffers start as quiet-NaN sentinels and carry a tail
that no correct thread writes. Each workload method places one sweep point's
buffers on the device and binds the call, so the timer only launches and
synchronizes. Only ``CeilingRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy

from .gpu_ceiling_reference import FMA_COEFFICIENTS, stream_threads
from .gpu_ceiling_source import (
    BANK,
    BANK_LANES,
    BANK_MAX_STRIDE,
    FMA_LANES,
    FMA_START_COLUMNS,
    FMA_STEPS_PER_TRIP,
    KERNELS,
    SHADER_SOURCE,
    STREAM_COPY,
    STREAM_KERNELS,
    GpuCeilingError,
    bank_kernel_name,
    fma_kernel_name,
    require_count,
)
from .kernel_metal_geometry import CheckedLaunch, PipelineLimits
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import BACKEND, probe_metal

# A copy writes no float4 at or past n; this tail would show one that did.
STREAM_GUARD_FLOAT4 = 32
MAX_TRIPS = 1 << 16


class GpuCeilingUnavailableError(GpuCeilingError):
    """Raised when this host cannot compile or run the ceiling kernels."""


@dataclass(frozen=True)
class KernelWorkload:
    """One sweep point's device buffers, bound as no-argument calls.

    ``run`` only queues one call and ``synchronize`` waits for it. ``output``
    waits too, then copies the latest output to the host for its check.
    """

    run: Callable[[], object]
    synchronize: Callable[[], None]
    output: Callable[[], numpy.ndarray]


def _require_start(start: object) -> numpy.ndarray:
    if (
        not isinstance(start, numpy.ndarray)
        or start.dtype != numpy.float32
        or start.shape != (FMA_LANES, FMA_START_COLUMNS)
    ):
        raise GpuCeilingError("start must be the 32 x 32 float32 start table")
    return numpy.ascontiguousarray(start)


def _require_source(source: object, float4_count: int) -> numpy.ndarray:
    if (
        not isinstance(source, numpy.ndarray)
        or source.dtype != numpy.float32
        or source.ndim != 1
        or source.size < 4 * float4_count
    ):
        raise GpuCeilingError("source must be a flat float32 array of the count")
    return numpy.ascontiguousarray(source)


def _trips(fma_steps: int) -> int:
    require_count(fma_steps, "fma_steps", 1, MAX_TRIPS * FMA_STEPS_PER_TRIP)
    if fma_steps % FMA_STEPS_PER_TRIP:
        raise GpuCeilingError(
            f"fma_steps must be a multiple of {FMA_STEPS_PER_TRIP}, the steps "
            "of one loop trip"
        )
    return fma_steps // FMA_STEPS_PER_TRIP


class CeilingRuntime:
    """The compiled ceiling kernels on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise GpuCeilingUnavailableError(availability.message)
        torch = import_module("torch")
        try:
            library = torch.mps.compile_shader(SHADER_SOURCE)
        except RuntimeError as error:
            raise GpuCeilingUnavailableError(
                f"the Metal compiler rejected the ceiling kernels: {error}"
            ) from error
        self._torch = torch
        self._library = library
        self.torch_version = availability.torch_version
        self.limits = {name: self._pipeline_limits(name) for name in KERNELS}

    def _pipeline_limits(self, name: str) -> PipelineLimits:
        kernel = getattr(self._library, name)
        return PipelineLimits(
            max_threads_per_threadgroup=kernel.max_threads_per_threadgroup,
            thread_execution_width=kernel.thread_execution_width,
            static_threadgroup_memory_bytes=kernel.static_thread_group_memory_length,
        )

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def upload(self, array: numpy.ndarray) -> object:
        """Copy a NumPy array to the MPS device once, for many calls."""

        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")

    def _output(self, count: int, *, guarded: bool) -> object:
        """Allocate an output: sentinels for a check, uninitialized for timing."""

        if guarded:
            return self.upload(float32_sentinels(count))
        return self._torch.empty(count, dtype=self._torch.float32, device="mps")

    def _words(self, tensor: object) -> numpy.ndarray:
        return tensor.cpu().numpy().view(numpy.uint32)

    def _floats(self, tensor: object) -> numpy.ndarray:
        return tensor.cpu().numpy().view(numpy.float32).reshape(-1)

    def _admit(self, name: str, launch: object) -> CheckedLaunch:
        """Accept only a 1D launch checked against this kernel's own pipeline."""

        limits = self.limits[name]
        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != limits
            or len(launch.geometry.extent) != 1
        ):
            raise GpuCeilingError(
                f"{name} runs only with a 1D launch checked against its own pipeline"
            )
        if KERNELS[name].family == BANK and (
            limits.thread_execution_width != BANK_LANES
            or launch.geometry.threads_per_group % BANK_LANES
        ):
            raise GpuCeilingError(
                f"{name} needs {BANK_LANES}-lane SIMD groups and a threadgroup "
                f"size that is a multiple of {BANK_LANES}"
            )
        return launch

    def _kernel(self, name: str, launch: CheckedLaunch, *arguments: object) -> partial:
        """Bind one admitted launch with its geometry and 32-bit scalar casts."""

        return partial(
            getattr(self._library, name),
            *arguments,
            **launch.geometry.launch_arguments(),
            arg_casts=KERNELS[name].arg_casts,
        )

    def _fma_call(
        self,
        chains: int,
        start: numpy.ndarray,
        launch: CheckedLaunch,
        steps: int,
        *,
        guarded: bool,
    ) -> tuple[partial, object]:
        name = fma_kernel_name(chains)
        (threads,) = self._admit(name, launch).geometry.extent
        length = threads + launch.geometry.padding_threads if guarded else threads
        output = self._output(length, guarded=guarded)
        coefficients = numpy.array(FMA_COEFFICIENTS, dtype=numpy.float32)
        call = self._kernel(
            name,
            launch,
            output,
            self.upload(_require_start(start)),
            self.upload(coefficients),
            threads,
            _trips(steps),
        )
        return call, output

    def fma(
        self, chains: int, start: numpy.ndarray, launch: CheckedLaunch, steps: int
    ) -> numpy.ndarray:
        """Run one FMA kernel once; return every output word, padding included."""

        call, output = self._fma_call(chains, start, launch, steps, guarded=True)
        call()
        self.synchronize()
        return self._words(output)

    def fma_workload(
        self, chains: int, start: numpy.ndarray, launch: CheckedLaunch, steps: int
    ) -> KernelWorkload:
        """Bind one FMA sweep point; its output holds only the active threads."""

        call, output = self._fma_call(chains, start, launch, steps, guarded=False)

        def latest() -> numpy.ndarray:
            self.synchronize()
            return self._words(output)

        return KernelWorkload(run=call, synchronize=self.synchronize, output=latest)

    def _stream_call(
        self,
        name: str,
        source: object,
        float4_count: int,
        launch: CheckedLaunch,
        *,
        guarded: bool,
    ) -> tuple[partial, object]:
        if name not in STREAM_KERNELS:
            raise GpuCeilingError(
                "a stream kernel is stream_copy or stream_read_reduce"
            )
        (threads,) = self._admit(name, launch).geometry.extent
        if name == STREAM_COPY:
            length = 4 * (float4_count + (STREAM_GUARD_FLOAT4 if guarded else 0))
        else:
            length = threads + (launch.geometry.padding_threads if guarded else 0)
        output = self._output(length, guarded=guarded)
        call = self._kernel(name, launch, output, source, float4_count, threads)
        return call, output

    def _check_walk(self, float4_count: int, per_thread: int, launch: object) -> None:
        """Require ``ceil(count / per_thread)`` active threads, the CPU's walk."""

        walkers = stream_threads(float4_count, per_thread)
        if isinstance(launch, CheckedLaunch) and launch.geometry.extent != (walkers,):
            raise GpuCeilingError(
                "a stream launch needs ceil(count / per_thread) active threads"
            )

    def stream(
        self,
        name: str,
        source: numpy.ndarray,
        launch: CheckedLaunch,
        *,
        float4_count: int,
        per_thread: int,
    ) -> numpy.ndarray:
        """Run one stream kernel once; return its whole float32 output buffer.

        A copy returns ``float4_count`` values plus the guard tail; a
        read-reduce returns one partial per launched thread.
        """

        self._check_walk(float4_count, per_thread, launch)
        values = self.upload(_require_source(source, float4_count))
        call, output = self._stream_call(
            name, values, float4_count, launch, guarded=True
        )
        call()
        self.synchronize()
        return self._floats(output)

    def stream_workload(
        self,
        name: str,
        source: object,
        launch: CheckedLaunch,
        *,
        float4_count: int,
        per_thread: int,
    ) -> KernelWorkload:
        """Bind one stream sweep point to a source already on the device.

        Sizes share one uploaded source: a kernel reads only its first
        ``float4_count`` values.
        """

        self._check_walk(float4_count, per_thread, launch)
        call, output = self._stream_call(
            name, source, float4_count, launch, guarded=False
        )

        def latest() -> numpy.ndarray:
            self.synchronize()
            return self._floats(output)

        return KernelWorkload(run=call, synchronize=self.synchronize, output=latest)

    def _bank_call(
        self,
        row_words: int,
        stride: int,
        trips: int,
        launch: CheckedLaunch,
        *,
        guarded: bool,
    ) -> tuple[partial, object]:
        name = bank_kernel_name(row_words)
        (threads,) = self._admit(name, launch).geometry.extent
        require_count(stride, "stride", 1, BANK_MAX_STRIDE)
        require_count(trips, "trips", 1, MAX_TRIPS)
        length = threads + launch.geometry.padding_threads if guarded else threads
        output = self._output(length, guarded=guarded)
        call = self._kernel(name, launch, output, threads, stride, trips)
        return call, output

    def bank(
        self, row_words: int, stride: int, trips: int, launch: CheckedLaunch
    ) -> numpy.ndarray:
        """Run one bank kernel once; return every output word, padding included."""

        call, output = self._bank_call(row_words, stride, trips, launch, guarded=True)
        call()
        self.synchronize()
        return self._words(output)

    def bank_workload(
        self, row_words: int, stride: int, trips: int, launch: CheckedLaunch
    ) -> KernelWorkload:
        """Bind one bank sweep point; its output holds only the active threads."""

        call, output = self._bank_call(row_words, stride, trips, launch, guarded=False)

        def latest() -> numpy.ndarray:
            self.synchronize()
            return self._words(output)

        return KernelWorkload(run=call, synchronize=self.synchronize, output=latest)
