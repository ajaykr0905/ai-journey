"""The Day 74 matmul runtime plus an empty kernel and MPS event pairs.

``GpuTimingRuntime`` builds the Day 74 ``MatmulMetalRuntime``, so the naive
kernels, their launch checks and the ``torch.matmul`` side are the same code
that Day 74 timed. It adds two things that the three clocks need:

- the near-empty kernel, compiled from its own small library, as one more
  side with a launch checked against its own pipeline;
- ``event_pair``, a start and end ``torch.mps.Event(enable_timing=True)``
  that the event clock records around one launch.

Importing this module never imports torch, so a host without torch can
still verify reports; the Day 74 probe names the first missing requirement,
and the runtime reports it as ``GpuTimingUnavailableError``. Only
``GpuTimingRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable, Mapping
from functools import partial

import numpy

from .gpu_timing_clocks import GpuTimingError
from .gpu_timing_source import EMPTY_KERNEL, EMPTY_SOURCE, empty_launch
from .kernel_metal_geometry import CheckedLaunch, PipelineLimits
from .matmul_metal_runtime import (
    MatmulMetalRuntime,
    MatmulMetalUnavailableError,
    MatmulWorkload,
)


class GpuTimingUnavailableError(GpuTimingError):
    """Raised when this host cannot compile or run the timed kernels."""


class EventPair:
    """Two timing events on the MPS stream, recorded around one launch."""

    def __init__(self, torch: object) -> None:
        self._start = torch.mps.Event(enable_timing=True)
        self._end = torch.mps.Event(enable_timing=True)

    def start(self) -> None:
        """Record the start event just before a launch."""

        self._start.record()

    def stop(self) -> None:
        """Record the end event just after a launch."""

        self._end.record()

    def elapsed_ms(self) -> float:
        """Return the interval in ms; call only after a synchronize."""

        return float(self._start.elapsed_time(self._end))


class GpuTimingRuntime:
    """The compiled timed kernels on the MPS device, NumPy arrays in."""

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        try:
            matmul = MatmulMetalRuntime(system=system, import_module=import_module)
        except MatmulMetalUnavailableError as error:
            raise GpuTimingUnavailableError(str(error)) from error
        torch = import_module("torch")
        try:
            library = torch.mps.compile_shader(EMPTY_SOURCE)
        except RuntimeError as error:
            raise GpuTimingUnavailableError(
                f"the Metal compiler rejected the empty kernel: {error}"
            ) from error
        kernel = getattr(library, EMPTY_KERNEL)
        self._torch = torch
        self._matmul = matmul
        self._empty = kernel
        self.backend = matmul.backend
        self.torch_version = matmul.torch_version
        self.float32_matmul_precision = matmul.float32_matmul_precision
        self.limits = {
            **matmul.limits,
            EMPTY_KERNEL: PipelineLimits(
                max_threads_per_threadgroup=kernel.max_threads_per_threadgroup,
                thread_execution_width=kernel.thread_execution_width,
                static_threadgroup_memory_bytes=kernel.static_thread_group_memory_length,
            ),
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def event_pair(self) -> EventPair:
        """Return a fresh start and end event for one side."""

        return EventPair(self._torch)

    def empty_side(self, launch: CheckedLaunch) -> Callable[[], object]:
        """Bind the empty kernel to its one-thread launch and a sink buffer."""

        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[EMPTY_KERNEL]
            or launch.geometry != empty_launch()
        ):
            raise GpuTimingError(
                "the empty kernel runs only with its one-thread launch checked "
                "against its own pipeline"
            )
        sink = self._torch.zeros(1, dtype=self._torch.int32, device="mps")
        return partial(self._empty, sink, **launch.geometry.launch_arguments())

    def workload(
        self,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launches: Mapping[str, CheckedLaunch],
    ) -> MatmulWorkload:
        """Bind the three Day 74 sides on one point's device operands."""

        return self._matmul.workload(left, right, launches)
