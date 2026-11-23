"""Run the fused softmax kernels and ``torch.softmax`` on the MPS device.

Importing this module never imports torch, so a host without torch can still
verify reports. ``MetalFusedRuntime`` first copies the two PyTorch MPS
switches from the environment, as the Day 76 runtime does:
``PYTORCH_MPS_FAST_MATH`` and ``PYTORCH_MPS_PREFER_METAL`` can each change
what PyTorch runs, and the lab cannot see PyTorch's kernel. It then runs the
Day 73 ``probe_metal``, which reads the operating system before it imports
torch, compiles the fused library through the public
``torch.mps.compile_shader`` and reads the limits of every pipeline.

A step runs only when every pipeline of the step admits it
(``admit``): each one allows the largest group of the run, reports a SIMD
width of 32, and reports the static threadgroup bytes of its text. A refused
step is data, not an error: the run records the verdict and never launches
the step. Each launch is one group of G threads per row, checked against the
pipeline of each kernel that it runs.

The runtime takes and returns NumPy arrays, so the correctness and benchmark
code never touch torch, and a stand-in can replace the runtime on a host
without MPS. Only ``MetalFusedRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import (
    CheckedLaunch,
    KernelMetalError,
    PipelineLimits,
    vector_launch,
)
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_tiled_runtime import mps_switches, pipeline_limits
from .metal_fused_plan import TORCH_SOFTMAX
from .metal_fused_reference import RowLayout, sentinel_buffer
from .metal_fused_source import (
    EXP_PROBE,
    EXP_PROBE_GROUP,
    KERNELS,
    LOWEST,
    SIMD_WIDTH,
    STEP_VARIANTS,
    VARIANTS,
    MetalFusedError,
    fused_library,
)

FEWER_THREADS = "pipeline_allows_fewer_threads_than_the_group"
OTHER_SIMD_WIDTH = "pipeline_simd_width_is_not_32"
OTHER_THREADGROUP_BYTES = "static_threadgroup_bytes_differ_from_the_text"
REFUSALS = (FEWER_THREADS, OTHER_SIMD_WIDTH, OTHER_THREADGROUP_BYTES)
TORCH_EQUIVALENT = (
    "torch.softmax(x, dim=-1, out=y) on the MPS device: on the rows x cols "
    "matrix of a dense point, and on the rows x longest matrix of a ragged "
    "point, with the most negative finite float32 in each slot past a row"
)


class MetalFusedUnavailableError(MetalFusedError):
    """Raised when this host cannot compile or run the fused softmax kernels."""


@dataclass(frozen=True)
class FusedAdmission:
    """Whether every pipeline of one step admits the largest group of the run."""

    variant: str
    required_threads: int
    limits: Mapping[str, PipelineLimits]

    def __post_init__(self) -> None:
        if self.variant not in STEP_VARIANTS:
            raise MetalFusedError(f"{self.variant!r} is not a registered step")
        if tuple(self.limits) != VARIANTS[self.variant].kernels:
            raise MetalFusedError("an admission needs the limits of each kernel")

    @property
    def reason(self) -> str | None:
        """The first refusal of the step's pipelines, or ``None``."""

        for name in VARIANTS[self.variant].kernels:
            limits = self.limits[name]
            if limits.max_threads_per_threadgroup < self.required_threads:
                return FEWER_THREADS
            if limits.thread_execution_width != SIMD_WIDTH:
                return OTHER_SIMD_WIDTH
            if (
                limits.static_threadgroup_memory_bytes
                != KERNELS[name].threadgroup_bytes
            ):
                return OTHER_THREADGROUP_BYTES
        return None

    @property
    def admitted(self) -> bool:
        """Whether the step may launch."""

        return self.reason is None

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict and the numbers behind it."""

        return {
            "variant": self.variant,
            "admitted": self.admitted,
            "reason": self.reason,
            "required_threads": self.required_threads,
            "kernels": [
                {
                    "name": name,
                    "allowed_threads": limits.max_threads_per_threadgroup,
                    "thread_execution_width": limits.thread_execution_width,
                    "declared_threadgroup_bytes": KERNELS[name].threadgroup_bytes,
                    "static_threadgroup_bytes": limits.static_threadgroup_memory_bytes,
                }
                for name, limits in self.limits.items()
            ],
        }


def admit(
    limits: Mapping[str, PipelineLimits], required_threads: int
) -> dict[str, FusedAdmission]:
    """Return the verdict of every step for a run whose largest group is given."""

    return {
        name: FusedAdmission(
            name,
            required_threads,
            {kernel: limits[kernel] for kernel in VARIANTS[name].kernels},
        )
        for name in STEP_VARIANTS
    }


def checked_launch(
    limits: Mapping[str, PipelineLimits], kernel: str, layout: RowLayout
) -> CheckedLaunch:
    """One group of G threads per row, checked against one kernel's pipeline."""

    if kernel not in KERNELS or kernel == EXP_PROBE:
        raise MetalFusedError(f"{kernel!r} is not a row kernel")
    size = layout.group_size
    try:
        return CheckedLaunch(vector_launch(layout.rows * size, size), limits[kernel])
    except MetalFusedError:
        raise
    except KernelMetalError as error:
        raise MetalFusedError(f"{kernel}: {error}") from error


@dataclass(frozen=True)
class FusedWorkload:
    """One timing point's device buffers, bound as no-argument calls.

    ``sides`` holds the admitted steps in plan order, then
    ``torch.softmax``. Each call only queues work; ``synchronize`` waits for
    it. ``outputs`` waits too and returns every side's latest output, packed
    in row order.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    outputs: Callable[[], Mapping[str, numpy.ndarray]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _compile(torch: object, source: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise MetalFusedUnavailableError(
            f"the Metal compiler rejected the fused softmax library: {error}"
        ) from error


def _buffer(values: object, layout: RowLayout) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.shape != (layout.buffer_elements,)
    ):
        raise MetalFusedError("an input must be a float32 buffer of its layout")
    return numpy.ascontiguousarray(values)


class MetalFusedRuntime:
    """The compiled fused library on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        library_source = fused_library()
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MetalFusedUnavailableError(availability.message)
        torch = import_module("torch")
        library = _compile(torch, library_source)
        self._torch = torch
        self.library_source = library_source
        self._functions = {name: getattr(library, name) for name in KERNELS}
        self.torch_version = availability.torch_version
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")

    def _call(self, variant: str, launch: CheckedLaunch, buffers: dict) -> partial:
        """Bind one launch of a step: three kernels for U0, one otherwise."""

        arguments = launch.geometry.launch_arguments()
        calls = [
            partial(
                self._functions[name],
                *(buffers[buffer] for buffer in KERNELS[name].buffers),
                **arguments,
            )
            for name in VARIANTS[variant].kernels
        ]
        if len(calls) == 1:
            return calls[0]

        def run_all() -> None:
            for call in calls:
                call()

        return partial(run_all)

    def _launch(self, variant: str, layout: RowLayout, required: int) -> CheckedLaunch:
        """Admit one step for the run and check its launch on each of its pipelines.

        Every kernel of a step runs the same geometry, so one checked launch
        binds all of them once each pipeline has accepted it.
        """

        if variant not in STEP_VARIANTS:
            raise MetalFusedError(f"{variant!r} is not a registered step")
        if layout.group_size > required:
            raise MetalFusedError("a launch needs more threads than the run admitted")
        admission = admit(self.limits, required)[variant]
        if not admission.admitted:
            raise MetalFusedError(
                f"{variant} was refused by its pipeline: {admission.reason}"
            )
        kernels = VARIANTS[variant].kernels
        for name in kernels:
            checked_launch(self.limits, name, layout)
        return checked_launch(self.limits, kernels[0], layout)

    def _shared(self, x: object, layout: RowLayout) -> dict[str, object]:
        """The buffers that every side of one layout reads or reuses."""

        torch = self._torch
        return {
            "x": x,
            "starts": self._to(layout.starts()),
            "lengths": self._to(layout.lengths_array()),
            "row_max": torch.empty(layout.rows, dtype=torch.float32, device="mps"),
            "row_sum": torch.empty(layout.rows, dtype=torch.float32, device="mps"),
        }

    def run(
        self, variant: str, values: numpy.ndarray, layout: RowLayout, required: int
    ) -> numpy.ndarray:
        """Run one step once and return its whole output buffer, gaps included.

        The buffer starts as sentinels, so a store into a gap shows there.
        ``required`` is the largest group of the run, which the pipelines of
        the step must admit.
        """

        launch = self._launch(variant, layout, required)
        buffers = self._shared(self._to(_buffer(values, layout)), layout)
        buffers["y"] = self._to(sentinel_buffer(layout))
        self._call(variant, launch, buffers)()
        self.synchronize()
        return buffers["y"].cpu().numpy()

    def exp_probe(self, arguments: numpy.ndarray) -> numpy.ndarray:
        """Run ``precise::exp`` once at each argument; return one float each."""

        if (
            not isinstance(arguments, numpy.ndarray)
            or arguments.dtype != numpy.float32
            or arguments.ndim != 1
            or arguments.size % EXP_PROBE_GROUP
        ):
            raise MetalFusedError(
                f"probe arguments must be float32, a multiple of {EXP_PROBE_GROUP}"
            )
        launch = CheckedLaunch(
            vector_launch(arguments.size, EXP_PROBE_GROUP), self.limits[EXP_PROBE]
        )
        x = self._to(arguments)
        y = self._torch.empty_like(x)
        self._functions[EXP_PROBE](y, x, **launch.geometry.launch_arguments())
        self.synchronize()
        return y.cpu().numpy()

    def workload(
        self, values: numpy.ndarray, layout: RowLayout, names: tuple[str, ...]
    ) -> FusedWorkload:
        """Place one point's rows on the device and bind every side once.

        ``names`` are the admitted steps in plan order; ``torch.softmax``
        writes into its own preallocated output after them. A ragged point
        gives torch the padded rows x longest matrix, built once on the host.
        """

        if layout.gap != 0:
            raise MetalFusedError("a timing point packs its rows with no gap")
        if not names or tuple(name for name in STEP_VARIANTS if name in names) != names:
            raise MetalFusedError("a workload binds admitted steps in plan order")
        torch = self._torch
        rows = _buffer(values, layout)
        shared = self._shared(self._to(rows), layout)
        launches = {
            name: self._launch(name, layout, layout.group_size) for name in names
        }
        outputs = {
            name: torch.empty(layout.elements, dtype=torch.float32, device="mps")
            for name in names
        }
        sides = {
            name: self._call(name, launches[name], {**shared, "y": outputs[name]})
            for name in names
        }
        mask = numpy.arange(layout.longest)[None, :] < layout.lengths_array()[:, None]
        if layout.dense:
            source = shared["x"].view(layout.rows, layout.longest)
        else:
            padded = numpy.full(mask.shape, LOWEST, dtype=numpy.float32)
            padded[mask] = rows
            source = self._to(padded)
        target = torch.empty_like(source)
        sides[TORCH_SOFTMAX] = partial(torch.softmax, source, dim=-1, out=target)

        def latest() -> dict[str, numpy.ndarray]:
            self.synchronize()
            results = {name: outputs[name].cpu().numpy() for name in names}
            results[TORCH_SOFTMAX] = target.cpu().numpy()[mask]
            return results

        return FusedWorkload(sides=sides, synchronize=self.synchronize, outputs=latest)
