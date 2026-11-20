"""Run the reduction kernels and ``torch.sum`` on the MPS device.

Importing this module never imports torch, so a host without torch can still
verify reports. ``GpuReduceRuntime`` first copies the two PyTorch MPS
switches from the environment, as the Day 76 runtime does:
``PYTORCH_MPS_FAST_MATH`` and ``PYTORCH_MPS_PREFER_METAL`` can change what
PyTorch's own kernels run, and the lab cannot see the kernel behind
``torch.sum``. It then runs the Day 73 ``probe_metal``, which reads the
operating system before it imports torch, and compiles the library through
the public ``torch.mps.compile_shader``.

Each pipeline reports its limits, and the runtime admits a kernel only when
its pipeline allows the group size, reports 32-lane SIMD groups where the
kernel needs them, and holds exactly the threadgroup memory that the kernel
declares. Every step needs the combine kernel and the lab compares all six
steps, so one refusal stops the runtime before any case runs.

A correctness run puts the input on the device with a tail of quiet-NaN
sentinels, so a read past n would turn the sum into NaN. The partials and
the output start as sentinels too and carry tails that no correct thread
writes. A timing workload binds every side once to one device buffer: each
kernel reads its first n values, and ``torch.sum`` reads the view of the
same values. Scalars reach every kernel as 32-bit integers through
``arg_casts``. The runtime takes and returns NumPy arrays, so a stand-in can
replace it on a host without MPS. Only ``GpuReduceRuntime`` makes a
measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .gpu_reduce_bound import combine_launch, pass_one_launch
from .gpu_reduce_plan import MAX_ELEMENTS, TORCH_SUM
from .gpu_reduce_source import (
    COMBINE,
    KERNELS,
    STEP_KERNELS,
    GpuReduceError,
    ReduceKernel,
    library_source,
)
from .kernel_metal_geometry import CheckedLaunch, PipelineLimits
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_tiled_runtime import mps_switches, pipeline_limits

INPUT_GUARD = 1024
PARTIAL_GUARD = 64
OUTPUT_SLOTS = 32
SIDES = (*STEP_KERNELS, TORCH_SUM)
FEWER_THREADS = "pipeline_allows_fewer_threads_than_the_group"
OTHER_SIMD_WIDTH = "pipeline_simd_width_is_not_32"
OTHER_THREADGROUP_BYTES = "static_threadgroup_bytes_differ_from_the_kernel"
TORCH_EQUIVALENT = "torch.sum(x[:n], 0, out=out) on the MPS device, float32"


class GpuReduceUnavailableError(GpuReduceError):
    """Raised when this host cannot compile or run the reduction kernels."""


@dataclass(frozen=True)
class KernelAdmission:
    """One pipeline's verdict on one kernel."""

    kernel: ReduceKernel
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, ReduceKernel):
            raise TypeError("kernel must be a ReduceKernel")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("limits must be PipelineLimits")

    @property
    def reason(self) -> str | None:
        """The first rule the pipeline breaks, or ``None`` when it admits."""

        limits, kernel = self.limits, self.kernel
        if limits.max_threads_per_threadgroup < kernel.group_size:
            return FEWER_THREADS
        if (
            kernel.simd_lanes is not None
            and limits.thread_execution_width != kernel.simd_lanes
        ):
            return OTHER_SIMD_WIDTH
        if limits.static_threadgroup_memory_bytes != kernel.threadgroup_bytes:
            return OTHER_THREADGROUP_BYTES
        return None

    @property
    def admitted(self) -> bool:
        """Whether the kernel may launch on this pipeline."""

        return self.reason is None

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict as a report records it."""

        return {
            "name": self.kernel.name,
            "admitted": self.admitted,
            "reason": self.reason,
            "required_threads": self.kernel.group_size,
            "allowed_threads": self.limits.max_threads_per_threadgroup,
            "required_simd_width": self.kernel.simd_lanes,
            "simd_width": self.limits.thread_execution_width,
            "declared_threadgroup_bytes": self.kernel.threadgroup_bytes,
            "static_threadgroup_bytes": self.limits.static_threadgroup_memory_bytes,
        }


def require_admitted(admissions: Mapping[str, KernelAdmission]) -> None:
    """Stop before any case when a pipeline refuses one kernel."""

    for name, admission in admissions.items():
        if not admission.admitted:
            raise GpuReduceError(
                f"{name} was refused by its pipeline: {admission.reason}; every "
                "step needs every kernel, so the run stops"
            )


def checked_launches(
    limits: Mapping[str, PipelineLimits], name: str, n: int
) -> tuple[CheckedLaunch, CheckedLaunch]:
    """Size both passes of one step kernel for n and check them on their pipelines."""

    return (
        CheckedLaunch(pass_one_launch(name, n), limits[name]),
        CheckedLaunch(combine_launch(), limits[COMBINE]),
    )


@dataclass(frozen=True)
class ReductionBuffers:
    """The whole partials and output buffers of one guarded run."""

    partials: numpy.ndarray
    output: numpy.ndarray


@dataclass(frozen=True)
class SideResult:
    """The latest timed result of one side and whether a repeat kept its bits."""

    output: float
    same_bits_on_repeat: bool


@dataclass(frozen=True)
class ReduceWorkload:
    """One timing size's bound calls on the shared device buffer.

    ``sides`` holds R0 to R5 and then ``torch.sum``. Each call only queues
    work; ``synchronize`` waits for it. ``check`` waits, reads each side's
    latest result, runs each side once more into a second output and
    compares the bits.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    check: Callable[[], Mapping[str, SideResult]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _require_values(values: object) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
        or not 1 <= values.size <= MAX_ELEMENTS
    ):
        raise GpuReduceError("values must be a 1D float32 array of 1 to 2^26 values")
    return numpy.ascontiguousarray(values)


def _compile(torch: object, source: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise GpuReduceUnavailableError(
            f"the Metal compiler rejected the reduction library: {error}"
        ) from error


def _first_float(tensor: object) -> tuple[float, int]:
    value = tensor.reshape(-1)[:1].cpu().numpy()
    return float(value[0]), int(value.view(numpy.uint32)[0])


class GpuReduceRuntime:
    """The compiled reduction library on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        source = library_source()
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise GpuReduceUnavailableError(availability.message)
        torch = import_module("torch")
        library = _compile(torch, source)
        self._torch = torch
        self.library_source = source
        self._functions = {name: getattr(library, name) for name in KERNELS}
        self.torch_version = availability.torch_version
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }
        self.admissions = {
            name: KernelAdmission(KERNELS[name], self.limits[name]) for name in KERNELS
        }
        require_admitted(self.admissions)

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")

    def _kernel(self, name: str, launch: CheckedLaunch, *buffers: object) -> partial:
        """Bind one checked launch with its geometry and 32-bit scalar cast."""

        if launch.limits != self.limits[name]:
            raise GpuReduceError(f"{name} runs only on its own pipeline's launch")
        return partial(
            self._functions[name],
            *buffers,
            **launch.geometry.launch_arguments(),
            arg_casts=KERNELS[name].arg_casts,
        )

    def _passes(self, name: str, source: object, n: int, partials: object, out):
        first, second = checked_launches(self.limits, name, n)
        groups = first.geometry.groups[0]
        return (
            self._kernel(name, first, partials, source, n),
            self._kernel(COMBINE, second, out, partials, groups),
        )

    def run(self, name: str, values: numpy.ndarray) -> ReductionBuffers:
        """Run both passes of one step once on guarded buffers; return them whole."""

        if name not in STEP_KERNELS:
            raise GpuReduceError(f"{name!r} is not a step kernel")
        array = _require_values(values)
        groups = pass_one_launch(name, array.size).groups[0]
        source = self._to(numpy.concatenate([array, float32_sentinels(INPUT_GUARD)]))
        partials = self._to(float32_sentinels(groups + PARTIAL_GUARD))
        out = self._to(float32_sentinels(OUTPUT_SLOTS))
        for call in self._passes(name, source, array.size, partials, out):
            call()
        self.synchronize()
        return ReductionBuffers(partials.cpu().numpy(), out.cpu().numpy())

    def upload(self, values: numpy.ndarray) -> object:
        """Copy the timing input to the device once, for every size and side."""

        tensor = self._to(_require_values(values))
        if tensor.storage_offset() != 0:
            raise GpuReduceError("the timing input must start its device buffer")
        return tensor

    def _device_source(self, source: object, n: int) -> object:
        """Refuse a device buffer that a kernel would read past or misread."""

        torch = self._torch
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type != "mps"
            or source.dtype != torch.float32
            or source.dim() != 1
            or not source.is_contiguous()
            or source.storage_offset() != 0
            or source.numel() < n
        ):
            raise GpuReduceError(
                "a workload needs a contiguous 1D float32 MPS tensor at offset 0 "
                f"with at least {n} values"
            )
        return source

    def workload(self, source: object, n: int) -> ReduceWorkload:
        """Bind every side of one timing size to the shared device buffer."""

        torch = self._torch
        values = self._device_source(source, n)
        view = values[:n]
        sides, again, outputs = {}, {}, {}
        for name in STEP_KERNELS:
            groups = pass_one_launch(name, n).groups[0]
            partials = torch.empty(groups, dtype=torch.float32, device="mps")
            out = torch.empty(1, dtype=torch.float32, device="mps")
            repeat = torch.empty(1, dtype=torch.float32, device="mps")
            sides[name] = _Sequence(self._passes(name, values, n, partials, out))
            again[name] = _Sequence(self._passes(name, values, n, partials, repeat))
            outputs[name] = (out, repeat)
        out = torch.empty((), dtype=torch.float32, device="mps")
        repeat = torch.empty((), dtype=torch.float32, device="mps")
        sides[TORCH_SUM] = partial(torch.sum, view, 0, out=out)
        again[TORCH_SUM] = partial(torch.sum, view, 0, out=repeat)
        outputs[TORCH_SUM] = (out, repeat)

        def check() -> dict[str, SideResult]:
            self.synchronize()
            latest = {name: _first_float(pair[0]) for name, pair in outputs.items()}
            for call in again.values():
                call()
            self.synchronize()
            results = {}
            for name, (value, bits) in latest.items():
                _, repeated = _first_float(outputs[name][1])
                results[name] = SideResult(value, repeated == bits)
            return results

        return ReduceWorkload(sides=sides, synchronize=self.synchronize, check=check)


class _Sequence:
    """Queue pass 1 and then the combine as one side call."""

    def __init__(self, calls: tuple[partial, partial]) -> None:
        self._first, self._second = calls

    def __call__(self) -> None:
        self._first()
        self._second()
