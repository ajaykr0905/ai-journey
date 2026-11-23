"""Run the scans, ``torch.cumsum`` and the Day 74 copy on the MPS device.

Importing this module never imports torch, so a host without torch can still
verify reports. ``ScanRuntime`` first copies the two PyTorch MPS switches
from the environment, as the Day 76 runtime does: either one can change what
a PyTorch call runs, and the lab cannot see that kernel. It then runs the
Day 73 ``probe_metal``, which reads the operating system before it imports
torch, and compiles two libraries through the public
``torch.mps.compile_shader``: the scan library, and the Day 74 ceiling source
unchanged for its ``stream_copy`` kernel, the same-run copy reference.

A pipeline refuses a scan kernel when it allows fewer threads than the
group, when its static threadgroup memory differs from the declared bytes,
or, for the SIMD-group kernel, when its SIMD width is not 32. A refusal is
data, and a refused kernel never runs. The add and copy kernels must take
their groups of 256, or no multi-block scan or copy can run on this host.

A scan of ``N`` elements runs as levels: level 1 scans the input into the
output in blocks of 1024, each later level scans the block sums of the level
before it, and the add kernel then adds the scanned sums back from the top
level down. Every launch is checked against the pipeline that runs it. The
copy walks ``ceil(N / 4)`` float4 values with 4 for each thread, as the
pinned Day 74 config does, from its own zero-padded copy of the input.

The runtime takes and returns NumPy arrays, so the correctness and benchmark
code never touch torch, and a stand-in can replace the runtime on a host
without MPS. Only ``ScanRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .gpu_ceiling_reference import stream_threads
from .gpu_ceiling_source import KERNELS as CEILING_KERNELS
from .gpu_ceiling_source import SHADER_SOURCE as CEILING_SOURCE
from .gpu_ceiling_source import STREAM_COPY
from .gpu_scan_plan import COPY, VENDOR
from .gpu_scan_source import (
    ADD_ARG_CASTS,
    ADD_GROUP,
    ADD_OFFSETS,
    KERNELS,
    GpuScanError,
    ScanKernel,
    block_count,
    guard_elements,
    level_lengths,
    library_source,
    require_length,
)
from .kernel_metal_geometry import CheckedLaunch, LaunchGeometry, PipelineLimits
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_tiled_runtime import mps_switches, pipeline_limits

FEWER_THREADS = "pipeline_allows_fewer_threads_than_the_group"
OTHER_THREADGROUP_BYTES = "static_threadgroup_bytes_differ_from_the_kernel"
OTHER_SIMD_WIDTH = "simd_width_is_not_32"
REFUSAL_REASONS = (FEWER_THREADS, OTHER_THREADGROUP_BYTES, OTHER_SIMD_WIDTH)
COPY_GROUP = 256
COPY_FLOAT4_PER_THREAD = 4
# The group that each helper kernel needs from its pipeline.
HELPERS = {ADD_OFFSETS: ADD_GROUP, COPY: COPY_GROUP}


class GpuScanUnavailableError(GpuScanError):
    """Raised when this host cannot compile or run the scan lab."""


@dataclass(frozen=True)
class ScanAdmission:
    """One pipeline's verdict on one scan kernel."""

    kernel: ScanKernel
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, ScanKernel):
            raise TypeError("kernel must be a ScanKernel")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("limits must be PipelineLimits")

    @property
    def reason(self) -> str | None:
        """The first rule the pipeline breaks, or ``None`` when it admits the kernel."""

        if self.limits.max_threads_per_threadgroup < self.kernel.threads_per_group:
            return FEWER_THREADS
        if self.limits.static_threadgroup_memory_bytes != self.kernel.threadgroup_bytes:
            return OTHER_THREADGROUP_BYTES
        if (
            self.kernel.simd_width is not None
            and self.limits.thread_execution_width != self.kernel.simd_width
        ):
            return OTHER_SIMD_WIDTH
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
            "required_threads": self.kernel.threads_per_group,
            "allowed_threads": self.limits.max_threads_per_threadgroup,
            "declared_threadgroup_bytes": self.kernel.threadgroup_bytes,
            "static_threadgroup_bytes": self.limits.static_threadgroup_memory_bytes,
            "required_simd_width": self.kernel.simd_width,
            "simd_width": self.limits.thread_execution_width,
        }


def admitted_names(admissions: Mapping[str, ScanAdmission]) -> tuple[str, ...]:
    """Every scan kernel that may launch, in registry order."""

    return tuple(name for name in KERNELS if admissions[name].admitted)


@dataclass(frozen=True)
class LevelLaunch:
    """One level of a scan: its block scan and, below the top, its add."""

    length: int
    scan: CheckedLaunch
    add: CheckedLaunch | None

    def to_mapping(self) -> dict[str, object]:
        """Return the level as a report records it, so a verifier can rederive it."""

        return {
            "length": self.length,
            "blocks": block_count(self.length),
            "scan": self.scan.to_mapping(),
            "add": None if self.add is None else self.add.to_mapping(),
        }


def scan_plan(
    name: str, limits: Mapping[str, PipelineLimits], length: int
) -> tuple[LevelLaunch, ...]:
    """Size and check every launch of one scan against its own pipeline.

    A block scan launches whole groups, one group for each block; an add
    launches one thread for each element of its level, in groups of 256.
    """

    if name not in KERNELS:
        raise GpuScanError(f"{name!r} is not a registered scan kernel")
    threads = KERNELS[name].threads_per_group
    lengths = level_lengths(length)
    levels = []
    for level, count in enumerate(lengths):
        scan = CheckedLaunch(
            LaunchGeometry((block_count(count) * threads,), (threads,)), limits[name]
        )
        add = None
        if level < len(lengths) - 1:
            add = CheckedLaunch(
                LaunchGeometry((count,), (ADD_GROUP,)), limits[ADD_OFFSETS]
            )
        levels.append(LevelLaunch(count, scan, add))
    return tuple(levels)


def copy_float4(length: int) -> int:
    """Float4 values that the copy reference moves: ``ceil(N / 4)``."""

    return -(-require_length(length) // 4)


def copy_launch(limits: Mapping[str, PipelineLimits], length: int) -> CheckedLaunch:
    """The Day 74 grid-stride copy of ``ceil(N / 4)`` float4 values, 4 per thread."""

    walkers = stream_threads(copy_float4(length), COPY_FLOAT4_PER_THREAD)
    return CheckedLaunch(LaunchGeometry((walkers,), (COPY_GROUP,)), limits[COPY])


def padded_copy_input(values: numpy.ndarray) -> numpy.ndarray:
    """The input with zeros up to whole float4 values, so the copy reads no further."""

    padded = numpy.zeros(4 * copy_float4(values.size), dtype=numpy.float32)
    padded[: values.size] = values
    return padded


@dataclass(frozen=True)
class ScanWorkload:
    """One sweep point's device buffers, bound as no-argument calls.

    ``sides`` holds the admitted kernels in registry order, then
    ``torch.cumsum`` and the copy. Each call only queues work; ``synchronize``
    waits for it. ``check`` waits too, then copies every side's latest output
    to the host.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    check: Callable[[], Mapping[str, numpy.ndarray]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _values(values: object) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.ndim != 1
    ):
        raise GpuScanError("a scan input must be a float32 vector")
    level_lengths(values.size)
    return numpy.ascontiguousarray(values)


def _compile(torch: object, source: str, label: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise GpuScanUnavailableError(
            f"the Metal compiler rejected {label}: {error}"
        ) from error


def _sequence(calls: tuple[partial, ...]) -> Callable[[], None]:
    """Queue every dispatch of one scan in order; the GPU runs them in order."""

    def run() -> None:
        for call in calls:
            call()

    return run


class ScanRuntime:
    """Both compiled libraries on the MPS device, NumPy arrays in and out."""

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
            raise GpuScanUnavailableError(availability.message)
        torch = import_module("torch")
        library = _compile(torch, source, "the scan library")
        ceiling = _compile(torch, CEILING_SOURCE, "the Day 74 ceiling source")
        self._torch = torch
        self.library_source = source
        self._functions = {
            name: getattr(library, name) for name in (*KERNELS, ADD_OFFSETS)
        }
        self._functions[COPY] = getattr(ceiling, STREAM_COPY)
        self.torch_version = availability.torch_version
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }
        for helper, threads in HELPERS.items():
            if self.limits[helper].max_threads_per_threadgroup < threads:
                raise GpuScanUnavailableError(
                    f"the {helper} pipeline allows fewer than {threads} threads"
                )
        self.admissions = {
            name: ScanAdmission(kernel, self.limits[name])
            for name, kernel in KERNELS.items()
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")

    def _empty(self, count: int) -> object:
        torch = self._torch
        return torch.empty(count, dtype=torch.float32, device="mps")

    def _admit(self, name: str) -> None:
        if name not in KERNELS:
            raise GpuScanError(f"{name!r} is not a registered scan kernel")
        if not self.admissions[name].admitted:
            raise GpuScanError(
                f"{name} was refused by its pipeline: {self.admissions[name].reason}"
            )

    def _bind(self, name: str, values: object, output: object, length: int) -> tuple:
        """Bind every dispatch of one scan: the block scans up, then the adds down."""

        plan = scan_plan(name, self.limits, length)
        kernel, add = self._functions[name], self._functions[ADD_OFFSETS]
        casts = KERNELS[name].arg_casts
        prefixes, calls, source = [output], [], values
        for level, entry in enumerate(plan):
            sums = self._empty(block_count(entry.length))
            target = output if level == 0 else self._empty(entry.length)
            if level:
                prefixes.append(target)
            calls.append(
                partial(
                    kernel,
                    target,
                    source,
                    sums,
                    entry.length,
                    **entry.scan.geometry.launch_arguments(),
                    arg_casts=casts,
                )
            )
            source = sums
        for level in reversed(range(len(plan) - 1)):
            entry = plan[level]
            calls.append(
                partial(
                    add,
                    prefixes[level],
                    prefixes[level + 1],
                    entry.length,
                    **entry.add.geometry.launch_arguments(),
                    arg_casts=ADD_ARG_CASTS,
                )
            )
        return tuple(calls)

    def scan(self, name: str, values: numpy.ndarray) -> numpy.ndarray:
        """Run one scan once and return its whole output buffer.

        The buffer starts as sentinels and ends with ``guard_elements`` slots
        past the data, so a store past its guard shows in the tail.
        """

        self._admit(name)
        data = _values(values)
        output = self._to(float32_sentinels(data.size + guard_elements(data.size)))
        _sequence(self._bind(name, self._to(data), output, data.size))()
        self.synchronize()
        return output.cpu().numpy()

    def cumsum(self, values: numpy.ndarray) -> numpy.ndarray:
        """Run ``torch.cumsum`` once on the MPS device and return its output."""

        torch = self._torch
        result = torch.cumsum(self._to(_values(values)), dim=0)
        self.synchronize()
        return result.cpu().numpy()

    def workload(self, values: numpy.ndarray, names: tuple[str, ...]) -> ScanWorkload:
        """Place one point's input on the device and bind every side once."""

        torch = self._torch
        data = _values(values)
        length = data.size
        for name in names:
            self._admit(name)
        if list(names) != [name for name in KERNELS if name in names]:
            raise GpuScanError("a workload binds the kernels in registry order")
        device = self._to(data)
        outputs = {name: self._empty(length) for name in (*names, VENDOR)}
        sides = {
            name: _sequence(self._bind(name, device, outputs[name], length))
            for name in names
        }
        sides[VENDOR] = partial(torch.cumsum, device, 0, out=outputs[VENDOR])
        padded = self._to(padded_copy_input(data))
        outputs[COPY] = self._empty(padded.numel())
        launch = copy_launch(self.limits, length)
        sides[COPY] = partial(
            self._functions[COPY],
            outputs[COPY],
            padded,
            copy_float4(length),
            launch.geometry.extent[0],
            **launch.geometry.launch_arguments(),
            arg_casts=CEILING_KERNELS[STREAM_COPY].arg_casts,
        )

        def check() -> dict[str, numpy.ndarray]:
            self.synchronize()
            return {name: tensor.cpu().numpy() for name, tensor in outputs.items()}

        return ScanWorkload(sides=sides, synchronize=self.synchronize, check=check)
