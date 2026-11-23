"""Run the Day 80 kernels and their PyTorch sequences on the MPS device.

Importing this module never imports torch, so a host without torch can
still verify reports. ``MetalLigerRuntime`` first copies the two PyTorch MPS
switches from the environment, as the Day 76 and Day 79 runtimes do:
``PYTORCH_MPS_FAST_MATH`` and ``PYTORCH_MPS_PREFER_METAL`` can each change
what PyTorch runs. It then runs the Day 73 ``probe_metal``, which reads the
operating system before it imports torch, compiles the library through the
public ``torch.mps.compile_shader`` and reads the limits of every pipeline.

A kernel launches only when its pipeline admits it (``admit``): the
pipeline allows the largest group of the run, reports a SIMD width of 32,
and reports the static threadgroup bytes of its text. The Day 79 refusal
names apply. Each launch is one group of G threads for each row, checked
against the kernel's pipeline.

The runtime takes and returns NumPy arrays, so the correctness and
benchmark code never touch torch, and a stand-in can replace the runtime on
a host without MPS. Only ``MetalLigerRuntime`` itself makes a measured
report.
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
from .metal_fused_runtime import (
    FEWER_THREADS,
    OTHER_SIMD_WIDTH,
    OTHER_THREADGROUP_BYTES,
)
from .metal_fused_source import SIMD_WIDTH
from .metal_liger_plan import TORCH_SIDES
from .metal_liger_reference import LigerInputs, LigerLayout, sentinel_output
from .metal_liger_source import (
    EPS,
    FAMILY_KERNELS,
    KERNELS,
    PROBE_GROUP,
    PROBES,
    RMSNORM,
    SWIGLU,
    MetalLigerError,
    liger_library,
)

REFUSALS = (FEWER_THREADS, OTHER_SIMD_WIDTH, OTHER_THREADGROUP_BYTES)


class MetalLigerUnavailableError(MetalLigerError):
    """Raised when this host cannot compile or run the Day 80 kernels."""


@dataclass(frozen=True)
class LigerAdmission:
    """Whether one pipeline admits the largest group that its launches need."""

    kernel: str
    required_threads: int
    limits: PipelineLimits

    def __post_init__(self) -> None:
        if self.kernel not in KERNELS:
            raise MetalLigerError(f"{self.kernel!r} is not a registered kernel")
        if not isinstance(self.limits, PipelineLimits):
            raise TypeError("an admission needs the limits of its pipeline")

    @property
    def reason(self) -> str | None:
        """The first refusal of the pipeline, or ``None``."""

        limits = self.limits
        if limits.max_threads_per_threadgroup < self.required_threads:
            return FEWER_THREADS
        if limits.thread_execution_width != SIMD_WIDTH:
            return OTHER_SIMD_WIDTH
        if (
            limits.static_threadgroup_memory_bytes
            != KERNELS[self.kernel].threadgroup_bytes
        ):
            return OTHER_THREADGROUP_BYTES
        return None

    @property
    def admitted(self) -> bool:
        """Whether the kernel may launch."""

        return self.reason is None

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict and the numbers behind it."""

        return {
            "kernel": self.kernel,
            "admitted": self.admitted,
            "reason": self.reason,
            "required_threads": self.required_threads,
            "allowed_threads": self.limits.max_threads_per_threadgroup,
            "thread_execution_width": self.limits.thread_execution_width,
            "declared_threadgroup_bytes": KERNELS[self.kernel].threadgroup_bytes,
            "static_threadgroup_bytes": self.limits.static_threadgroup_memory_bytes,
        }


def admit(
    limits: Mapping[str, PipelineLimits], required_threads: int
) -> dict[str, LigerAdmission]:
    """Return the verdict of every kernel; a probe needs its own group of 256."""

    return {
        name: LigerAdmission(
            name,
            required_threads if kernel.family is not None else PROBE_GROUP,
            limits[name],
        )
        for name, kernel in KERNELS.items()
    }


def checked_launch(
    limits: Mapping[str, PipelineLimits], family: str, layout: LigerLayout
) -> CheckedLaunch:
    """One group of G threads per row, checked against the family's pipeline."""

    size = layout.group_size(family)
    try:
        return CheckedLaunch(
            vector_launch(layout.rows * size, size), limits[FAMILY_KERNELS[family]]
        )
    except MetalLigerError:
        raise
    except KernelMetalError as error:
        raise MetalLigerError(f"{family}: {error}") from error


@dataclass(frozen=True)
class LigerWorkload:
    """One timing point's device buffers, bound as no-argument calls.

    ``sides`` holds the fused kernel and then its PyTorch sequence. Each call
    only queues work; ``synchronize`` waits for it. ``outputs`` waits too and
    returns each side's latest output, packed in row order.
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
        raise MetalLigerUnavailableError(
            f"the Metal compiler rejected the Day 80 library: {error}"
        ) from error


def _require_inputs(inputs: object) -> LigerInputs:
    if not isinstance(inputs, LigerInputs):
        raise TypeError("inputs must be LigerInputs")
    return inputs


class MetalLigerRuntime:
    """The compiled Day 80 library on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        library_source = liger_library()
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MetalLigerUnavailableError(availability.message)
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
        array = numpy.ascontiguousarray(array)
        if array.dtype == numpy.uint32:
            # The kernels read uint; int32 carries the same bits into torch.
            array = array.view(numpy.int32)
        return self._torch.from_numpy(array).to("mps")

    def _launch(self, family: str, layout: LigerLayout, required: int) -> CheckedLaunch:
        if layout.group_size(family) > required:
            raise MetalLigerError("a launch needs more threads than the run admitted")
        kernel = FAMILY_KERNELS[family]
        admission = admit(self.limits, required)[kernel]
        if not admission.admitted:
            raise MetalLigerError(
                f"{kernel} was refused by its pipeline: {admission.reason}"
            )
        return checked_launch(self.limits, family, layout)

    def _bind(self, family: str, launch: CheckedLaunch, buffers: list) -> partial:
        return partial(
            self._functions[FAMILY_KERNELS[family]],
            *buffers,
            **launch.geometry.launch_arguments(),
        )

    def _device(self, inputs: LigerInputs) -> list[object]:
        layout = inputs.layout
        return [
            self._to(inputs.main),
            self._to(inputs.second),
            self._to(layout.starts()),
            self._to(layout.lengths_array()),
        ]

    def run(self, inputs: LigerInputs, required: int) -> numpy.ndarray:
        """Run one family's kernel once and return its whole output buffer.

        The buffer starts as sentinels, so a store into a gap shows there.
        ``required`` is the largest group of the run, which the pipeline must
        admit.
        """

        inputs = _require_inputs(inputs)
        family, layout = inputs.family, inputs.layout
        launch = self._launch(family, layout, required)
        output = self._to(sentinel_output(family, layout))
        self._bind(family, launch, [output, *self._device(inputs)])()
        self.synchronize()
        return output.cpu().numpy()

    def probe(self, function: str, arguments: numpy.ndarray) -> numpy.ndarray:
        """Run one ``precise::`` function once at each argument."""

        if function not in PROBES:
            raise MetalLigerError(f"{function!r} has no probe kernel")
        if (
            not isinstance(arguments, numpy.ndarray)
            or arguments.dtype != numpy.float32
            or arguments.ndim != 1
            or arguments.size % PROBE_GROUP
        ):
            raise MetalLigerError(
                f"probe arguments must be float32, a multiple of {PROBE_GROUP}"
            )
        name = PROBES[function]
        launch = CheckedLaunch(
            vector_launch(arguments.size, PROBE_GROUP), self.limits[name]
        )
        x = self._to(arguments)
        y = self._torch.empty_like(x)
        self._functions[name](y, x, **launch.geometry.launch_arguments())
        self.synchronize()
        return y.cpu().numpy()

    def _torch_side(self, inputs: LigerInputs, device: list[object]):
        """Bind the PyTorch sequence of one family; return it and its reader."""

        torch = self._torch
        layout = inputs.layout
        rows, cols = layout.rows, layout.longest
        main = device[0].view(rows, cols)
        if inputs.family == RMSNORM:
            weights = device[1]
            target = torch.empty_like(main)

            def rmsnorm() -> None:
                mean = main.pow(2).mean(-1, keepdim=True)
                inverse = torch.rsqrt(mean + EPS)
                torch.mul(main * inverse, weights, out=target)

            return rmsnorm, lambda: target.cpu().numpy().reshape(-1)
        if inputs.family == SWIGLU:
            gate = device[1].view(rows, cols)
            target = torch.empty_like(main)
            silu = torch.nn.functional.silu

            def swiglu() -> None:
                torch.mul(silu(main), gate, out=target)

            return swiglu, lambda: target.cpu().numpy().reshape(-1)
        targets = self._torch.from_numpy(inputs.second.astype(numpy.int64)).to("mps")
        latest: dict[str, object] = {}
        loss = torch.nn.functional.cross_entropy

        def cross_entropy() -> None:
            latest["loss"] = loss(main, targets, reduction="none")

        return cross_entropy, lambda: latest["loss"].cpu().numpy()

    def workload(self, inputs: LigerInputs) -> LigerWorkload:
        """Place one point's rows on the device and bind both sides once.

        The fused kernel writes into its own output; the PyTorch sequence
        reads the same device rows as a rows x cols matrix.
        """

        inputs = _require_inputs(inputs)
        family, layout = inputs.family, inputs.layout
        if layout.gap != 0 or not layout.dense:
            raise MetalLigerError("a timing point packs dense rows with no gap")
        launch = self._launch(family, layout, layout.group_size(family))
        device = self._device(inputs)
        output = self._torch.empty(
            layout.output_elements(family), dtype=self._torch.float32, device="mps"
        )
        fused = self._bind(family, launch, [output, *device])
        vendor, read_vendor = self._torch_side(inputs, device)
        kernel, side = FAMILY_KERNELS[family], TORCH_SIDES[family]

        def latest() -> dict[str, numpy.ndarray]:
            self.synchronize()
            return {kernel: output.cpu().numpy(), side: read_vendor()}

        return LigerWorkload(
            sides={kernel: fused, side: vendor},
            synchronize=self.synchronize,
            outputs=latest,
        )
