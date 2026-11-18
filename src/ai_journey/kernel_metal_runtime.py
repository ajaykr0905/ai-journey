"""Run the Metal kernels through PyTorch's MPS backend, importing torch on demand.

Importing this module never imports torch, so a host without torch can still
verify reports. ``probe_metal`` checks the operating system first and only
then imports torch, and it names the first missing requirement: macOS,
torch itself, a PyTorch build with MPS, an available MPS device, or
``torch.mps.compile_shader``. ``MetalRuntime`` compiles ``SHADER_SOURCE``
once, reads each pipeline's threadgroup limits, and runs a kernel only with
a launch checked against that kernel's own pipeline. Scalars reach the
kernels as 32-bit integers through ``arg_casts``, the width of their
``constant uint&`` parameters; PyTorch would otherwise pass 64 bits.

The runtime takes and returns NumPy arrays, so correctness and benchmark
code never touch torch, and a recorded stand-in can replace the runtime on
a host without MPS. Only ``MetalRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch, KernelMetalError, PipelineLimits
from .kernel_metal_reference import (
    GRAY_SCALE,
    GRAY_WEIGHTS_PER_10000,
    MAX_LEVEL,
    MAX_LEVEL_DIFFERENCE,
    float32_sentinels,
    uint8_sentinels,
)
from .kernel_metal_source import (
    CONTRACTION_PROBE,
    KERNELS,
    PROBE_INPUTS,
    RGB_TO_GRAY,
    SHADER_SOURCE,
    VECTOR_ADD,
)

BACKEND = "torch.mps.compile_shader"
DEVICES = ("mps", "cpu")
UNAVAILABLE_REASONS = {
    "unsupported_operating_system": "Metal kernels run only on macOS",
    "torch_not_importable": "PyTorch cannot be imported",
    "pytorch_without_mps": "this PyTorch build has no MPS backend",
    "mps_unavailable": "this PyTorch has an MPS backend, but no MPS device is ready",
    "compile_shader_unavailable": "this PyTorch has no torch.mps.compile_shader",
}
REQUIREMENT = (
    "the run needs an Apple GPU on macOS and a PyTorch build with MPS and "
    "torch.mps.compile_shader"
)


class KernelMetalUnavailableError(KernelMetalError):
    """Raised when this host cannot compile or run the Metal kernels."""


@dataclass(frozen=True)
class MetalAvailability:
    """Whether this host can run the kernels, or the first missing requirement."""

    available: bool
    reason: str | None
    torch_version: str | None

    def __post_init__(self) -> None:
        if self.available != (self.reason is None) or (
            self.reason is not None and self.reason not in UNAVAILABLE_REASONS
        ):
            raise KernelMetalError("an unavailable host needs one known reason")

    @property
    def message(self) -> str:
        """Explain the result in one line that a CLI can print."""

        if self.reason is None:
            return f"Metal kernels can run through {BACKEND}"
        return f"{UNAVAILABLE_REASONS[self.reason]} ({self.reason}); {REQUIREMENT}"


def probe_metal(
    *,
    system: Callable[[], str] = platform.system,
    import_module: Callable[[str], object] = importlib.import_module,
) -> MetalAvailability:
    """Name the first missing requirement for the Metal kernels on this host.

    The operating system is read before torch is imported, so a Linux host
    gets its answer without loading torch.
    """

    if system() != "Darwin":
        return MetalAvailability(False, "unsupported_operating_system", None)
    try:
        torch = import_module("torch")
    except (ImportError, OSError):
        return MetalAvailability(False, "torch_not_importable", None)
    version = str(getattr(torch, "__version__", "unknown"))
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None or not backend.is_built():
        return MetalAvailability(False, "pytorch_without_mps", version)
    if not backend.is_available():
        return MetalAvailability(False, "mps_unavailable", version)
    compile_shader = getattr(getattr(torch, "mps", None), "compile_shader", None)
    native = getattr(getattr(torch, "_C", None), "_mps_compileShader", None)
    if not callable(compile_shader) or native is None:
        return MetalAvailability(False, "compile_shader_unavailable", version)
    return MetalAvailability(True, None, version)


@dataclass(frozen=True)
class DeviceWorkload:
    """One benchmark point's device buffers, bound as no-argument calls.

    ``metal`` and ``torch`` only queue work; ``synchronize`` waits for it.
    ``check`` waits too, then reports whether the latest outputs of the two
    implementations are equal.
    """

    metal: Callable[[], object]
    torch: Callable[[], object]
    synchronize: Callable[[], None]
    check: Callable[[], bool]


def _require_vector(values: object, length: int, label: str) -> numpy.ndarray:
    if (
        not isinstance(values, numpy.ndarray)
        or values.dtype != numpy.float32
        or values.shape != (length,)
    ):
        raise KernelMetalError(f"{label} must be a float32 vector of the launch length")
    return numpy.ascontiguousarray(values)


def _require_image(rgb: object, launch: CheckedLaunch) -> numpy.ndarray:
    width, height = launch.geometry.extent
    if (
        not isinstance(rgb, numpy.ndarray)
        or rgb.dtype != numpy.uint8
        or rgb.shape != (height, width, 3)
    ):
        raise KernelMetalError("rgb must be a uint8 image of the launch extent")
    return numpy.ascontiguousarray(rgb)


class MetalRuntime:
    """The compiled kernels on the MPS device, with NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise KernelMetalUnavailableError(availability.message)
        torch = import_module("torch")
        try:
            library = torch.mps.compile_shader(SHADER_SOURCE)
        except RuntimeError as error:
            raise KernelMetalUnavailableError(
                f"the Metal compiler rejected the kernel source: {error}"
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

    def _to(self, array: numpy.ndarray, device: str = "mps") -> object:
        if device not in DEVICES:
            raise KernelMetalError(f"device must be one of {', '.join(DEVICES)}")
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to(device)

    def _host(self, tensor: object) -> numpy.ndarray:
        return tensor.cpu().numpy()

    def _admit(self, name: str, launch: object) -> CheckedLaunch:
        """Accept only a launch checked against this kernel's own pipeline."""

        axes = KERNELS[name].axes
        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[name]
            or len(launch.geometry.extent) != axes
        ):
            raise KernelMetalError(
                f"{name} runs only with a {axes}D launch checked against its "
                "own pipeline"
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

    def contraction_probe(self) -> float:
        """Return the probe's float32 result: 0 unfused, 2**-24 fused."""

        inputs = self._to(numpy.array(PROBE_INPUTS, dtype=numpy.float32))
        output = self._to(float32_sentinels(1))
        getattr(self._library, CONTRACTION_PROBE)(
            output, inputs, threads=1, group_size=1
        )
        self.synchronize()
        return float(self._host(output)[0])

    def vector_add(
        self, first: numpy.ndarray, second: numpy.ndarray, launch: CheckedLaunch
    ) -> numpy.ndarray:
        """Run the Metal kernel once and return its whole output buffer.

        The buffer starts as sentinels and is longer than the data by the
        launch's padding threads, so a guard failure would show in its tail.
        """

        (length,) = self._admit(VECTOR_ADD, launch).geometry.extent
        left = _require_vector(first, length, "first")
        right = _require_vector(second, length, "second")
        output = self._to(float32_sentinels(length + launch.geometry.padding_threads))
        self._kernel(
            VECTOR_ADD, launch, output, self._to(left), self._to(right), length
        )()
        self.synchronize()
        return self._host(output)

    def rgb_to_gray(self, rgb: numpy.ndarray, launch: CheckedLaunch) -> numpy.ndarray:
        """Run the Metal kernel once and return its whole flat output buffer."""

        image = _require_image(rgb, self._admit(RGB_TO_GRAY, launch))
        width, height = launch.geometry.extent
        output = self._to(
            uint8_sentinels(width * height + launch.geometry.padding_threads)
        )
        self._kernel(RGB_TO_GRAY, launch, output, self._to(image), width, height)()
        self.synchronize()
        return self._host(output)

    def _gray(self, image: object) -> object:
        """PyTorch's eager equivalent: the reference's integer steps in order.

        Every value stays below 2**22 in int32, so each step is exact on the
        MPS device and on the CPU.
        """

        torch = self._torch
        pixels = image.to(torch.int32)
        red, green, blue = GRAY_WEIGHTS_PER_10000
        level = pixels[..., 0] * red + pixels[..., 1] * green
        level = level + pixels[..., 2] * blue
        level = torch.div(level + GRAY_SCALE // 2, GRAY_SCALE, rounding_mode="floor")
        return level.clamp_(0, MAX_LEVEL).to(torch.uint8)

    def torch_vector_add(
        self, first: numpy.ndarray, second: numpy.ndarray, device: str
    ) -> numpy.ndarray:
        """Add with ``torch.add`` on ``device`` and return the result."""

        length = first.size if isinstance(first, numpy.ndarray) else 0
        left = _require_vector(first, length, "first")
        right = _require_vector(second, length, "second")
        return self._host(
            self._torch.add(self._to(left, device), self._to(right, device))
        )

    def torch_rgb_to_gray(self, rgb: numpy.ndarray, device: str) -> numpy.ndarray:
        """Convert with PyTorch's eager equivalent on ``device``."""

        if not isinstance(rgb, numpy.ndarray) or rgb.dtype != numpy.uint8:
            raise KernelMetalError("rgb must be a uint8 image")
        return self._host(self._gray(self._to(rgb, device)))

    def vector_add_workload(
        self, first: numpy.ndarray, second: numpy.ndarray, launch: CheckedLaunch
    ) -> DeviceWorkload:
        """Place one benchmark point's vectors on the device and bind both sides."""

        torch = self._torch
        (length,) = self._admit(VECTOR_ADD, launch).geometry.extent
        left = self._to(_require_vector(first, length, "first"))
        right = self._to(_require_vector(second, length, "second"))
        metal_output = torch.empty_like(left)
        torch_output = torch.empty_like(left)

        def check() -> bool:
            self.synchronize()
            return bool(
                torch.equal(
                    metal_output.view(torch.int32), torch_output.view(torch.int32)
                )
            )

        return DeviceWorkload(
            metal=self._kernel(VECTOR_ADD, launch, metal_output, left, right, length),
            torch=partial(torch.add, left, right, out=torch_output),
            synchronize=self.synchronize,
            check=check,
        )

    def rgb_to_gray_workload(
        self, rgb: numpy.ndarray, launch: CheckedLaunch
    ) -> DeviceWorkload:
        """Place one benchmark point's image on the device and bind both sides."""

        torch = self._torch
        image = self._to(_require_image(rgb, self._admit(RGB_TO_GRAY, launch)))
        width, height = launch.geometry.extent
        metal_output = torch.empty((height, width), dtype=torch.uint8, device="mps")
        latest: list[object] = []

        def equivalent() -> None:
            latest[:] = [self._gray(image)]

        def check() -> bool:
            self.synchronize()
            if not latest:
                return False
            difference = (
                metal_output.to(torch.int16) - latest[0].to(torch.int16)
            ).abs()
            return int(difference.max().item()) <= MAX_LEVEL_DIFFERENCE

        return DeviceWorkload(
            metal=self._kernel(RGB_TO_GRAY, launch, metal_output, image, width, height),
            torch=equivalent,
            synchronize=self.synchronize,
            check=check,
        )
