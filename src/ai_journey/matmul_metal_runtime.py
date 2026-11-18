"""Run the naive matmul kernels through PyTorch's MPS backend, importing torch late.

Importing this module never imports torch, so a host without torch can still
verify reports. ``MatmulMetalRuntime`` reuses the Day 73 ``probe_metal``,
which reads the operating system before it imports torch and names the first
missing requirement. The runtime then compiles ``SHADER_SOURCE`` once, reads
each pipeline's threadgroup limits, and runs a kernel only with a launch that
was checked against that kernel's own pipeline over its own mapping's grid.
Scalars reach the kernels as 32-bit integers through ``arg_casts``, the
width of their ``constant uint&`` parameters; PyTorch would otherwise pass
64 bits.

The runtime takes and returns NumPy arrays, so the correctness and benchmark
code never touch torch, and a recorded stand-in can replace the runtime on a
host without MPS. Only ``MatmulMetalRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch, PipelineLimits
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import BACKEND, DEVICES, probe_metal
from .matmul_metal_mapping import MatmulMetalError, grid_extent, matmul_shape
from .matmul_metal_source import KERNELS, MATMUL_X_COLUMNS, MATMUL_X_ROWS, SHADER_SOURCE
from .memory_access_tiling import MatmulShape

TORCH_MATMUL = "torch_matmul"
# The three implementations a benchmark point times, in their first order.
SIDES = (MATMUL_X_COLUMNS, MATMUL_X_ROWS, TORCH_MATMUL)


class MatmulMetalUnavailableError(MatmulMetalError):
    """Raised when this host cannot compile or run the Metal matmul kernels."""


@dataclass(frozen=True)
class WorkloadCheck:
    """What the latest outputs of one benchmark point showed after the timing.

    ``naive_same_bits`` compares the two naive kernels bit for bit.
    ``torch_disagreement`` is the largest ``|torch - naive| / (|A| @ |B|)``,
    computed on the device in float32.
    """

    naive_same_bits: bool
    torch_disagreement: float


@dataclass(frozen=True)
class MatmulWorkload:
    """One benchmark point's device buffers, bound as no-argument calls.

    Each side only queues work; ``synchronize`` waits for it. ``check`` waits
    too, then compares the latest outputs of the three sides.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    check: Callable[[], WorkloadCheck]


def _operands(left: object, right: object) -> tuple[numpy.ndarray, numpy.ndarray]:
    if (
        not isinstance(left, numpy.ndarray)
        or not isinstance(right, numpy.ndarray)
        or left.dtype != numpy.float32
        or right.dtype != numpy.float32
        or left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
    ):
        raise MatmulMetalError("operands must be conformable 2D float32 arrays")
    return numpy.ascontiguousarray(left), numpy.ascontiguousarray(right)


def operand_shape(left: numpy.ndarray, right: numpy.ndarray) -> MatmulShape:
    """Return the bounded shape of ``left @ right``."""

    a, b = _operands(left, right)
    return matmul_shape(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))


class MatmulMetalRuntime:
    """The compiled matmul kernels on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MatmulMetalUnavailableError(availability.message)
        torch = import_module("torch")
        try:
            library = torch.mps.compile_shader(SHADER_SOURCE)
        except RuntimeError as error:
            raise MatmulMetalUnavailableError(
                f"the Metal compiler rejected the matmul source: {error}"
            ) from error
        self._torch = torch
        self._library = library
        self.torch_version = availability.torch_version
        self.float32_matmul_precision = str(torch.get_float32_matmul_precision())
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
            raise MatmulMetalError(f"device must be one of {', '.join(DEVICES)}")
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to(device)

    def _admit(self, name: str, launch: object, shape: MatmulShape) -> CheckedLaunch:
        """Accept only a launch checked against this kernel's pipeline and grid."""

        spec = KERNELS[name]
        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[name]
            or launch.geometry.extent != grid_extent(spec.mapping, shape)
        ):
            raise MatmulMetalError(
                f"{name} runs only with a launch checked against its own pipeline "
                f"over the {spec.mapping} grid of this shape"
            )
        return launch

    def _kernel(
        self, name: str, launch: CheckedLaunch, shape: MatmulShape, *buffers: object
    ) -> partial:
        """Bind one admitted launch with its geometry and 32-bit scalar casts."""

        return partial(
            getattr(self._library, name),
            *buffers,
            shape.m,
            shape.n,
            shape.k,
            **launch.geometry.launch_arguments(),
            arg_casts=KERNELS[name].arg_casts,
        )

    def run(
        self,
        name: str,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launch: CheckedLaunch,
    ) -> numpy.ndarray:
        """Run one kernel once and return its whole flat output buffer.

        The buffer starts as sentinels and is longer than C by the launch's
        padding threads, so a write past the guard would show in its tail.
        """

        if name not in KERNELS:
            raise MatmulMetalError(f"kernel must be one of {', '.join(KERNELS)}")
        a, b = _operands(left, right)
        shape = operand_shape(a, b)
        admitted = self._admit(name, launch, shape)
        elements = shape.m * shape.n + admitted.geometry.padding_threads
        output = self._to(float32_sentinels(elements))
        self._kernel(name, admitted, shape, output, self._to(a), self._to(b))()
        self.synchronize()
        return output.cpu().numpy()

    def torch_matmul(
        self, left: numpy.ndarray, right: numpy.ndarray, device: str
    ) -> numpy.ndarray:
        """Multiply with ``torch.matmul`` on ``device`` and return the product."""

        a, b = _operands(left, right)
        return (
            self._torch.matmul(self._to(a, device), self._to(b, device)).cpu().numpy()
        )

    def workload(
        self,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launches: Mapping[str, CheckedLaunch],
    ) -> MatmulWorkload:
        """Place one benchmark point's operands on the device and bind every side."""

        torch = self._torch
        a_host, b_host = _operands(left, right)
        shape = operand_shape(a_host, b_host)
        if set(launches) != set(KERNELS):
            raise MatmulMetalError("a workload needs one launch per naive kernel")
        admitted = {
            name: self._admit(name, launch, shape) for name, launch in launches.items()
        }
        a, b = self._to(a_host), self._to(b_host)
        outputs = {
            name: torch.empty((shape.m, shape.n), dtype=torch.float32, device="mps")
            for name in SIDES
        }
        sides = {
            name: self._kernel(name, admitted[name], shape, outputs[name], a, b)
            for name in KERNELS
        }
        sides[TORCH_MATMUL] = partial(torch.matmul, a, b, out=outputs[TORCH_MATMUL])

        def check() -> WorkloadCheck:
            self.synchronize()
            columns, rows = outputs[MATMUL_X_COLUMNS], outputs[MATMUL_X_ROWS]
            same = bool(torch.equal(columns.view(torch.int32), rows.view(torch.int32)))
            scale = torch.matmul(a.abs(), b.abs())
            difference = (outputs[TORCH_MATMUL] - columns).abs_().div_(scale)
            disagreement = float(difference.max().item())
            self.synchronize()
            return WorkloadCheck(same, disagreement)

        return MatmulWorkload(
            sides={name: sides[name] for name in SIDES},
            synchronize=self.synchronize,
            check=check,
        )
