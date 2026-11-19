"""Run the naive Day 74 kernel, the tiled variants and ``torch.matmul`` on MPS.

Importing this module never imports torch, so a host without torch can still
verify reports. ``MatmulTiledRuntime`` first copies two PyTorch MPS switches
from the environment as tokens: ``PYTORCH_MPS_FAST_MATH``, which turns on
fast math in PyTorch's own Metal kernels, and ``PYTORCH_MPS_PREFER_METAL``,
which makes matmul prefer Metal kernels over MPS Graph. Either one can change
what ``torch.matmul`` runs, and the lab cannot see that kernel. It then
runs the Day 73 ``probe_metal``, which reads the operating system before it
imports torch and names the first missing requirement. Through the public
``torch.mps.compile_shader`` it compiles two libraries: the Day 74
``SHADER_SOURCE`` unchanged, and the tiled library of the run's variants. It
reads the limits of every pipeline in both libraries and each tiled
pipeline's verdict on its tile.

A refused variant is data, not an error: the runtime keeps its verdict and
never launches it. Every launch must be checked against its kernel's own
pipeline over the x-over-columns grid of its shape, and a tiled launch must
use the variant's T x T group. Scalars reach every kernel as 32-bit
integers through ``arg_casts``.

The runtime takes and returns NumPy arrays, so the correctness and benchmark
code never touch torch, and a stand-in can replace the runtime on a host
without MPS. Only ``MatmulTiledRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch, PipelineLimits
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_report import _token
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_metal_runtime import TORCH_MATMUL, operand_shape
from .matmul_metal_source import KERNELS, MATMUL_X_COLUMNS, SHADER_SOURCE
from .matmul_tiled_source import MatmulTiledError
from .matmul_tiled_variants import (
    TiledVariant,
    VariantAdmission,
    library_source,
)
from .memory_access_tiling import MatmulShape

NAIVE_KERNEL = MATMUL_X_COLUMNS
MPS_SWITCHES = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")


class MatmulTiledUnavailableError(MatmulTiledError):
    """Raised when this host cannot compile or run the tiled matmul kernels."""


def mps_switches(environ: Mapping[str, str]) -> dict[str, str | None]:
    """Copy each MPS switch as a short token; ``None`` means it is unset."""

    return {
        name: None if environ.get(name) is None else _token(environ.get(name))
        for name in MPS_SWITCHES
    }


def pipeline_limits(kernel: object) -> PipelineLimits:
    """Read the limits one compiled pipeline reports about itself."""

    return PipelineLimits(
        max_threads_per_threadgroup=kernel.max_threads_per_threadgroup,
        thread_execution_width=kernel.thread_execution_width,
        static_threadgroup_memory_bytes=kernel.static_thread_group_memory_length,
    )


@dataclass(frozen=True)
class SideCheck:
    """How one timed side's latest output compares with the naive output.

    ``disagreement`` is the largest ``|side - naive| / (|A| @ |B|)``,
    computed on the device in float32.
    """

    same_bits_as_naive: bool
    disagreement: float


@dataclass(frozen=True)
class TiledWorkload:
    """One benchmark point's device buffers, bound as no-argument calls.

    ``sides`` runs the naive kernel first, then the variants, then
    ``torch.matmul``. Each call only queues work; ``synchronize`` waits for
    it. ``check`` waits too, then compares every other side with the naive
    output.
    """

    sides: Mapping[str, Callable[[], object]]
    synchronize: Callable[[], None]
    check: Callable[[], Mapping[str, SideCheck]]

    @property
    def calls(self) -> tuple[Callable[[], object], ...]:
        """The side calls in timing order, as the rotating timer takes them."""

        return tuple(self.sides.values())


def _operands(left: object, right: object) -> tuple[numpy.ndarray, numpy.ndarray]:
    if (
        not isinstance(left, numpy.ndarray)
        or not isinstance(right, numpy.ndarray)
        or left.dtype != numpy.float32
        or right.dtype != numpy.float32
    ):
        raise MatmulTiledError("operands must be float32 arrays")
    operand_shape(left, right)
    return numpy.ascontiguousarray(left), numpy.ascontiguousarray(right)


def _compile(torch: object, source: str, label: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise MatmulTiledUnavailableError(
            f"the Metal compiler rejected {label}: {error}"
        ) from error


class MatmulTiledRuntime:
    """Both compiled libraries on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        variants: tuple[TiledVariant, ...],
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        tiled_source = library_source(variants)
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MatmulTiledUnavailableError(availability.message)
        torch = import_module("torch")
        naive = _compile(torch, SHADER_SOURCE, "the Day 74 naive source")
        tiled = _compile(torch, tiled_source, "the tiled source")
        self._torch = torch
        self.variants = {variant.name: variant for variant in variants}
        self._functions = {name: getattr(naive, name) for name in KERNELS}
        self._functions.update({name: getattr(tiled, name) for name in self.variants})
        self._casts = {name: spec.arg_casts for name, spec in KERNELS.items()}
        self._casts.update(
            {name: variant.arg_casts for name, variant in self.variants.items()}
        )
        self.torch_version = availability.torch_version
        self.float32_matmul_precision = str(torch.get_float32_matmul_precision())
        self.limits = {
            name: pipeline_limits(function)
            for name, function in self._functions.items()
        }
        self.admissions = {
            name: VariantAdmission(variant, self.limits[name])
            for name, variant in self.variants.items()
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        return self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")

    def _admit(self, name: str, launch: object, shape: MatmulShape) -> CheckedLaunch:
        """Accept only a checked launch of a runnable kernel over its own grid."""

        if name != NAIVE_KERNEL and name not in self.admissions:
            raise MatmulTiledError(
                f"{name!r} is neither the naive kernel nor a variant"
            )
        if name in self.admissions and not self.admissions[name].admitted:
            raise MatmulTiledError(
                f"{name} was refused by its pipeline: {self.admissions[name].reason}"
            )
        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[name]
            or launch.geometry.extent != (shape.n, shape.m)
            or (
                name in self.variants
                and launch.geometry.group_size != self.variants[name].group_size
            )
        ):
            raise MatmulTiledError(
                f"{name} runs only with a launch checked against its own pipeline "
                "over the x_columns grid of this shape, in its own group"
            )
        return launch

    def _kernel(
        self, name: str, launch: CheckedLaunch, shape: MatmulShape, *buffers: object
    ) -> partial:
        """Bind one admitted launch with its geometry and 32-bit scalar casts."""

        return partial(
            self._functions[name],
            *buffers,
            shape.m,
            shape.n,
            shape.k,
            **launch.geometry.launch_arguments(),
            arg_casts=self._casts[name],
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

        a, b = _operands(left, right)
        shape = operand_shape(a, b)
        admitted = self._admit(name, launch, shape)
        elements = shape.m * shape.n + admitted.geometry.padding_threads
        output = self._to(float32_sentinels(elements))
        self._kernel(name, admitted, shape, output, self._to(a), self._to(b))()
        self.synchronize()
        return output.cpu().numpy()

    def workload(
        self,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launches: Mapping[str, CheckedLaunch],
    ) -> TiledWorkload:
        """Place one point's operands on the device and bind every side once.

        ``launches`` holds the naive kernel first and then the admitted
        variants, in timing order; ``torch.matmul`` writes into its own
        preallocated output.
        """

        torch = self._torch
        a_host, b_host = _operands(left, right)
        shape = operand_shape(a_host, b_host)
        names = tuple(launches)
        if not names or names[0] != NAIVE_KERNEL or TORCH_MATMUL in names:
            raise MatmulTiledError("a workload starts with the naive kernel's launch")
        admitted = {name: self._admit(name, launches[name], shape) for name in names}
        a, b = self._to(a_host), self._to(b_host)
        outputs = {
            name: torch.empty((shape.m, shape.n), dtype=torch.float32, device="mps")
            for name in (*names, TORCH_MATMUL)
        }
        sides = {
            name: self._kernel(name, admitted[name], shape, outputs[name], a, b)
            for name in names
        }
        sides[TORCH_MATMUL] = partial(torch.matmul, a, b, out=outputs[TORCH_MATMUL])

        def check() -> dict[str, SideCheck]:
            self.synchronize()
            naive = outputs[NAIVE_KERNEL]
            scale = torch.matmul(a.abs(), b.abs())
            results = {}
            for name in (*names[1:], TORCH_MATMUL):
                output = outputs[name]
                same = bool(
                    torch.equal(output.view(torch.int32), naive.view(torch.int32))
                )
                difference = (output - naive).abs_().div_(scale)
                results[name] = SideCheck(same, float(difference.max().item()))
            self.synchronize()
            return results

        return TiledWorkload(sides=sides, synchronize=self.synchronize, check=check)
