"""Run every checklist kernel and ``torch.matmul`` on the MPS device.

Importing this module never imports torch, so a host without torch can still
verify reports. ``MatmulChecklistRuntime`` first copies the two PyTorch MPS
switches from the environment, as the Day 76 runtime does:
``PYTORCH_MPS_FAST_MATH`` and ``PYTORCH_MPS_PREFER_METAL`` can each change
what ``torch.matmul`` runs, and the lab cannot see that kernel. It then runs
the Day 73 ``probe_metal``, which reads the operating system before it
imports torch, and compiles two libraries through the public
``torch.mps.compile_shader``: the Day 74 ``SHADER_SOURCE`` unchanged, and
the checklist library of the Day 76 tile-32 kernel and the Day 77 kernels.
It reads the limits of every pipeline and each pipeline's verdict on the
threadgroup memory of its kernel.

A refused kernel is data, not an error: the runtime keeps its verdict and
never launches it. A launch runs only when it equals the variant's own
launch for the shape, checked against that kernel's pipeline. The Day 77
kernels load ``float4`` values, so every operand is a fresh contiguous MPS
tensor at storage offset 0, whose buffer starts 16-byte aligned. Scalars
reach every kernel as 32-bit integers through ``arg_casts``.

The runtime takes and returns NumPy arrays, so the correctness and benchmark
code never touch torch, and a stand-in can replace the runtime on a host
without MPS. Only ``MatmulChecklistRuntime`` itself makes a measured report.
"""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

import numpy

from .kernel_metal_geometry import CheckedLaunch
from .kernel_metal_reference import float32_sentinels
from .kernel_metal_runtime import BACKEND, probe_metal
from .matmul_checklist_source import MatmulChecklistError
from .matmul_checklist_steps import REFERENCE
from .matmul_checklist_variants import (
    LIBRARY_VARIANTS,
    NAIVE_VARIANTS,
    VARIANTS,
    ChecklistAdmission,
    checklist_library,
    guard_elements,
    variant_launch,
)
from .matmul_metal_runtime import TORCH_MATMUL, operand_shape
from .matmul_metal_source import SHADER_SOURCE
from .matmul_tiled_runtime import SideCheck, mps_switches, pipeline_limits
from .memory_access_tiling import MatmulShape


class MatmulChecklistUnavailableError(MatmulChecklistError):
    """Raised when this host cannot compile or run the checklist kernels."""


def checked_launch(
    limits: Mapping[str, object],
    name: str,
    shape: MatmulShape,
    naive_group: tuple[int, int],
) -> CheckedLaunch:
    """Size one kernel's launch for a shape and check it against its pipeline."""

    if name not in VARIANTS:
        raise MatmulChecklistError(f"{name!r} is not a registered variant")
    return CheckedLaunch(
        variant_launch(VARIANTS[name], shape, naive_group), limits[name]
    )


@dataclass(frozen=True)
class ChecklistWorkload:
    """One benchmark point's device buffers, bound as no-argument calls.

    ``sides`` starts with the S0 kernel, then the other kernels in registry
    order, then ``torch.matmul``. Each call only queues work;
    ``synchronize`` waits for it. ``check`` waits too, then compares every
    other side with the S0 output.
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
        raise MatmulChecklistError("operands must be float32 arrays")
    operand_shape(left, right)
    return numpy.ascontiguousarray(left), numpy.ascontiguousarray(right)


def _compile(torch: object, source: str, label: str) -> object:
    try:
        return torch.mps.compile_shader(source)
    except RuntimeError as error:
        raise MatmulChecklistUnavailableError(
            f"the Metal compiler rejected {label}: {error}"
        ) from error


class MatmulChecklistRuntime:
    """Both compiled libraries on the MPS device, NumPy arrays in and out."""

    backend = BACKEND

    def __init__(
        self,
        *,
        system: Callable[[], str] = platform.system,
        import_module: Callable[[str], object] = importlib.import_module,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        library_source = checklist_library()
        self.mps_switches = mps_switches(environ)
        availability = probe_metal(system=system, import_module=import_module)
        if not availability.available:
            raise MatmulChecklistUnavailableError(availability.message)
        torch = import_module("torch")
        naive = _compile(torch, SHADER_SOURCE, "the Day 74 naive source")
        library = _compile(torch, library_source, "the checklist library")
        self._torch = torch
        self.library_source = library_source
        self._functions = {name: getattr(naive, name) for name in NAIVE_VARIANTS}
        self._functions.update(
            {name: getattr(library, name) for name in LIBRARY_VARIANTS}
        )
        self.torch_version = availability.torch_version
        self.float32_matmul_precision = str(torch.get_float32_matmul_precision())
        self.limits = {
            name: pipeline_limits(self._functions[name]) for name in VARIANTS
        }
        self.admissions = {
            name: ChecklistAdmission(VARIANTS[name], self.limits[name])
            for name in LIBRARY_VARIANTS
        }

    def synchronize(self) -> None:
        """Wait until every queued MPS command has finished."""

        self._torch.mps.synchronize()

    def _to(self, array: numpy.ndarray) -> object:
        tensor = self._torch.from_numpy(numpy.ascontiguousarray(array)).to("mps")
        # A float4 load needs a 16-byte aligned address; offset 0 starts the
        # binding at the start of its buffer.
        if tensor.storage_offset() != 0:
            raise MatmulChecklistError("an operand must start its device buffer")
        return tensor

    def _admit(self, name: str, launch: object, shape: MatmulShape) -> CheckedLaunch:
        """Accept only a checked launch of a runnable kernel in its own shape."""

        if name not in VARIANTS:
            raise MatmulChecklistError(f"{name!r} is not a registered variant")
        if name in self.admissions and not self.admissions[name].admitted:
            raise MatmulChecklistError(
                f"{name} was refused by its pipeline: {self.admissions[name].reason}"
            )
        if (
            not isinstance(launch, CheckedLaunch)
            or launch.limits != self.limits[name]
            or launch.geometry
            != variant_launch(VARIANTS[name], shape, launch.geometry.group_size)
        ):
            raise MatmulChecklistError(
                f"{name} runs only with its own launch for this shape, checked "
                "against its own pipeline"
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
            arg_casts=VARIANTS[name].arg_casts,
        )

    def run(
        self,
        name: str,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launch: CheckedLaunch,
    ) -> numpy.ndarray:
        """Run one kernel once and return its whole flat output buffer.

        The buffer starts as sentinels and is longer than C by every spare
        output slot of the launch, so a store past its guard shows in the
        tail.
        """

        a, b = _operands(left, right)
        shape = operand_shape(a, b)
        admitted = self._admit(name, launch, shape)
        elements = shape.m * shape.n + guard_elements(
            VARIANTS[name], shape, admitted.geometry
        )
        output = self._to(float32_sentinels(elements))
        self._kernel(name, admitted, shape, output, self._to(a), self._to(b))()
        self.synchronize()
        return output.cpu().numpy()

    def workload(
        self,
        left: numpy.ndarray,
        right: numpy.ndarray,
        launches: Mapping[str, CheckedLaunch],
    ) -> ChecklistWorkload:
        """Place one point's operands on the device and bind every side once.

        ``launches`` holds the S0 kernel first and then the admitted kernels,
        in timing order; ``torch.matmul`` writes into its own preallocated
        output.
        """

        torch = self._torch
        a_host, b_host = _operands(left, right)
        shape = operand_shape(a_host, b_host)
        names = tuple(launches)
        if not names or names[0] != REFERENCE or TORCH_MATMUL in names:
            raise MatmulChecklistError("a workload starts with the S0 kernel's launch")
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
            reference = outputs[REFERENCE]
            scale = torch.matmul(a.abs(), b.abs())
            results = {}
            for name in (*names[1:], TORCH_MATMUL):
                output = outputs[name]
                same = bool(
                    torch.equal(output.view(torch.int32), reference.view(torch.int32))
                )
                difference = (output - reference).abs_().div_(scale)
                results[name] = SideCheck(same, float(difference.max().item()))
            self.synchronize()
            return results

        return ChecklistWorkload(sides=sides, synchronize=self.synchronize, check=check)
