"""Declared compute capability 8.0 occupancy of the Day 77 CUDA kernels.

The estimate follows ``gpu_tiled_occupancy``: the calculator rules of
``simt_occupancy`` and the pinned 8.0 record, which must repeat every Day 73
launch limit. Each kernel declares the threads of its block and the static
shared memory of its two tiles. It uses no dynamic shared memory and no
opt-in. This lab never compiles a kernel, so it cannot read the register
count that ``ptxas`` would assign. Each count is an assumption, and a sweep
of three counts shows how the count changes residency.

A count must at least hold the live floats of one thread: its accumulators
and the two operands of an fma in the coarsened kernel, or its TM * TN
accumulators and its TM + TN operand registers in the blocked kernel. This
floor is necessary, not sufficient, because indices and pointers need
registers too. A later run must replace each count with the count that
``ptxas`` reports, and check that it reports no local memory for spills.

Theoretical occupancy is the calculator's prediction of resident warps. It is
not achieved occupancy, and it predicts no speed: the blocked kernels trade
resident warps for fewer shared and global loads per fma.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .gpu_checklist_cuda import (
    PINNED_KERNELS,
    ChecklistKernel,
    CoarsenedKernel,
    GpuChecklistError,
    require_kernel,
)
from .matmul_model_occupancy import ARCHITECTURE, launch_limits_agree, occupancy_mapping
from .simt_occupancy import KernelResources, OccupancyResult, analyze_occupancy

# Assumptions, not ptxas counts: three counts for each pinned kernel.
DECLARED_REGISTERS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "coarsened_matmul_t16_c2": (32, 40, 64),
        "coarsened_matmul_t16_c4": (32, 40, 64),
        "coarsened_matmul_t32_c2": (32, 40, 64),
        "blocked_matmul_m64_n64_k16_t4x4": (48, 64, 96),
        "blocked_matmul_m128_n128_k8_t8x8": (96, 128, 168),
    }
)
REGISTER_SOURCE = (
    "declared assumption; this lab compiles no kernel, so ptxas reports no count"
)


def live_floats(kernel: ChecklistKernel) -> int:
    """Return the floats that one thread keeps live in its inner loop."""

    checked = require_kernel(kernel)
    if type(checked) is CoarsenedKernel:
        return checked.coarsening + 2
    return checked.outputs_per_thread + checked.thread_m + checked.thread_n


def kernel_resources(
    kernel: ChecklistKernel, *, registers_per_thread: int
) -> KernelResources:
    """Declare one block: its threads, a register count and its two tiles."""

    checked = require_kernel(kernel)
    if type(registers_per_thread) is not int:
        raise TypeError("registers_per_thread must be an integer")
    floor = live_floats(checked)
    if not floor <= registers_per_thread <= ARCHITECTURE.max_registers_per_thread:
        raise GpuChecklistError(
            f"{checked.name} needs {floor} to "
            f"{ARCHITECTURE.max_registers_per_thread} registers per thread"
        )
    return KernelResources(
        threads_per_block=checked.threads_per_block,
        registers_per_thread=registers_per_thread,
        static_shared_bytes=checked.shared_memory_bytes,
        dynamic_shared_bytes=0,
        shared_memory_opt_in=False,
    )


def estimate_occupancy(
    kernel: ChecklistKernel, *, registers_per_thread: int
) -> OccupancyResult:
    """Return the SIMT lab's theoretical occupancy for the declared inputs.

    A count that leaves no block resident raises the SIMT lab's
    ``LaunchNotResidentError``, which names each failed check.
    """

    if not launch_limits_agree():
        raise GpuChecklistError(
            "the SIMT 8.0 record and the Day 73 launch limits disagree"
        )
    return analyze_occupancy(
        ARCHITECTURE,
        kernel_resources(kernel, registers_per_thread=registers_per_thread),
    )


def declared_registers(kernel: ChecklistKernel) -> tuple[int, ...]:
    """Return the declared counts of a pinned kernel, in increasing order."""

    checked = require_kernel(kernel)
    if checked not in PINNED_KERNELS:
        raise GpuChecklistError(f"{checked.name} has no declared register counts")
    return DECLARED_REGISTERS[checked.name]


def estimate_mapping(
    kernel: ChecklistKernel, *, registers_per_thread: int
) -> dict[str, object]:
    """Return one estimate with the occupancy as an exact ratio and its source.

    ``kernel`` in the mapping holds the declared resources, as in the SIMT
    lab, so ``name`` holds the kernel name.
    """

    checked = require_kernel(kernel)
    return {
        "name": checked.name,
        "register_source": REGISTER_SOURCE,
        **occupancy_mapping(
            estimate_occupancy(checked, registers_per_thread=registers_per_thread)
        ),
    }


def declared_estimates() -> tuple[dict[str, object], ...]:
    """Return the estimate of each pinned kernel at each declared count."""

    return tuple(
        estimate_mapping(kernel, registers_per_thread=count)
        for kernel in PINNED_KERNELS
        for count in declared_registers(kernel)
    )
