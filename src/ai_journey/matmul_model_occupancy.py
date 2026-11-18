"""Theoretical occupancy of the naive matmul's CUDA kernel on compute capability 8.0.

The estimate reuses the SIMT lab's calculator rules in ``simt_occupancy`` and
its pinned 8.0 record. The inputs are declared: the threads per block of the
pinned launch, one register count per thread, and 0 bytes of static and
dynamic shared memory, which is all that the naive kernel uses. This lab
never compiles the kernel, so it cannot read the register count that a
compiler would assign. Each count is an assumption, and the estimates show
how the count changes residency. The 8.0 record must repeat every Day 73
launch limit, so the launch and the occupancy use one set of limits.

Theoretical occupancy is the calculator's prediction of resident warps. It is
not achieved occupancy, and it says nothing about latency hiding or speed.
"""

from __future__ import annotations

from fractions import Fraction

from .kernel_launch_config import COMPUTE_CAPABILITY, LAUNCH_LIMITS
from .matmul_model_reference import MatmulModelError
from .matmul_model_traffic import ratio_mapping
from .simt_device import PINNED_ARCHITECTURES
from .simt_occupancy import KernelResources, OccupancyResult, analyze_occupancy

ARCHITECTURE = PINNED_ARCHITECTURES[COMPUTE_CAPABILITY]
ARCHITECTURE_FIELDS = (
    "compute_capability",
    "warp_size",
    "max_threads_per_block",
    "max_warps_per_sm",
    "max_blocks_per_sm",
    "registers_per_sm",
    "registers_per_block",
    "max_registers_per_thread",
    "register_allocation_unit",
    "register_sub_partitions_per_sm",
    "shared_memory_per_sm_bytes",
    "shared_memory_allocation_unit_bytes",
    "reserved_shared_memory_per_block_bytes",
)


def launch_limits_agree() -> bool:
    """Return whether the SIMT 8.0 record repeats every Day 73 launch limit."""

    return dict(LAUNCH_LIMITS) == {
        name: getattr(ARCHITECTURE, name) for name in LAUNCH_LIMITS
    }


def architecture_mapping() -> dict[str, object]:
    """Return the 8.0 values that the estimate reads, with their sources."""

    return {
        **{name: getattr(ARCHITECTURE, name) for name in ARCHITECTURE_FIELDS},
        "sources": list(ARCHITECTURE.sources),
        "launch_limits_agree": launch_limits_agree(),
    }


def naive_matmul_resources(
    *, threads_per_block: int, registers_per_thread: int
) -> KernelResources:
    """Declare the naive kernel's resources: no shared memory of any kind."""

    return KernelResources(
        threads_per_block=threads_per_block,
        registers_per_thread=registers_per_thread,
        static_shared_bytes=0,
        dynamic_shared_bytes=0,
        shared_memory_opt_in=False,
    )


def estimate_occupancy(
    *, threads_per_block: int, registers_per_thread: int
) -> OccupancyResult:
    """Return the SIMT lab's theoretical occupancy for the declared inputs."""

    if not launch_limits_agree():
        raise MatmulModelError(
            "the SIMT 8.0 record and the Day 73 launch limits disagree"
        )
    return analyze_occupancy(
        ARCHITECTURE,
        naive_matmul_resources(
            threads_per_block=threads_per_block,
            registers_per_thread=registers_per_thread,
        ),
    )


def occupancy_mapping(result: OccupancyResult) -> dict[str, object]:
    """Return the SIMT lab's mapping with the occupancy as an exact ratio."""

    if not isinstance(result, OccupancyResult):
        raise TypeError("result must be OccupancyResult")
    return {
        **result.to_mapping(),
        "occupancy": ratio_mapping(
            Fraction(result.active_warps_per_sm, result.max_warps_per_sm)
        ),
    }
