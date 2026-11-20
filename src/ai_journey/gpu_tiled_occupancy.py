"""Declared compute capability 8.0 occupancy of the tiled CUDA kernel.

The estimate follows ``matmul_model_occupancy``: the SIMT lab's calculator
rules in ``simt_occupancy`` and its pinned 8.0 record, which must repeat
every Day 73 launch limit. A T x T block declares T * T threads and the
static shared memory of its two float tiles, ``2 * T * T * 4`` bytes from
``gpu_tiled_cuda``. It uses no dynamic shared memory and no opt-in. This lab
never compiles the kernel, so it cannot read the register count that a
compiler would assign. Each register count is an assumption, and the
estimates show how the count changes residency.

With the 1 KB that the driver reserves per block, one block allocates 1536,
3072 or 9216 bytes of the 164 KB that one SM holds. Thus shared memory does
not limit residency at these tiles; the warp slots and the registers do.
Theoretical occupancy is the calculator's prediction of resident warps. It is
not achieved occupancy, and it says nothing about latency hiding or speed.
"""

from __future__ import annotations

from .gpu_tiled_cuda import GpuTiledError, require_tile, shared_memory_bytes
from .matmul_model_occupancy import ARCHITECTURE, launch_limits_agree
from .simt_occupancy import KernelResources, OccupancyResult, analyze_occupancy


def tiled_matmul_resources(*, tile: int, registers_per_thread: int) -> KernelResources:
    """Declare one T x T block: its threads, a register count and its tiles."""

    width = require_tile(tile)
    return KernelResources(
        threads_per_block=width * width,
        registers_per_thread=registers_per_thread,
        static_shared_bytes=shared_memory_bytes(width),
        dynamic_shared_bytes=0,
        shared_memory_opt_in=False,
    )


def estimate_occupancy(*, tile: int, registers_per_thread: int) -> OccupancyResult:
    """Return the SIMT lab's theoretical occupancy for the declared inputs.

    A count that leaves no block resident raises the SIMT lab's
    ``LaunchNotResidentError``, which names each failed check.
    """

    if not launch_limits_agree():
        raise GpuTiledError("the SIMT 8.0 record and the Day 73 launch limits disagree")
    return analyze_occupancy(
        ARCHITECTURE,
        tiled_matmul_resources(tile=tile, registers_per_thread=registers_per_thread),
    )
