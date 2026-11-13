"""Theoretical SM residency limits under the CUDA occupancy-calculator rules.

The rules mirror ``cudaOccMaxActiveBlocksPerMultiprocessor`` in the CUDA 12.6
``cuda_occupancy.h`` for architectures without partitioned global caching,
which covers every pinned compute capability. Each limit is the number of
blocks one SM could hold if that resource were the only constraint. These are
calculator predictions from declared inputs, not achieved occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simt_device import SimtArchitecture, require_int


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _round_up(value: int, unit: int) -> int:
    return _ceil_div(value, unit) * unit


def _require_architecture(architecture: object) -> SimtArchitecture:
    if not isinstance(architecture, SimtArchitecture):
        raise TypeError("architecture must be SimtArchitecture")
    return architecture


def warps_per_block(architecture: SimtArchitecture, threads_per_block: int) -> int:
    """Return warp slots one block occupies; a partial warp takes a full slot."""

    checked = _require_architecture(architecture)
    threads = require_int(
        threads_per_block,
        "threads_per_block",
        minimum=1,
        maximum=checked.max_threads_per_block,
    )
    return _ceil_div(threads, checked.warp_size)


def warp_slot_limit(architecture: SimtArchitecture, threads_per_block: int) -> int:
    """Return blocks per SM allowed by the resident-warp limit alone."""

    warps = warps_per_block(architecture, threads_per_block)
    return architecture.max_warps_per_sm // warps


def block_slot_limit(architecture: SimtArchitecture) -> int:
    """Return blocks per SM allowed by the resident-block limit alone."""

    return _require_architecture(architecture).max_blocks_per_sm


@dataclass(frozen=True)
class RegisterLimit:
    """Register-file allocation for one block and the residency it allows.

    ``blockers`` names every failed fit check: the per-thread maximum, the
    per-block register file (software check), and the per-block check that
    rounds warps up to the sub-partition count (hardware check). Any blocker
    makes ``blocks`` zero.
    """

    registers_per_thread: int
    warps_per_block: int
    registers_per_warp: int
    registers_per_block: int
    hardware_check_registers_per_block: int
    warps_per_sub_partition: int
    blocks: int
    blockers: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "registers_per_thread": self.registers_per_thread,
            "warps_per_block": self.warps_per_block,
            "registers_per_warp": self.registers_per_warp,
            "registers_per_block": self.registers_per_block,
            "hardware_check_registers_per_block": (
                self.hardware_check_registers_per_block
            ),
            "warps_per_sub_partition": self.warps_per_sub_partition,
            "blocks": self.blocks,
            "blockers": list(self.blockers),
        }


def register_limit(
    architecture: SimtArchitecture,
    *,
    threads_per_block: int,
    registers_per_thread: int,
) -> RegisterLimit:
    """Apply warp-granular register allocation across SM sub-partitions.

    Registers per warp are ``registers_per_thread * warp_size`` rounded up to
    the allocation unit. Warps fit per sub-partition, never across them, so
    the SM holds ``sub_partitions * floor(sub_partition_registers /
    registers_per_warp)`` warps.
    """

    checked = _require_architecture(architecture)
    warps = warps_per_block(checked, threads_per_block)
    registers = require_int(
        registers_per_thread, "registers_per_thread", minimum=1, maximum=1024
    )
    per_warp = _round_up(
        registers * checked.warp_size, checked.register_allocation_unit
    )
    per_block = per_warp * warps
    hardware = per_warp * _round_up(warps, checked.register_sub_partitions_per_sm)
    per_sub_partition = (
        checked.registers_per_sm // checked.register_sub_partitions_per_sm
    ) // per_warp
    blockers = tuple(
        name
        for name, failed in (
            ("registers_per_thread", registers > checked.max_registers_per_thread),
            ("registers_per_block", per_block > checked.registers_per_block),
            (
                "registers_per_block_partition_check",
                hardware > checked.registers_per_block,
            ),
        )
        if failed
    )
    blocks = (
        0
        if blockers
        else per_sub_partition * checked.register_sub_partitions_per_sm // warps
    )
    return RegisterLimit(
        registers_per_thread=registers,
        warps_per_block=warps,
        registers_per_warp=per_warp,
        registers_per_block=per_block,
        hardware_check_registers_per_block=hardware,
        warps_per_sub_partition=per_sub_partition,
        blocks=blocks,
        blockers=blockers,
    )
