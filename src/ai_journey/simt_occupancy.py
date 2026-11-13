"""Theoretical SM occupancy under the CUDA occupancy-calculator rules.

The rules mirror ``cudaOccMaxActiveBlocksPerMultiprocessor`` in the CUDA 12.6
``cuda_occupancy.h`` for architectures without partitioned global caching,
which covers every pinned compute capability. Each limit is the number of
blocks one SM could hold if that resource were the only constraint. These are
calculator predictions from declared inputs, not achieved occupancy. Two
deliberate differences: register counts above the guide's 255-per-thread
maximum are rejected (the header's check admits 256), and the header's
block-barrier limit, which only applies from compute capability 9.0, is not
modeled.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simt_device import SimtArchitecture, SimtError, require_int


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


def carveout_for_preference(architecture: SimtArchitecture, percent: int) -> int:
    """Map a carveout percentage to the next supported shared-memory capacity.

    This is the guide's documented rounding (50% on 8.0 is 100 KB, not 82 KB);
    the driver may still pick another capacity at run time, which is not
    modeled.
    """

    checked = _require_architecture(architecture)
    share = require_int(percent, "carveout percent", minimum=0, maximum=100)
    target = share * checked.shared_memory_per_sm_bytes // 100
    return next(
        size for size in checked.shared_memory_carveouts_bytes if size >= target
    )


@dataclass(frozen=True)
class SharedMemoryLimit:
    """Shared-memory allocation for one block against the configured carveout.

    ``blocks`` is ``None`` when a block allocates no shared memory at all, so
    shared memory cannot limit residency. When one block needs more than the
    configured carveout, the effective carveout is the smallest supported
    capacity that holds it, as ``cuda_occupancy.h`` assumes on Volta and later.
    """

    requested_bytes_per_block: int
    allocated_bytes_per_block: int
    per_block_limit_bytes: int
    configured_carveout_bytes: int
    effective_carveout_bytes: int
    blocks: int | None
    blockers: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "requested_bytes_per_block": self.requested_bytes_per_block,
            "allocated_bytes_per_block": self.allocated_bytes_per_block,
            "per_block_limit_bytes": self.per_block_limit_bytes,
            "configured_carveout_bytes": self.configured_carveout_bytes,
            "effective_carveout_bytes": self.effective_carveout_bytes,
            "blocks": self.blocks,
            "blockers": list(self.blockers),
        }


def shared_memory_limit(
    architecture: SimtArchitecture,
    *,
    static_shared_bytes: int,
    dynamic_shared_bytes: int,
    shared_memory_opt_in: bool,
    carveout_bytes: int | None = None,
) -> SharedMemoryLimit:
    """Apply shared-memory allocation units, per-block limits, and carveouts.

    A block requests static + dynamic + driver-reserved bytes, rounded up to
    the allocation unit. The per-block limit is the 48 KB default, or the
    opt-in maximum when the kernel opted in and needs more than the default;
    from compute capability 8.0 the reservation is added to that limit.
    Opting in is modeled as setting the maximum dynamic size to the requested
    dynamic bytes.
    """

    checked = _require_architecture(architecture)
    limit = 4 * 2**20
    static = require_int(
        static_shared_bytes, "static_shared_bytes", minimum=0, maximum=limit
    )
    dynamic = require_int(
        dynamic_shared_bytes, "dynamic_shared_bytes", minimum=0, maximum=limit
    )
    if type(shared_memory_opt_in) is not bool:
        raise TypeError("shared_memory_opt_in must be a boolean")
    if static > checked.shared_memory_per_block_bytes:
        raise SimtError(
            "static shared memory above the default per-block limit must be "
            "dynamic shared memory with an explicit opt-in"
        )
    if carveout_bytes is None:
        configured = checked.shared_memory_per_sm_bytes
    elif (
        type(carveout_bytes) is not int
        or carveout_bytes not in checked.shared_memory_carveouts_bytes
    ):
        raise SimtError(
            "carveout_bytes must be a supported shared memory capacity for "
            f"compute capability {checked.compute_capability}"
        )
    else:
        configured = carveout_bytes
    reserved = checked.reserved_shared_memory_per_block_bytes
    requested = static + dynamic + reserved
    allocated = _round_up(requested, checked.shared_memory_allocation_unit_bytes)
    needs_optin = shared_memory_opt_in and requested > (
        checked.shared_memory_per_block_bytes
    )
    per_block = (
        checked.shared_memory_per_block_optin_bytes
        if needs_optin
        else checked.shared_memory_per_block_bytes
    ) + (reserved if checked.major >= 8 else 0)
    if allocated > per_block:
        return SharedMemoryLimit(
            requested_bytes_per_block=requested,
            allocated_bytes_per_block=allocated,
            per_block_limit_bytes=per_block,
            configured_carveout_bytes=configured,
            effective_carveout_bytes=configured,
            blocks=0,
            blockers=("shared_memory_per_block",),
        )
    effective = (
        configured
        if configured >= allocated
        else next(
            size for size in checked.shared_memory_carveouts_bytes if size >= allocated
        )
    )
    return SharedMemoryLimit(
        requested_bytes_per_block=requested,
        allocated_bytes_per_block=allocated,
        per_block_limit_bytes=per_block,
        configured_carveout_bytes=configured,
        effective_carveout_bytes=effective,
        blocks=effective // allocated if allocated else None,
        blockers=(),
    )


@dataclass(frozen=True)
class KernelResources:
    """Declared per-block launch resources of one kernel; nothing is compiled."""

    threads_per_block: int
    registers_per_thread: int
    static_shared_bytes: int = 0
    dynamic_shared_bytes: int = 0
    shared_memory_opt_in: bool = False

    def __post_init__(self) -> None:
        require_int(
            self.threads_per_block, "threads_per_block", minimum=1, maximum=4096
        )
        require_int(
            self.registers_per_thread, "registers_per_thread", minimum=1, maximum=1024
        )
        for name in ("static_shared_bytes", "dynamic_shared_bytes"):
            require_int(getattr(self, name), name, minimum=0, maximum=4 * 2**20)
        if type(self.shared_memory_opt_in) is not bool:
            raise TypeError("shared_memory_opt_in must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        return {
            "threads_per_block": self.threads_per_block,
            "registers_per_thread": self.registers_per_thread,
            "static_shared_bytes": self.static_shared_bytes,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "shared_memory_opt_in": self.shared_memory_opt_in,
        }


class LaunchNotResidentError(SimtError):
    """Raised when not even one block of a launch fits on one SM."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__("launch cannot be resident on one SM: " + ", ".join(blockers))


LIMIT_NAMES = ("warps", "registers", "shared_memory", "blocks")


@dataclass(frozen=True)
class OccupancyResult:
    """Theoretical occupancy and the per-resource limits that produced it."""

    compute_capability: str
    kernel: KernelResources
    warps_per_block: int
    warp_limit_blocks: int
    block_limit_blocks: int
    registers: RegisterLimit
    shared_memory: SharedMemoryLimit
    active_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    limiting_factors: tuple[str, ...]

    @property
    def occupancy(self) -> float:
        """Return active warps divided by the SM's maximum resident warps."""

        return self.active_warps_per_sm / self.max_warps_per_sm

    def to_mapping(self) -> dict[str, object]:
        return {
            "compute_capability": self.compute_capability,
            "kernel": self.kernel.to_mapping(),
            "warps_per_block": self.warps_per_block,
            "limits_blocks_per_sm": {
                "warps": self.warp_limit_blocks,
                "registers": self.registers.blocks,
                "shared_memory": self.shared_memory.blocks,
                "blocks": self.block_limit_blocks,
            },
            "registers": self.registers.to_mapping(),
            "shared_memory": self.shared_memory.to_mapping(),
            "active_blocks_per_sm": self.active_blocks_per_sm,
            "active_warps_per_sm": self.active_warps_per_sm,
            "max_warps_per_sm": self.max_warps_per_sm,
            "occupancy": self.occupancy,
            "limiting_factors": list(self.limiting_factors),
        }


def analyze_occupancy(
    architecture: SimtArchitecture,
    kernel: KernelResources,
    *,
    carveout_bytes: int | None = None,
) -> OccupancyResult:
    """Return active blocks as the minimum of every per-resource limit.

    Every resource whose limit equals that minimum is reported as limiting.
    Launches that cannot place a single block on an SM are rejected with
    :class:`LaunchNotResidentError` naming each failed check.
    """

    checked = _require_architecture(architecture)
    if not isinstance(kernel, KernelResources):
        raise TypeError("kernel must be KernelResources")
    warps = warps_per_block(checked, kernel.threads_per_block)
    registers = register_limit(
        checked,
        threads_per_block=kernel.threads_per_block,
        registers_per_thread=kernel.registers_per_thread,
    )
    shared = shared_memory_limit(
        checked,
        static_shared_bytes=kernel.static_shared_bytes,
        dynamic_shared_bytes=kernel.dynamic_shared_bytes,
        shared_memory_opt_in=kernel.shared_memory_opt_in,
        carveout_bytes=carveout_bytes,
    )
    blockers = registers.blockers + shared.blockers
    if blockers:
        raise LaunchNotResidentError(blockers)
    limits = {
        "warps": warp_slot_limit(checked, kernel.threads_per_block),
        "registers": registers.blocks,
        "shared_memory": shared.blocks,
        "blocks": block_slot_limit(checked),
    }
    active = min(value for value in limits.values() if value is not None)
    return OccupancyResult(
        compute_capability=checked.compute_capability,
        kernel=kernel,
        warps_per_block=warps,
        warp_limit_blocks=limits["warps"],
        block_limit_blocks=limits["blocks"],
        registers=registers,
        shared_memory=shared,
        active_blocks_per_sm=active,
        active_warps_per_sm=active * warps,
        max_warps_per_sm=checked.max_warps_per_sm,
        limiting_factors=tuple(name for name in LIMIT_NAMES if limits[name] == active),
    )
