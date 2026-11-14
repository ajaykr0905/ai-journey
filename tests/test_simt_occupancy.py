import pytest

from ai_journey.simt_device import PINNED_ARCHITECTURES, SimtError
from ai_journey.simt_occupancy import (
    KernelResources,
    LaunchNotResidentError,
    analyze_occupancy,
    block_slot_limit,
    carveout_for_preference,
    register_limit,
    shared_memory_limit,
    warp_slot_limit,
    warps_per_block,
)

VOLTA = PINNED_ARCHITECTURES["7.0"]
TURING = PINNED_ARCHITECTURES["7.5"]
AMPERE = PINNED_ARCHITECTURES["8.0"]
KIB = 1024


def test_warp_slots_count_partial_warps_as_whole_warps():
    # A100: 256 threads = 8 warps; 64 warp slots / 8 = 8 blocks.
    assert warps_per_block(AMPERE, 256) == 8
    assert warp_slot_limit(AMPERE, 256) == 8
    # 33 threads still occupy 2 warp slots: 64 / 2 = 32 blocks.
    assert warps_per_block(AMPERE, 33) == 2
    assert warp_slot_limit(AMPERE, 33) == 32
    # 96 threads = 3 warps; floor(64 / 3) = 21 blocks leave one slot unused.
    assert warp_slot_limit(AMPERE, 96) == 21
    # T4: one 1024-thread block fills all 32 warp slots.
    assert warp_slot_limit(TURING, 1024) == 1
    # One-warp blocks: warp slots allow 64 (A100) and 32 (T4) blocks, but the
    # block-slot limits are 32 and 16.
    assert warp_slot_limit(AMPERE, 32) == 64
    assert block_slot_limit(AMPERE) == 32
    assert warp_slot_limit(TURING, 32) == 32
    assert block_slot_limit(TURING) == 16


def test_a100_register_limits_for_256_thread_blocks():
    # Registers per warp = registers per thread * 32, rounded up to 256.
    # Each of the 4 sub-partitions holds 65,536 / 4 = 16,384 registers.
    #  32 regs: 1,024 per warp; 16,384 / 1,024 = 16 warps per sub-partition;
    #           4 * 16 = 64 warps per SM; 64 / 8 warps per block = 8 blocks.
    #  64 regs: 2,048 per warp; 8 per sub-partition; 32 warps; 4 blocks.
    # 128 regs: 4,096 per warp; 4 per sub-partition; 16 warps; 2 blocks.
    expected = {32: (1_024, 16, 8), 64: (2_048, 8, 4), 128: (4_096, 4, 2)}
    for registers, (per_warp, per_partition, blocks) in expected.items():
        limit = register_limit(
            AMPERE, threads_per_block=256, registers_per_thread=registers
        )
        assert limit.registers_per_warp == per_warp
        assert limit.registers_per_block == per_warp * 8
        assert limit.hardware_check_registers_per_block == per_warp * 8
        assert limit.warps_per_sub_partition == per_partition
        assert limit.blocks == blocks
        assert limit.blockers == ()


def test_register_allocation_unit_rounding_costs_whole_blocks():
    # 33 and 40 registers both round to 1,280 per warp (33*32 = 1,056 and
    # 40*32 = 1,280 -> 5 units of 256), so floor(16,384 / 1,280) = 12 warps per
    # sub-partition, 48 per SM, and floor(48 / 8) = 6 blocks instead of 8.
    for registers in (33, 40):
        limit = register_limit(
            AMPERE, threads_per_block=256, registers_per_thread=registers
        )
        assert (limit.registers_per_warp, limit.blocks) == (1_280, 6)
    # 255 registers: 8,160 rounds to 8,192 per warp; 2 warps per sub-partition,
    # 8 per SM, exactly one 256-thread block.
    widest = register_limit(AMPERE, threads_per_block=256, registers_per_thread=255)
    assert (widest.registers_per_warp, widest.blocks) == (8_192, 1)


def test_registers_fit_per_sub_partition_not_across_the_whole_sm():
    # 64-thread blocks (2 warps) at 40 registers: 1,280 registers per warp.
    # Whole-SM division would give floor(65,536 / 1,280) = 51 warps -> 25 blocks,
    # but warps cannot straddle sub-partitions: 4 * 12 = 48 warps -> 24 blocks.
    limit = register_limit(AMPERE, threads_per_block=64, registers_per_thread=40)

    assert 65_536 // 1_280 // 2 == 25
    assert limit.warps_per_sub_partition == 12
    assert limit.blocks == 24


def test_hardware_check_rounds_block_warps_up_to_sub_partitions():
    # 800 threads = 25 warps at 80 registers: 2,560 per warp.
    # Software check: 25 * 2,560 = 64,000 <= 65,536 registers per block.
    # Hardware check: ceil(25 / 4) * 4 = 28 warps * 2,560 = 71,680 > 65,536.
    limit = register_limit(AMPERE, threads_per_block=800, registers_per_thread=80)

    assert limit.registers_per_block == 64_000
    assert limit.hardware_check_registers_per_block == 71_680
    assert limit.blockers == ("registers_per_block_partition_check",)
    assert limit.blocks == 0
    # One allocation unit less per warp (72 registers -> 2,304) passes both:
    # 25 * 2,304 = 57,600 and 28 * 2,304 = 64,512 registers.
    assert (
        register_limit(AMPERE, threads_per_block=800, registers_per_thread=72).blocks
        == 1
    )


def test_blocks_that_overflow_the_register_file_report_every_blocker():
    # 1,024 threads * 128 registers = 32 warps * 4,096 = 131,072 > 65,536.
    oversized = register_limit(VOLTA, threads_per_block=1024, registers_per_thread=128)
    assert oversized.blockers == (
        "registers_per_block",
        "registers_per_block_partition_check",
    )
    assert oversized.blocks == 0
    # 256 registers exceed the 255-register per-thread maximum in Table 21.
    too_wide = register_limit(VOLTA, threads_per_block=32, registers_per_thread=256)
    assert too_wide.blockers == ("registers_per_thread",)
    assert too_wide.to_mapping()["blockers"] == ["registers_per_thread"]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: warps_per_block(AMPERE, 0), SimtError, "between 1 and 1024"),
        (lambda: warps_per_block(AMPERE, 1025), SimtError, "between 1 and 1024"),
        (lambda: warp_slot_limit(AMPERE, True), TypeError, "must be an integer"),
        (lambda: block_slot_limit("8.0"), TypeError, "SimtArchitecture"),
        (
            lambda: register_limit(
                AMPERE, threads_per_block=32, registers_per_thread=0
            ),
            SimtError,
            "registers_per_thread must be between 1",
        ),
    ],
)
def test_residency_limits_reject_invalid_inputs(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_a100_256_thread_occupancy_falls_with_register_pressure():
    # Shared memory: only the 1 KB driver reserve, 164 KB / 1 KB = 164 blocks.
    #  32 regs: min(warps 8, registers 8, shared 164, blocks 32) = 8 blocks,
    #           8 * 8 = 64 warps of 64 -> 100%; warps and registers tie.
    #  64 regs: registers allow 4 blocks -> 32 warps -> 50%.
    # 128 regs: registers allow 2 blocks -> 16 warps -> 25%.
    expected = {
        32: (8, 64, 1.0, ("warps", "registers")),
        64: (4, 32, 0.5, ("registers",)),
        128: (2, 16, 0.25, ("registers",)),
    }
    for registers, (blocks, warps, occupancy, limiting) in expected.items():
        result = analyze_occupancy(AMPERE, KernelResources(256, registers))
        assert result.shared_memory.allocated_bytes_per_block == 1 * KIB
        assert result.shared_memory.blocks == 164
        assert result.active_blocks_per_sm == blocks
        assert result.active_warps_per_sm == warps
        assert result.occupancy == occupancy
        assert result.limiting_factors == limiting


def test_shared_memory_limited_occupancy_rounds_each_block_allocation():
    # A100, 256 threads, 32 registers, 48 KB static shared memory:
    # 49,152 + 1,024 reserved = 50,176 bytes, already a multiple of 128.
    # The default per-block limit is 48 KB + 1 KB reserve = 50,176, so it fits.
    # floor(167,936 carveout / 50,176) = 3 blocks -> 24 warps -> 37.5%.
    result = analyze_occupancy(
        AMPERE, KernelResources(256, 32, static_shared_bytes=48 * KIB)
    )

    assert result.shared_memory.allocated_bytes_per_block == 50_176
    assert result.shared_memory.per_block_limit_bytes == 50_176
    assert result.shared_memory.blocks == 3
    assert result.active_warps_per_sm == 24
    assert result.occupancy == 0.375
    assert result.limiting_factors == ("shared_memory",)
    # 100 bytes of static shared memory round up to one 128-byte unit after the
    # 1,024-byte reserve: 1,124 -> 1,152 bytes; floor(167,936 / 1,152) = 145.
    small = shared_memory_limit(
        AMPERE,
        static_shared_bytes=100,
        dynamic_shared_bytes=0,
        shared_memory_opt_in=False,
    )
    assert (small.allocated_bytes_per_block, small.blocks) == (1_152, 145)
    # Volta allocates in 256-byte units and reserves nothing: 100 -> 256.
    volta = shared_memory_limit(
        VOLTA,
        static_shared_bytes=100,
        dynamic_shared_bytes=0,
        shared_memory_opt_in=False,
    )
    assert (volta.allocated_bytes_per_block, volta.blocks) == (256, 384)


def test_configured_carveout_limits_blocks_and_grows_to_fit_one_block():
    kernel = KernelResources(256, 32, dynamic_shared_bytes=32 * KIB)
    # 32 KB + 1 KB = 33,792 bytes per block.
    # 100 KB carveout: floor(102,400 / 33,792) = 3 blocks.
    # 64 KB carveout: floor(65,536 / 33,792) = 1 block.
    # 32 KB carveout cannot hold one block, so the smallest capacity that does
    # (64 KB) is assumed, again 1 block.
    for carveout, effective, blocks in (
        (100 * KIB, 100 * KIB, 3),
        (64 * KIB, 64 * KIB, 1),
        (32 * KIB, 64 * KIB, 1),
    ):
        result = analyze_occupancy(AMPERE, kernel, carveout_bytes=carveout)
        assert result.shared_memory.configured_carveout_bytes == carveout
        assert result.shared_memory.effective_carveout_bytes == effective
        assert result.active_blocks_per_sm == blocks
    with pytest.raises(SimtError, match="supported shared memory capacity"):
        analyze_occupancy(AMPERE, kernel, carveout_bytes=48 * KIB)


def test_carveout_percentages_round_up_to_supported_capacities():
    # Guide 16.7.3: 50% of 164 KB maps to 100 KB on 8.0, not 82 KB.
    assert carveout_for_preference(AMPERE, 50) == 100 * KIB
    # Guide 16.6.4: 50% of 96 KB is 48 KB, which rounds up to 64 KB on 7.0.
    assert carveout_for_preference(VOLTA, 50) == 64 * KIB
    assert carveout_for_preference(VOLTA, 0) == 0
    # Turing has no 0 KB capacity; the smallest is 32 KB.
    assert carveout_for_preference(TURING, 0) == 32 * KIB
    assert carveout_for_preference(AMPERE, 100) == 164 * KIB


def test_block_count_limited_occupancy():
    # A100, one-warp blocks at 16 registers: warps allow 64 blocks, registers
    # 4 * (16,384 / 512) / 1 = 128, shared memory 164, but only 32 blocks may
    # be resident: 32 warps of 64 -> 50%, limited by the block count alone.
    ampere = analyze_occupancy(AMPERE, KernelResources(32, 16))
    assert ampere.registers.blocks == 128
    assert ampere.active_blocks_per_sm == 32
    assert ampere.occupancy == 0.5
    assert ampere.limiting_factors == ("blocks",)
    # T4, 64-thread blocks at 32 registers: warps 32 / 2 = 16 and blocks 16 tie;
    # Turing reserves no shared memory, so shared memory never limits here.
    turing = analyze_occupancy(TURING, KernelResources(64, 32))
    assert turing.shared_memory.blocks is None
    assert turing.active_warps_per_sm == 32
    assert turing.limiting_factors == ("warps", "blocks")
    assert turing.to_mapping()["limits_blocks_per_sm"]["shared_memory"] is None


def test_dynamic_shared_memory_above_48_kb_requires_opt_in():
    # 100 KB dynamic + 1 KB reserve = 103,424 bytes exceeds the 50,176-byte
    # default limit, so without opt-in no block can launch.
    without = KernelResources(256, 32, dynamic_shared_bytes=100 * KIB)
    with pytest.raises(LaunchNotResidentError, match="shared_memory_per_block"):
        analyze_occupancy(AMPERE, without)
    # With opt-in the limit is 163 KB + 1 KB = 167,936: floor(167,936 /
    # 103,424) = 1 block -> 8 warps -> 12.5%.
    opted = KernelResources(
        256, 32, dynamic_shared_bytes=100 * KIB, shared_memory_opt_in=True
    )
    result = analyze_occupancy(AMPERE, opted)
    assert result.shared_memory.per_block_limit_bytes == 167_936
    assert result.occupancy == 0.125
    # The 163 KB opt-in maximum fits exactly; one more byte does not.
    exact = KernelResources(
        32, 32, dynamic_shared_bytes=163 * KIB, shared_memory_opt_in=True
    )
    assert analyze_occupancy(AMPERE, exact).active_blocks_per_sm == 1
    over = KernelResources(
        32, 32, dynamic_shared_bytes=163 * KIB + 1, shared_memory_opt_in=True
    )
    with pytest.raises(LaunchNotResidentError) as error:
        analyze_occupancy(AMPERE, over)
    assert error.value.blockers == ("shared_memory_per_block",)
    # Volta has no reserve: 48 KB static + 48 KB dynamic opt-in = 96 KB fits.
    volta = KernelResources(
        64,
        32,
        static_shared_bytes=48 * KIB,
        dynamic_shared_bytes=48 * KIB,
        shared_memory_opt_in=True,
    )
    assert analyze_occupancy(VOLTA, volta).shared_memory.blocks == 1


def test_launches_that_cannot_fit_are_rejected_with_every_reason():
    # 1,024 threads * 128 registers overflow the register file and 64 KB of
    # dynamic shared memory without opt-in overflows the per-block limit.
    kernel = KernelResources(1024, 128, dynamic_shared_bytes=64 * KIB)
    with pytest.raises(LaunchNotResidentError) as error:
        analyze_occupancy(AMPERE, kernel)

    assert error.value.blockers == (
        "registers_per_block",
        "registers_per_block_partition_check",
        "shared_memory_per_block",
    )
    assert "cannot be resident on one SM" in str(error.value)
    with pytest.raises(SimtError, match="between 1 and 1024"):
        analyze_occupancy(TURING, KernelResources(2048, 16))
    with pytest.raises(SimtError, match="must be dynamic shared memory"):
        analyze_occupancy(AMPERE, KernelResources(64, 16, static_shared_bytes=49_153))


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: KernelResources(0, 32), SimtError, "threads_per_block"),
        (
            lambda: KernelResources(32, 32, dynamic_shared_bytes=-1),
            SimtError,
            "dynamic",
        ),
        (
            lambda: KernelResources(32, 32, shared_memory_opt_in=1),
            TypeError,
            "must be a boolean",
        ),
        (lambda: analyze_occupancy(AMPERE, (256, 32)), TypeError, "KernelResources"),
        (
            lambda: carveout_for_preference(AMPERE, 101),
            SimtError,
            "carveout percent must be between 0 and 100",
        ),
        (
            lambda: analyze_occupancy(
                AMPERE, KernelResources(32, 32), carveout_bytes=True
            ),
            SimtError,
            "supported shared memory capacity",
        ),
    ],
)
def test_occupancy_inputs_reject_ambiguous_values(call, error, message):
    with pytest.raises(error, match=message):
        call()
