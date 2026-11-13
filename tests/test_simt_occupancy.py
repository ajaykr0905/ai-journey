import pytest

from ai_journey.simt_device import PINNED_ARCHITECTURES, SimtError
from ai_journey.simt_occupancy import (
    block_slot_limit,
    register_limit,
    warp_slot_limit,
    warps_per_block,
)

VOLTA = PINNED_ARCHITECTURES["7.0"]
TURING = PINNED_ARCHITECTURES["7.5"]
AMPERE = PINNED_ARCHITECTURES["8.0"]


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
