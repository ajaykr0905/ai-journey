"""CPU tests of the declared occupancy. No test compiles or runs CUDA."""

import pytest

from ai_journey import matmul_model_occupancy
from ai_journey.gpu_checklist_cuda import (
    PINNED_BLOCKED,
    PINNED_COARSENED,
    PINNED_KERNELS,
    BlockedKernel,
    CoarsenedKernel,
    GpuChecklistError,
)
from ai_journey.gpu_checklist_occupancy import (
    DECLARED_REGISTERS,
    REGISTER_SOURCE,
    declared_estimates,
    declared_registers,
    estimate_mapping,
    estimate_occupancy,
    kernel_resources,
    live_floats,
)
from ai_journey.kernel_launch_config import LAUNCH_LIMITS
from ai_journey.simt_occupancy import LaunchNotResidentError

T16_C2, T16_C4, T32_C2 = PINNED_COARSENED
M64, M128 = PINNED_BLOCKED


@pytest.mark.parametrize("kernel", PINNED_KERNELS, ids=lambda item: item.name)
def test_each_block_declares_its_threads_and_its_two_tiles(kernel):
    count = declared_registers(kernel)[0]
    resources = kernel_resources(kernel, registers_per_thread=count)

    assert resources.to_mapping() == {
        "threads_per_block": kernel.threads_per_block,
        "registers_per_thread": count,
        "static_shared_bytes": kernel.shared_memory_bytes,
        "dynamic_shared_bytes": 0,
        "shared_memory_opt_in": False,
    }


@pytest.mark.parametrize(
    ("kernel", "registers", "blocks", "warps", "limits", "ratio"),
    [
        (T16_C2, 32, 8, 64, ["warps", "registers"], (1, 1)),
        (T16_C2, 40, 6, 48, ["registers"], (3, 4)),
        (T16_C2, 64, 4, 32, ["registers"], (1, 2)),
        (T16_C4, 32, 8, 64, ["warps", "registers"], (1, 1)),
        (T16_C4, 40, 6, 48, ["registers"], (3, 4)),
        (T16_C4, 64, 4, 32, ["registers"], (1, 2)),
        (T32_C2, 32, 2, 64, ["warps", "registers"], (1, 1)),
        (T32_C2, 40, 1, 32, ["registers"], (1, 2)),
        (T32_C2, 64, 1, 32, ["registers"], (1, 2)),
        (M64, 48, 5, 40, ["registers"], (5, 8)),
        (M64, 64, 4, 32, ["registers"], (1, 2)),
        (M64, 96, 2, 16, ["registers"], (1, 4)),
        (M128, 96, 2, 16, ["registers"], (1, 4)),
        (M128, 128, 2, 16, ["registers"], (1, 4)),
        (M128, 168, 1, 8, ["registers"], (1, 8)),
    ],
    ids=lambda item: getattr(item, "name", str(item)),
)
def test_warp_slots_and_declared_registers_set_the_residency(
    kernel, registers, blocks, warps, limits, ratio
):
    mapping = estimate_mapping(kernel, registers_per_thread=registers)

    assert mapping["name"] == kernel.name
    assert mapping["register_source"] == REGISTER_SOURCE
    assert mapping["compute_capability"] == "8.0"
    assert mapping["warps_per_block"] == kernel.threads_per_block // 32
    assert mapping["active_blocks_per_sm"] == blocks
    assert mapping["active_warps_per_sm"] == warps
    assert mapping["limiting_factors"] == limits
    assert (
        mapping["occupancy"]["numerator"],
        mapping["occupancy"]["denominator"],
    ) == ratio


def test_the_declared_sweep_covers_each_pinned_kernel_and_stays_resident():
    estimates = declared_estimates()

    assert list(DECLARED_REGISTERS) == [kernel.name for kernel in PINNED_KERNELS]
    assert len(estimates) == 15
    for kernel in PINNED_KERNELS:
        counts = declared_registers(kernel)
        assert list(counts) == sorted(set(counts)) and len(counts) == 3
        assert counts[0] >= live_floats(kernel)
    assert all(item["active_blocks_per_sm"] >= 1 for item in estimates)
    assert "assumption" in REGISTER_SOURCE and "ptxas" in REGISTER_SOURCE


def test_the_floor_is_the_live_floats_of_one_thread():
    # The coarsened threads keep C sums and two operands; the blocked threads
    # keep TM * TN sums and TM + TN operands.
    assert [live_floats(kernel) for kernel in PINNED_KERNELS] == [4, 6, 4, 24, 80]
    with pytest.raises(GpuChecklistError, match="needs 80 to 255 registers"):
        kernel_resources(M128, registers_per_thread=64)
    with pytest.raises(GpuChecklistError, match="needs 24 to 255 registers"):
        kernel_resources(M64, registers_per_thread=256)
    with pytest.raises(TypeError, match="registers_per_thread must be an integer"):
        kernel_resources(M64, registers_per_thread=64.0)


def test_shared_memory_never_limits_these_kernels():
    # Each block adds the 1 KB that the driver reserves to its two tiles.
    allocated = {}
    for kernel in PINNED_KERNELS:
        shared = estimate_occupancy(
            kernel, registers_per_thread=declared_registers(kernel)[0]
        ).shared_memory
        allocated[kernel.name] = (shared.allocated_bytes_per_block, shared.blocks)

    assert allocated == {
        "coarsened_matmul_t16_c2": (3072, 54),
        "coarsened_matmul_t16_c4": (3072, 54),
        "coarsened_matmul_t32_c2": (9216, 18),
        "blocked_matmul_m64_n64_k16_t4x4": (9216, 18),
        "blocked_matmul_m128_n128_k8_t8x8": (9216, 18),
    }


def test_a_full_block_with_many_registers_is_not_resident():
    with pytest.raises(LaunchNotResidentError, match="registers_per_block"):
        estimate_occupancy(T32_C2, registers_per_thread=128)
    assert estimate_occupancy(M128, registers_per_thread=255).active_blocks_per_sm == 1


def test_estimates_refuse_other_kernels_and_disagreeing_limits(monkeypatch):
    with pytest.raises(GpuChecklistError, match="no declared register counts"):
        declared_registers(CoarsenedKernel(tile=8, coarsening=8))
    with pytest.raises(TypeError, match="CoarsenedKernel or BlockedKernel"):
        estimate_occupancy((16, 2), registers_per_thread=32)
    assert (
        estimate_occupancy(
            BlockedKernel(32, 32, 32, 2, 2), registers_per_thread=32
        ).active_blocks_per_sm
        == 8
    )
    monkeypatch.setattr(
        matmul_model_occupancy, "LAUNCH_LIMITS", {**LAUNCH_LIMITS, "warp_size": 64}
    )
    with pytest.raises(GpuChecklistError, match="disagree"):
        estimate_occupancy(T16_C2, registers_per_thread=32)
