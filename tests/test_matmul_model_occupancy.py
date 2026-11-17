import pytest

from ai_journey import matmul_model_occupancy
from ai_journey.kernel_launch_config import LAUNCH_LIMITS
from ai_journey.matmul_model_occupancy import (
    ARCHITECTURE,
    architecture_mapping,
    estimate_occupancy,
    launch_limits_agree,
    naive_matmul_resources,
    occupancy_mapping,
)
from ai_journey.matmul_model_reference import MatmulModelError
from ai_journey.simt_occupancy import LaunchNotResidentError


def test_the_simt_record_repeats_every_day_73_launch_limit():
    assert launch_limits_agree()
    assert ARCHITECTURE.compute_capability == "8.0"
    for name, value in LAUNCH_LIMITS.items():
        assert getattr(ARCHITECTURE, name) == value
    mapping = architecture_mapping()
    assert mapping["launch_limits_agree"] is True
    assert (mapping["warp_size"], mapping["max_threads_per_block"]) == (32, 1024)
    assert (mapping["max_warps_per_sm"], mapping["max_blocks_per_sm"]) == (64, 32)
    assert mapping["reserved_shared_memory_per_block_bytes"] == 1024
    assert any("Table 21" in source for source in mapping["sources"])


@pytest.mark.parametrize(
    ("registers", "blocks", "warps", "limits", "ratio"),
    [
        (32, 8, 64, ["warps", "registers"], (1, 1)),
        (40, 6, 48, ["registers"], (3, 4)),
        (64, 4, 32, ["registers"], (1, 2)),
        (128, 2, 16, ["registers"], (1, 4)),
        (255, 1, 8, ["registers"], (1, 8)),
    ],
)
def test_registers_set_the_residency_of_a_256_thread_block(
    registers, blocks, warps, limits, ratio
):
    mapping = occupancy_mapping(
        estimate_occupancy(threads_per_block=256, registers_per_thread=registers)
    )

    assert mapping["warps_per_block"] == 8
    assert mapping["active_blocks_per_sm"] == blocks
    assert mapping["active_warps_per_sm"] == warps
    assert mapping["limiting_factors"] == limits
    assert (
        mapping["occupancy"]["numerator"],
        mapping["occupancy"]["denominator"],
    ) == ratio
    # Only the 1 KB that the driver reserves per block counts against shared
    # memory, so 164 blocks would fit and shared memory never limits.
    assert mapping["shared_memory"]["allocated_bytes_per_block"] == 1024
    assert mapping["limits_blocks_per_sm"]["shared_memory"] == 164
    assert mapping["kernel"]["static_shared_bytes"] == 0
    assert mapping["kernel"]["dynamic_shared_bytes"] == 0


def test_large_blocks_fit_until_one_block_needs_every_register():
    full = estimate_occupancy(threads_per_block=1024, registers_per_thread=32)
    assert (full.active_blocks_per_sm, full.active_warps_per_sm) == (2, 64)

    with pytest.raises(LaunchNotResidentError, match="registers_per_block"):
        estimate_occupancy(threads_per_block=1024, registers_per_thread=128)
    with pytest.raises(TypeError, match="registers_per_thread"):
        naive_matmul_resources(threads_per_block=256, registers_per_thread=True)


def test_estimates_refuse_to_run_when_the_limits_disagree(monkeypatch):
    monkeypatch.setattr(
        matmul_model_occupancy, "LAUNCH_LIMITS", {**LAUNCH_LIMITS, "warp_size": 64}
    )

    assert not launch_limits_agree()
    with pytest.raises(MatmulModelError, match="disagree"):
        estimate_occupancy(threads_per_block=256, registers_per_thread=32)
    with pytest.raises(TypeError, match="OccupancyResult"):
        occupancy_mapping({"occupancy": 1.0})
