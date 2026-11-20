import pytest

from ai_journey import matmul_model_occupancy
from ai_journey.gpu_tiled_cuda import TILES, GpuTiledError
from ai_journey.gpu_tiled_occupancy import estimate_occupancy, tiled_matmul_resources
from ai_journey.kernel_launch_config import LAUNCH_LIMITS
from ai_journey.matmul_model_occupancy import occupancy_mapping
from ai_journey.simt_occupancy import LaunchNotResidentError


@pytest.mark.parametrize("tile", TILES)
def test_each_block_declares_its_threads_and_two_float_tiles(tile):
    resources = tiled_matmul_resources(tile=tile, registers_per_thread=32)

    assert resources.to_mapping() == {
        "threads_per_block": tile * tile,
        "registers_per_thread": 32,
        "static_shared_bytes": 2 * tile * tile * 4,
        "dynamic_shared_bytes": 0,
        "shared_memory_opt_in": False,
    }


@pytest.mark.parametrize(
    ("tile", "registers", "blocks", "warps", "limits", "ratio"),
    [
        (8, 32, 32, 64, ["warps", "registers", "blocks"], (1, 1)),
        (8, 40, 24, 48, ["registers"], (3, 4)),
        (8, 64, 16, 32, ["registers"], (1, 2)),
        (16, 32, 8, 64, ["warps", "registers"], (1, 1)),
        (16, 40, 6, 48, ["registers"], (3, 4)),
        (16, 64, 4, 32, ["registers"], (1, 2)),
        (32, 32, 2, 64, ["warps", "registers"], (1, 1)),
        (32, 40, 1, 32, ["registers"], (1, 2)),
        (32, 64, 1, 32, ["registers"], (1, 2)),
    ],
)
def test_warp_slots_and_declared_registers_set_the_residency(
    tile, registers, blocks, warps, limits, ratio
):
    mapping = occupancy_mapping(
        estimate_occupancy(tile=tile, registers_per_thread=registers)
    )

    assert mapping["compute_capability"] == "8.0"
    assert mapping["warps_per_block"] == tile * tile // 32
    assert mapping["active_blocks_per_sm"] == blocks
    assert mapping["active_warps_per_sm"] == warps
    assert mapping["limiting_factors"] == limits
    assert (
        mapping["occupancy"]["numerator"],
        mapping["occupancy"]["denominator"],
    ) == ratio


def test_shared_memory_never_limits_these_tiles():
    # Each block adds the 1 KB that the driver reserves to its two tiles.
    allocated = {}
    for tile in TILES:
        shared = estimate_occupancy(tile=tile, registers_per_thread=32).shared_memory
        allocated[tile] = (shared.allocated_bytes_per_block, shared.blocks)

    assert allocated == {8: (1536, 109), 16: (3072, 54), 32: (9216, 18)}


def test_a_full_block_with_many_registers_is_not_resident():
    with pytest.raises(LaunchNotResidentError, match="registers_per_block"):
        estimate_occupancy(tile=32, registers_per_thread=128)
    assert (
        estimate_occupancy(tile=16, registers_per_thread=128).active_blocks_per_sm == 2
    )


def test_estimates_refuse_bad_inputs_and_disagreeing_limits(monkeypatch):
    with pytest.raises(GpuTiledError, match="tile must be one of"):
        estimate_occupancy(tile=4, registers_per_thread=32)
    with pytest.raises(TypeError, match="registers_per_thread"):
        tiled_matmul_resources(tile=8, registers_per_thread=True)
    monkeypatch.setattr(
        matmul_model_occupancy, "LAUNCH_LIMITS", {**LAUNCH_LIMITS, "warp_size": 64}
    )
    with pytest.raises(GpuTiledError, match="disagree"):
        estimate_occupancy(tile=16, registers_per_thread=32)
