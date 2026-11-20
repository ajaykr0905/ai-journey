import json
from pathlib import Path

import pytest

from ai_journey.gpu_tiled_config import (
    MAX_SIMULATED_STEPS,
    PINNED_PLAN,
    GpuTiledPlan,
    load_plan,
    simulated_steps,
)
from ai_journey.gpu_tiled_cuda import TILES, GpuTiledError
from ai_journey.kernel_launch_config import KernelLaunchError
from ai_journey.matmul_metal_config import load_config
from ai_journey.simt_occupancy import LaunchNotResidentError

ROOT = Path(__file__).resolve().parents[1]
DAY_74_METAL = ROOT / "config/day-74-matmul-metal.json"


def small_mapping(**changes):
    mapping = {
        "schema_version": 1,
        "name": "small-tiled",
        "statement": "A small tiled plan for tests.",
        "compute_capability": "8.0",
        "seed": 74,
        "tiles": [8, 16],
        "shapes": [[1, 1, 1], [17, 33, 9], [37, 45, 29]],
        "registers_per_thread": [32, 64],
    }
    return {**mapping, **changes}


def write_plan(path, **changes):
    path.write_text(json.dumps(small_mapping(**changes)), encoding="utf-8")
    return path


def test_the_pinned_plan_reuses_the_day_74_correctness_cases():
    day_74 = load_config(DAY_74_METAL)

    assert PINNED_PLAN.seed == day_74.seed == 74
    assert PINNED_PLAN.shapes == day_74.correctness.shapes
    assert PINNED_PLAN.tiles == TILES
    assert PINNED_PLAN.registers_per_thread == (32, 40, 64)
    assert PINNED_PLAN.name == "day-76-gpu-tiled-cuda"
    assert "No GPU runs" in PINNED_PLAN.statement
    assert simulated_steps(PINNED_PLAN.tiles, PINNED_PLAN.shapes) == 931_565_672
    assert GpuTiledPlan.from_mapping(PINNED_PLAN.to_mapping()) == PINNED_PLAN


def test_a_plan_file_round_trips_through_its_mapping(tmp_path):
    plan = load_plan(write_plan(tmp_path / "plan.json"))

    assert plan.to_mapping() == small_mapping()
    assert plan.shapes[2].to_mapping() == {"m": 37, "n": 45, "k": 29}
    assert GpuTiledPlan.from_mapping(plan.to_mapping()) == plan


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"schema_version": 2}, GpuTiledError, "schema_version"),
        ({"name": "Small Plan"}, GpuTiledError, "slug"),
        ({"statement": " padded "}, GpuTiledError, "printable ASCII"),
        ({"compute_capability": "9.0"}, GpuTiledError, "'8.0'"),
        ({"seed": True}, TypeError, "seed"),
        ({"seed": -1}, ValueError, "seed"),
        ({"tiles": []}, GpuTiledError, "1 to 3"),
        ({"tiles": [16, 8]}, GpuTiledError, "increase"),
        ({"tiles": [8, 8]}, GpuTiledError, "increase"),
        ({"tiles": [12]}, GpuTiledError, "tile must be one of"),
        ({"tiles": [8.0]}, TypeError, "integers"),
        ({"shapes": []}, GpuTiledError, "1 to 16"),
        ({"shapes": [[1, 1]]}, GpuTiledError, "triple"),
        ({"shapes": [[1, 1, 1], [1, 1, 1]]}, GpuTiledError, "repeat"),
        ({"shapes": [[8193, 1, 1]]}, GpuTiledError, "at most 8192"),
        ({"shapes": [[1, True, 1]]}, TypeError, "integer"),
        ({"registers_per_thread": [64, 32]}, GpuTiledError, "increase"),
        ({"registers_per_thread": [0]}, ValueError, "registers_per_thread"),
        ({"registers_per_thread": [256]}, ValueError, "registers_per_thread"),
    ],
)
def test_invalid_fields_are_refused(changes, error, message):
    with pytest.raises(error, match=message):
        GpuTiledPlan.from_mapping(small_mapping(**changes))


def test_missing_extra_and_duplicate_fields_are_refused(tmp_path):
    mapping = small_mapping()
    del mapping["seed"]
    with pytest.raises(KernelLaunchError, match=r"missing=\['seed'\]"):
        GpuTiledPlan.from_mapping(mapping)
    with pytest.raises(KernelLaunchError, match=r"extra=\['block'\]"):
        GpuTiledPlan.from_mapping(small_mapping(block=[16, 16]))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"name":"a","name":"b"}')
    with pytest.raises(KernelLaunchError, match="duplicate JSON key"):
        load_plan(duplicate)
    with pytest.raises(GpuTiledError, match="invalid plan"):
        load_plan(write_plan(tmp_path / "typed.json", tiles=[True]))


def test_each_register_count_must_leave_a_block_resident_at_each_tile():
    with pytest.raises(LaunchNotResidentError, match="registers_per_block"):
        GpuTiledPlan.from_mapping(
            small_mapping(tiles=[8, 32], registers_per_thread=[32, 128])
        )
    plan = GpuTiledPlan.from_mapping(small_mapping(registers_per_thread=[32, 128]))
    assert plan.registers_per_thread == (32, 128)


def test_the_simulated_work_of_a_plan_is_bounded():
    shapes = [[8192, 8192, 16]]
    steps = simulated_steps(TILES, (PINNED_PLAN.shapes[0],))

    assert steps == (64 + 256 + 1024) * 1 + 2
    with pytest.raises(GpuTiledError, match=f"at most {MAX_SIMULATED_STEPS}"):
        GpuTiledPlan.from_mapping(small_mapping(tiles=[8, 16, 32], shapes=shapes))
