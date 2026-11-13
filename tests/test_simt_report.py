from copy import deepcopy

import pytest

from ai_journey.simt_device import SimtError
from ai_journey.simt_json import canonical_json_bytes
from ai_journey.simt_plan import SimtPlan
from ai_journey.simt_report import (
    SIMT_EVIDENCE_BOUNDARY,
    SIMT_EXCLUSIONS,
    SIMT_REPORT_KIND,
    build_simt_report,
    seal_simt_report,
    simt_report_digest,
)
from test_simt_plan import plan_mapping


def report():
    return build_simt_report(SimtPlan.from_mapping(plan_mapping()))


def test_report_embeds_declared_inputs_and_an_unweakened_boundary():
    built = report()

    assert built["schema_version"] == 1
    assert built["kind"] == SIMT_REPORT_KIND
    assert built["measured"] is False
    assert built["declared_inputs"] == plan_mapping()
    assert built["evidence_boundary"] == dict(SIMT_EVIDENCE_BOUNDARY)
    assert built["evidence_boundary"]["gpu_executed"] is False
    assert built["exclusions"] == list(SIMT_EXCLUSIONS)
    assert any("Partitioned global caching" in item for item in built["exclusions"])
    assert built["devices"]["tesla-t4"]["fp32_datasheet_check"][
        "within_published_rounding"
    ]
    assert len(canonical_json_bytes(built)) < 64 * 1024


def test_vector_add_section_reconciles_launch_coverage_occupancy_and_waves():
    # 5,000 elements in 256-thread blocks: ceil(5000 / 256) = 20 blocks and
    # 5,120 threads; 120 fail the guard. Warp 156 holds threads 4992..5023 (8
    # pass), warps 157..159 are fully masked, and warps 0..155 are active.
    section = report()["vector_add"]
    t4 = section["devices"]["tesla-t4"]

    assert section["workload"]["arithmetic_intensity"] == 1 / 12
    assert t4["launch"]["grid"] == [20, 1, 1]
    assert t4["coverage"]["masked_threads"] == 120
    assert t4["coverage"]["unguarded_out_of_bounds_writes"] == 120
    assert (
        t4["coverage"]["fully_active_warps"],
        t4["coverage"]["partially_masked_warps"],
        t4["coverage"]["fully_masked_warps"],
    ) == (156, 1, 3)
    # T4: 32 warp slots / 8 warps per block = 4 blocks -> 100% occupancy;
    # 40 SMs * 4 = 160 slots hold all 20 blocks in one wave (12.5% of slots).
    assert t4["occupancy"]["active_blocks_per_sm"] == 4
    assert t4["occupancy"]["occupancy"] == 1.0
    assert t4["waves"]["waves"] == 1
    assert t4["waves"]["sm_slot_utilization"] == 0.125
    assert t4["roofline"]["bottleneck"] == "memory"
    a100 = section["devices"]["a100-sxm4-80gb"]
    assert a100["occupancy"]["active_blocks_per_sm"] == 8
    assert a100["occupancy"]["limiting_factors"] == ["warps"]


def test_grayscale_section_flags_the_floor_division_counterexample():
    # 40 x 21 pixels with 16 x 16 blocks: ceil grid (3, 2) launches 48 x 32 =
    # 1,536 threads, 696 of them masked; the floor grid (2, 1) covers 32 x 16.
    a100 = report()["rgb_to_grayscale"]["devices"]["a100-sxm4-80gb"]

    assert a100["launch"]["grid"] == [3, 2, 1]
    assert a100["coverage"]["covered_elements"] == 840
    assert a100["coverage"]["masked_threads"] == 696
    assert a100["coverage"]["under_covered"] is False
    assert a100["floor_division_launch"]["grid"] == [2, 1, 1]
    assert a100["floor_division_coverage"]["uncovered_elements"] == 840 - 512
    assert a100["floor_division_coverage"]["under_covered"] is True


def test_sweep_latency_and_wave_sections_follow_the_declared_rows():
    built = report()
    sweep = built["occupancy_sweep"]

    # 128 threads at 128 registers: 4,096 per warp -> 16 warps -> 4 blocks;
    # T4 has 32 warp slots (50%), the A100 64 (25%).
    t4_cell = sweep["tesla-t4"][1]
    assert (t4_cell["threads_per_block"], t4_cell["registers_per_thread"]) == (128, 128)
    assert (t4_cell["active_blocks_per_sm"], t4_cell["occupancy"]) == (4, 0.5)
    assert sweep["a100-sxm4-80gb"][1]["occupancy"] == 0.25
    # 1,024 threads at 128 registers need 131,072 registers: never resident.
    assert sweep["tesla-t4"][3] == {
        "threads_per_block": 1024,
        "registers_per_thread": 128,
        "resident": False,
        "blockers": ["registers_per_block", "registers_per_block_partition_check"],
    }
    memory = built["latency_hiding"]["memory"]
    assert [row["warps_needed"] for row in memory] == [42, 84]
    assert [row["resident_warps"] for row in memory] == [64, 64]
    arithmetic = built["latency_hiding"]["arithmetic"]
    assert [(row["pipeline"], row["warps_needed"]) for row in arithmetic] == [
        ("issue", 16),
        ("issue", 8),
        ("fp32", 8),
        ("fp32", 4),
    ]
    assert built["wave_quantization"]["tesla-t4"]["waves"]["waves"] == 7
    assert built["wave_quantization"]["a100-sxm4-80gb"]["waves"]["waves"] == 3
    assert built["summary"] == {
        "devices": 2,
        "occupancy_sweep_cells": 8,
        "non_resident_sweep_cells": 2,
        "memory_latency_rows": 2,
        "memory_rows_beyond_warp_limit": 1,
        "arithmetic_latency_rows": 4,
    }


def test_seal_is_canonical_stable_and_sensitive_to_every_value():
    built = report()
    first = seal_simt_report(built)
    second = seal_simt_report(deepcopy(built))

    assert first == second
    assert "integrity" not in built
    assert first["integrity"] == {
        "algorithm": "sha256",
        "digest": simt_report_digest(first),
    }
    assert seal_simt_report(first) == first
    changed = deepcopy(built)
    changed["vector_add"]["devices"]["tesla-t4"]["waves"]["waves"] += 1
    assert seal_simt_report(changed)["integrity"] != first["integrity"]


def test_report_rejects_declared_kernels_that_cannot_be_resident():
    mapping = plan_mapping()
    mapping["vector_add"].update(threads_per_block=1024, registers_per_thread=255)

    with pytest.raises(SimtError, match="not resident on tesla-t4"):
        build_simt_report(SimtPlan.from_mapping(mapping))
    with pytest.raises(TypeError, match="SimtPlan"):
        build_simt_report(plan_mapping())
    with pytest.raises(TypeError, match="mapping"):
        seal_simt_report([("schema_version", 1)])
