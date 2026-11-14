from pathlib import Path

from ai_journey.simt_device import PINNED_ARCHITECTURES, PINNED_DEVICES
from ai_journey.simt_plan import load_simt_plan
from ai_journey.simt_report import build_simt_report, seal_simt_report
from ai_journey.simt_verify import verify_simt_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-71-simt-execution-plan.json"
GUIDE = ROOT / "docs/simt-execution-model.md"


def pinned_report():
    return seal_simt_report(build_simt_report(load_simt_plan(CONFIG)))


def test_pinned_plan_declares_every_pinned_device_with_labeled_inputs():
    plan = load_simt_plan(CONFIG)

    assert plan.name == "day-71-simt-execution-plan"
    assert plan.architectures == tuple(PINNED_ARCHITECTURES.values())
    assert plan.devices == tuple(PINNED_DEVICES.values())
    assert plan.vector_add.elements == 2**24
    assert plan.vector_add.threads_per_block == 256
    assert (plan.rgb_to_grayscale.width, plan.rgb_to_grayscale.height) == (1920, 1080)
    assert (plan.rgb_to_grayscale.block_x, plan.rgb_to_grayscale.block_y) == (16, 16)
    assert plan.occupancy_sweep.devices == ("a100-sxm4-80gb", "tesla-t4")
    assert plan.arithmetic_latency.architectures == ("7.0", "7.5")
    assert "hundreds of clock cycles" in plan.memory_latency.basis
    assert "not measured" in plan.memory_latency.basis
    assert "not a compiler result" in plan.vector_add.basis
    assert "No GPU, CUDA kernel, or latency was executed or measured" in (
        plan.statement
    )


def test_pinned_report_reproduces_the_documented_headline_numbers():
    report = pinned_report()
    assert verify_simt_report(report).name == "day-71-simt-execution-plan"
    assert report["summary"] == {
        "devices": 3,
        "occupancy_sweep_cells": 40,
        "non_resident_sweep_cells": 6,
        "memory_latency_rows": 18,
        "memory_rows_beyond_warp_limit": 1,
        "arithmetic_latency_rows": 12,
    }
    vector = report["vector_add"]["devices"]
    assert {key: row["waves"]["waves"] for key, row in vector.items()} == {
        "tesla-v100-sxm2": 103,
        "tesla-t4": 410,
        "a100-sxm4-80gb": 76,
    }
    assert all(row["occupancy"]["occupancy"] == 1.0 for row in vector.values())
    assert all(row["coverage"]["masked_threads"] == 0 for row in vector.values())
    image = report["rgb_to_grayscale"]["devices"]["a100-sxm4-80gb"]
    assert image["launch"]["grid"] == [120, 68, 1]
    assert image["coverage"]["fully_masked_warps"] == 480
    assert image["coverage"]["partially_masked_warps"] == 0
    assert image["floor_division_coverage"]["uncovered_elements"] == 15_360
    a100_256 = [
        cell["occupancy"]
        for cell in report["occupancy_sweep"]["a100-sxm4-80gb"]
        if cell["threads_per_block"] == 256
    ]
    assert a100_256 == [1.0, 0.5, 0.25, 0.125]
    beyond = [
        row
        for row in report["latency_hiding"]["memory"]
        if not row["full_bandwidth_within_warp_limit"]
    ]
    assert [
        (row["device"], row["latency_cycles"], row["pattern"]) for row in beyond
    ] == [("a100-sxm4-80gb", 800, "fp32-scalar")]
    assert beyond[0]["warps_needed"] == 84
    assert round(beyond[0]["sustainable_bandwidth_fraction"], 3) == 0.765
    waves = report["wave_quantization"]
    assert {
        key: (row["waves"]["waves"], round(row["waves"]["sm_slot_utilization"], 3))
        for key, row in waves.items()
    } == {
        "tesla-v100-sxm2": (4, 0.8),
        "tesla-t4": (7, 0.914),
        "a100-sxm4-80gb": (3, 0.79),
    }


def test_guide_documents_commands_sources_numbers_and_boundaries():
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())

    for phrase in (
        "ai-journey-simt build",
        "--config config/day-71-simt-execution-plan.json",
        "ai-journey-simt verify",
        "python -m ai_journey.simt_cli",
        "measured=false gpu_executed=false",
        "Table 21",
        "cuda_occupancy.h",
        "WP-08608-001_v1.1",
        "WP-09183-001_v01",
        "datasheet (JUN21)",
        "300 GB/sec",
        "320 GB/sec",
        "lists its boost clock as not finalized",
        "typically hundreds of clock cycles",
        "103 waves",
        "410 on the T4",
        "76 on the A100",
        "15,360 pixels uncovered",
        "480 warps",
        "Six of the 40 cells",
        "84 warps",
        "76.5%",
        "79.0% SM-slot utilization",
        "partitioned global caching",
        "carveout selection",
        "no GPU, CUDA kernel, or latency was executed or measured",
        "learner understanding",
    ):
        assert phrase in guide, phrase


def test_committed_plan_and_guide_hold_no_host_specific_paths():
    for path in (CONFIG, GUIDE):
        text = path.read_text(encoding="utf-8")
        assert text.isascii()
        for marker in ("/Users/", "/home/", "/private/", "C:\\", "localhost"):
            assert marker not in text
