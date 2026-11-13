import json
from pathlib import Path

from ai_journey.memory_access_plan import MAX_BENCHMARK_FLOPS
from ai_journey.memory_access_report import build_analysis_report, seal_report
from ai_journey.memory_access_report_io import load_plan
from ai_journey.memory_access_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/day-74-memory-access-plan.json"
DOC = ROOT / "docs/memory-access-model.md"


def pinned_report():
    plan = load_plan(PLAN)
    report = seal_report(build_analysis_report(plan))
    verify_report(report, plan=plan)
    return report


def test_pinned_patterns_reproduce_the_documented_coalescing_and_bank_results():
    patterns = {item["name"]: item for item in pinned_report()["patterns"]}

    def global_memory(name):
        values = patterns[name]["global_memory"]
        return values["sectors"], values["lines"], values["efficiency"]

    assert global_memory("contiguous-float32") == (4, 1, 1.0)
    assert global_memory("offset-4-bytes-float32") == (5, 2, 0.8)
    assert global_memory("stride-2-float32") == (8, 2, 0.5)
    assert global_memory("stride-8-float32") == (32, 8, 0.125)
    assert global_memory("broadcast-float32") == (1, 1, 0.125)
    assert global_memory("permutation-float32") == (4, 1, 1.0)
    assert global_memory("half-warp-float32") == (2, 1, 1.0)
    assert global_memory("row-walk-unpadded-float32")[0] == 5
    assert global_memory("row-walk-padded-float32")[0] == 4
    assert global_memory("contiguous-float4-vector") == (16, 4, 1.0)
    degrees = {
        name: item["shared_memory"]["conflict_degree"]
        for name, item in patterns.items()
        if item["shared_memory"] is not None
    }
    assert degrees["stride-2-float32"] == 2
    assert degrees["stride-8-float32"] == 8
    assert degrees["stride-32-float32"] == 32
    assert degrees["stride-33-float32"] == 1
    assert patterns["contiguous-float64"]["shared_memory"] is None


def test_pinned_transposes_show_padding_and_store_coalescing():
    transposes = {item["name"]: item for item in pinned_report()["transposes"]}

    def column_wavefronts(name):
        return transposes[name]["shared_column_reads_per_tile"]["wavefronts"]

    assert column_wavefronts("float32-tile-unpadded") == 1024
    assert column_wavefronts("float32-tile-padded") == 32
    assert column_wavefronts("float16-tile-one-element-pad") == 48
    assert column_wavefronts("float16-tile-one-word-pad") == 32
    padded = transposes["float32-tile-padded"]
    assert padded["naive_global_stores"]["sectors_per_request"] == 32
    assert padded["naive_global_stores"]["transferred_bytes"] == 536_870_912
    assert padded["tiled_global_stores"]["sectors_per_request"] == 4
    assert padded["tiled_global_stores"]["transferred_bytes"] == 67_108_864


def test_pinned_tile_sweep_stays_memory_bound_below_the_declared_ridge():
    matmul = pinned_report()["matmul"]
    sweeps = {
        (item["shape"]["m"], item["shape"]["n"], item["shape"]["k"]): item
        for item in matmul["sweeps"]
    }
    square = sweeps[(1024, 1024, 1024)]
    ragged = sweeps[(1000, 1000, 1000)]

    assert matmul["roofline"]["ridge_point_flops_per_byte"] == 19.5e12 / 2.039e12
    assert round(matmul["roofline"]["ridge_point_flops_per_byte"], 4) == 9.5635
    assert square["compulsory"]["bottleneck"] == "compute"
    assert round(square["compulsory"]["arithmetic_intensity"], 2) == 170.67
    assert round(square["tiled"][2]["roofline"]["arithmetic_intensity"], 3) == 7.877
    assert all(
        tile["roofline"]["bottleneck"] == "memory"
        for sweep in matmul["sweeps"]
        for tile in sweep["tiled"]
    )
    assert [tile["traffic"]["padded_slots"] for tile in ragged["tiled"]] == [
        0,
        2_024_064,
        3_108_864,
    ]
    degrees = {
        item["tile"]: item["layouts"]["b_column_major"]["all_steps"][
            "max_conflict_degree"
        ]
        for item in matmul["inner_loop_banks"]
    }
    assert degrees == {8: 2, 16: 8, 32: 32}


def test_pinned_benchmark_is_small_and_the_roofline_is_cited_as_declared():
    plan = load_plan(PLAN)
    raw = json.loads(PLAN.read_text(encoding="utf-8"))
    cases = plan.benchmark.cases()
    work = sum(case.shape.flops for case in cases) * (
        plan.benchmark.warmup + plan.benchmark.repeats
    )

    assert [case.backend for case in cases] == ["numpy"] * 3 + ["torch"] * 3
    assert work < MAX_BENCHMARK_FLOPS // 10
    assert raw["roofline"]["declared"] is True
    assert (
        raw["roofline"]["source_url"]
        == "https://www.nvidia.com/en-us/data-center/a100/"
    )
    assert "not measured" in raw["roofline"]["source_note"]


def test_runbook_lists_commands_sources_and_exclusions():
    documentation = DOC.read_text(encoding="utf-8")
    compact = " ".join(documentation.split())

    for command in ("analyze", "benchmark", "verify"):
        assert f"ai-journey-memory-access {command}" in documentation
    assert "config/day-74-memory-access-plan.json" in documentation
    for source in (
        "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
        "https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html",
        "https://www.nvidia.com/en-us/data-center/a100/",
    ):
        assert source in documentation
    for exclusion in (
        "L1 and L2 caching",
        "hardware prefetching",
        "TLB",
        "64-bit shared-memory bank modes",
        "asynchronous global-to-shared copies",
        "tensor cores",
    ):
        assert exclusion in compact
    assert "no number in the analytical report is a measurement" in compact
    assert "wall-clock CPU measurement on the host that produced the report" in compact
