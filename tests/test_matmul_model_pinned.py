from pathlib import Path

import pytest

from ai_journey.matmul_model_cli import summary_line
from ai_journey.matmul_model_plan import load_plan
from ai_journey.matmul_model_report import build_report, seal_report
from ai_journey.matmul_model_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-74-matmul-model.json"
GUIDE = ROOT / "docs/matmul-model-lab.md"
REPORT_SHA256 = "6fac533806b4746c46cca614bad20d9cd04eda07bb8a8d7eacd76951fc40ebb8"
RAGGED_SHA256 = "41d50935c365946c5bce24e34eeb75d2e1589ab7ff729920591204c344950cd8"
ALIGNED_SHA256 = "5d3baafab74b128d9c5039142a041e2e0c6b4d855e7706e703688c046867c35f"
CUDA_SHA256 = "a97031329a75820abe14a8cd5a3a55dbe728bbedd64871a79f8d5e2e7b815fc0"


@pytest.fixture(scope="module")
def report():
    plan = load_plan(CONFIG)
    sealed = seal_report(build_report(plan))
    assert verify_report(sealed, plan=plan) == plan
    return sealed


@pytest.fixture(scope="module")
def guide():
    return GUIDE.read_text(encoding="utf-8")


def mapping(report, case, name):
    (section,) = [
        item for item in report["cases"][case]["mappings"] if item["mapping"] == name
    ]
    return section


def test_pinned_runs_reproduce_the_documented_numbers(report):
    ragged, aligned = report["cases"]

    assert report["report_sha256"] == REPORT_SHA256
    assert (ragged["reference_sha256"], aligned["reference_sha256"]) == (
        RAGGED_SHA256,
        ALIGNED_SHA256,
    )
    assert report["cuda_source"]["sha256"] == CUDA_SHA256
    launches = [
        (
            section["launch"]["grid"],
            section["launch"]["launched_threads"],
            section["launch"]["idle_threads"],
        )
        for case in (ragged, aligned)
        for section in case["mappings"]
    ]
    assert launches == [
        ([2, 5, 1], 2560, 895),
        ([2, 6, 1], 3072, 1407),
        ([2, 4, 1], 2048, 0),
        ([1, 8, 1], 2048, 0),
    ]
    for case in (ragged, aligned):
        for section in case["mappings"]:
            assert [run["output_sha256"] for run in section["runs"]] == [
                case["reference_sha256"]
            ] * 4
    assert report["summary"] == {
        "cases": 2,
        "launches": 4,
        "ragged_launches": 2,
        "runs": 16,
        "runs_matching_reference": 16,
        "memory_safe_runs": 16,
        "race_free_runs": 16,
        "order_independent_launches": 4,
        "cross_checked_launches": 4,
        "cross_checked_requests": 13900,
        "probes": 8,
        "probe_traces_matching_model": 8,
        "accounting_shapes": 4,
        "roofline_points": 16,
        "memory_bound_points": 10,
        "occupancy_estimates": 4,
        "cuda_checks_passed": 16,
        "cuda_checks": 16,
    }


def test_pinned_access_totals_reproduce_the_documented_mappings(report):
    columns = mapping(report, 1, "x_columns")["access"]
    rows = mapping(report, 1, "x_rows")["access"]

    def per_request(access, name):
        operand = access["operands"][name]
        (kind,) = [key for key, count in operand["classes"].items() if count]
        return operand["requests"], kind, operand["sectors"] // operand["requests"]

    assert [per_request(columns, name) for name in "abc"] == [
        (1024, "broadcast", 1),
        (1024, "coalesced", 4),
        (64, "coalesced", 4),
    ]
    assert [per_request(rows, name) for name in "abc"] == [
        (1024, "strided", 32),
        (1024, "broadcast", 1),
        (64, "strided", 32),
    ]
    assert (columns["transferred_bytes"], rows["transferred_bytes"]) == (
        172_032,
        1_146_880,
    )
    assert 3 * rows["transferred_bytes"] == 20 * columns["transferred_bytes"]
    for access in (columns, rows):
        assert (
            sum(item["requested_bytes"] for item in access["operands"].values())
            == 270_336
        )
    step = mapping(report, 0, "x_columns")["probes"][0]["steps"][1]
    assert (step["a"]["lane_addresses"], step["a"]["sectors"]) == ([4] * 32, 1)
    assert step["b"]["lane_addresses"] == list(range(180, 308, 4))
    assert (step["b"]["sectors"], step["b"]["lines"]) == (5, 2)
    strided = mapping(report, 0, "x_rows")["probes"][0]["steps"][1]["a"]
    assert strided["lane_addresses"][:3] == [4, 120, 236]
    assert (strided["lane_stride_bytes"], strided["sectors"], strided["lines"]) == (
        116,
        32,
        29,
    )


def test_the_report_holds_no_floating_point_values(report):
    # Without floats, the canonical bytes cannot depend on float formatting.
    pending = [report]
    while pending:
        value = pending.pop()
        assert type(value) is not float
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def test_guide_tables_match_the_report(report, guide):
    lines = guide.splitlines()
    for case in report["cases"]:
        shape = case["shape"]
        rounding = case["order_rounding"]
        assert (
            f"| `{case['name']}`, {shape['m']} x {shape['n']} x {shape['k']} | "
            f"{rounding['elements']:,} | {rounding['differing']:,} | "
            f"{rounding['max_ulp_difference']} ulps |"
        ) in lines
        sector_row = " | ".join(
            f"{mapping(report, report['cases'].index(case), name)['access']['flops_per_transferred_byte']['decimal']} FLOP per byte"
            for name in ("x_columns", "x_rows")
        )
        assert f"| `{case['name']}` | {sector_row} |" in lines
    for item in report["accounting"]:
        shape = item["shape"]
        name = f"{shape['m']} x {shape['n']} x {shape['k']}"
        assert (
            f"| {name} | {item['flops']:,} | {item['no_reuse']['intensity']['decimal']} | "
            f"{item['ideal_reuse']['intensity']['decimal']} |"
        ) in lines
        fractions = " | ".join(
            point["no_reuse"]["fraction_of_compute_ceiling"]["decimal"]
            for point in item["roofline"]
        )
        assert f"| {name} | {fractions} |" in lines
    for estimate in report["occupancy"]["estimates"]:
        assert (
            f"| {estimate['kernel']['registers_per_thread']} | "
            f"{estimate['registers']['registers_per_warp']:,} | "
            f"{estimate['active_blocks_per_sm']} | {estimate['active_warps_per_sm']} | "
            f"{estimate['occupancy']['decimal']} | {', '.join(estimate['limiting_factors'])} |"
        ) in lines


def test_guide_states_commands_numbers_sources_and_boundaries(report, guide):
    text = " ".join(guide.split())

    assert summary_line("matmul_model_report", report) in guide.splitlines()
    for phrase in (
        "python -m ai_journey.matmul_model_cli run",
        "--config config/day-74-matmul-model.json",
        "python -m ai_journey.matmul_model_cli verify",
        "GPU MODE Lecture 4, `Intro to Compute and Memory Architecture`",
        '"Write a naive matmul kernel. Measure its GFLOPs. Note how far below peak it is."',
        "`1`: the command read or built a report, but the report failed verification.",
        "This half measures no GFLOP/s.",
        'In this lab, "peak" means a ceiling that a run measured on this host.',
        "This half measured no ceiling, so it states no peak.",
        "Double has 53 bits, and 53 is at least 2 * 24 + 2.",
        "When x runs over rows, the A loads and the C stores are `strided` requests, and the B loads are broadcasts.",
        "Both mappings request the same 270,336 bytes.",
        "With x over rows, the sectors move 20/3 times as many bytes.",
        "the lanes read bytes 180 to 304, coalesced, in 5 sectors and 2 lines.",
        "The lanes of that request are 116 bytes apart, in 32 sectors and 29 lines.",
        "All 13,900 requests of the pinned plan agree.",
        "the intensity is k / (4 * k + 2).",
        "These values are model inputs, not measurements, and they describe no device.",
        "It reaches less than 1 / (4 * ridge point) of the compute ceiling",
        "They reach 0.373215 and 0.285714 of the compute ceiling.",
        "Each count is an assumption.",
        "Thus shared memory alone admits 164 blocks, and it never sets the limit.",
        "They are not achieved occupancy, and they do not show speed.",
        "CUDA never merges these two intrinsics into a fused multiply-add.",
        "For each check, a mutation test shows that the check rejects its defect.",
        "the launch has 2,560 threads: 1,665 useful and 895 idle.",
        "the launch has 3,072 threads: 1,665 useful and 1,407 idle.",
        "Both mappings launch 2,048 threads and no idle thread.",
        "All 8 probe traces match the model, and all 16 CUDA checks pass.",
        "no NVIDIA GPU, no CUDA toolkit and no `nvcc`",
        "it gives no NVIDIA, CUDA, production or distributed evidence",
        "learner understanding",
        RAGGED_SHA256,
        ALIGNED_SHA256,
        CUDA_SHA256,
        REPORT_SHA256,
    ):
        assert phrase in text, phrase


def test_committed_config_and_guide_hold_no_host_specific_paths():
    for path in (CONFIG, GUIDE):
        text = path.read_text(encoding="utf-8")
        assert text.isascii()
        for marker in ("/Users/", "/home/", "/private/", "C:\\", "localhost"):
            assert marker not in text
