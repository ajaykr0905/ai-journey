from pathlib import Path

import pytest

from ai_journey.kernel_launch_cli import summary_line
from ai_journey.kernel_launch_plan import load_plan
from ai_journey.kernel_launch_report import build_report, seal_report
from ai_journey.kernel_launch_verify import verify_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/day-73-kernel-launch.json"
GUIDE = ROOT / "docs/kernel-launch-lab.md"
REPORT_SHA256 = "4dfd9f8c599ae45e5033e56dddbd33f82bd02827bc9fbd41574c167995ae8af0"
VECTOR_SHA256 = "879aaad6e7345f9a1f64a662323d767deb8b3322d70bc3df550b19d0e47bf870"
GRAY_SHA256 = "681b798d88c39d1160f671a756c5a9df66f84f1bb1e6ea2763a70d703e735726"
CUDA_SHA256 = "cf14adedc20211d11c978cf0d6af291a447853ec01e4d0dee1ecfd02c8631bcc"


@pytest.fixture(scope="module")
def report():
    plan = load_plan(CONFIG)
    sealed = seal_report(build_report(plan))
    assert verify_report(sealed, plan=plan) == plan
    return sealed


def test_pinned_guarded_runs_reproduce_the_documented_numbers(report):
    vector = report["vector_add"]
    gray = report["rgb_to_grayscale"]

    assert report["report_sha256"] == REPORT_SHA256
    assert vector["launch"] == {
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
        "threads_per_block": 256,
        "blocks": 4,
        "launched_threads": 1024,
        "warps_per_block": 8,
        "idle_lanes_per_block": 0,
        "useful_threads": 1000,
        "idle_threads": 24,
    }
    assert vector["rounded_sums"] == 851
    assert gray["launch"]["grid"] == [3, 2, 1]
    assert (gray["launch"]["useful_threads"], gray["launch"]["idle_threads"]) == (
        1200,
        336,
    )
    assert gray["rounding"] == {"ties": 0, "truncation_differs": 612}
    for section, digest in ((vector, VECTOR_SHA256), (gray, GRAY_SHA256)):
        assert section["reference_sha256"] == digest
        assert [run["output_sha256"] for run in section["runs"]] == [digest] * 4
        assert section["order_independent"] is True
    assert report["cuda_source"]["sha256"] == CUDA_SHA256


def test_pinned_unguarded_demos_reproduce_the_documented_misuse(report):
    counts = [
        (
            demo["kernel"],
            demo["extent"],
            demo["out_of_bounds_writes"],
            demo["out_of_bounds_reads"],
            demo["write_conflicts"],
        )
        for demo in report["unguarded_demos"]
    ]

    assert counts == [
        ("vector_add_unguarded", [1000, 1, 1], 24, 48, 0),
        ("vector_add_unguarded", [1024, 1, 1], 0, 0, 0),
        ("rgb_to_grayscale_unguarded", [40, 30, 1], 104, 312, 232),
        ("rgb_to_grayscale_unguarded", [48, 32, 1], 0, 0, 0),
    ]
    first = [
        demo["first_out_of_bounds_writes"][:1] for demo in report["unguarded_demos"]
    ]
    assert first[0] == [
        {"buffer": "out", "access": "write", "index": 1000, "block": [3, 0, 0], "thread": [232, 0, 0]}
    ]  # fmt: skip
    assert first[2] == [
        {"buffer": "gray", "access": "write", "index": 1200, "block": [0, 1, 0], "thread": [0, 14, 0]}
    ]  # fmt: skip
    assert all(demo["orders_agree"] for demo in report["unguarded_demos"])
    assert report["summary"] == {
        "kernels": 2,
        "runs": 8,
        "runs_matching_reference": 8,
        "race_free_runs": 8,
        "order_independent_kernels": 2,
        "unguarded_demos": 4,
        "ragged_demos_with_out_of_bounds_writes": 2,
        "ragged_demos": 2,
        "aligned_demos_without_misuse": 2,
        "aligned_demos": 2,
        "out_of_bounds_writes_caught": 128,
        "write_conflicts_caught": 232,
        "cuda_checks_passed": 10,
        "cuda_checks": 10,
    }


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


def test_guide_states_commands_numbers_sources_and_boundaries(report):
    text = GUIDE.read_text(encoding="utf-8")
    guide = " ".join(text.split())

    assert summary_line("kernel_launch_report", report) in text.splitlines()
    for phrase in (
        "python -m ai_journey.kernel_launch_cli run",
        "--config config/day-73-kernel-launch.json",
        "python -m ai_journey.kernel_launch_cli verify",
        "`Getting Started With CUDA for Python Programmers`",
        "does not install a console script",
        "`1`: the command read or built a report, but the report failed verification.",
        "Table 21",
        "NVIDIA CUDA C++ Programming Guide v12.6 (archive 12.6.3)",
        "block.z = 65 breaks max_block_dim_z = 64 for compute capability 8.0",
        "0.2989, 0.5870 and 0.1140",
        "Rec. ITU-R BT.601",
        "Round half up: add 5,000, then divide by 10,000 with floor division.",
        "Clamp the result to [0, 255].",
        "1,703 have an exact sum that ends in .5",
        "For 292 of these ties",
        "1,024 threads: 1,000 useful and 24 idle",
        "1,536 threads: 1,200 useful and 336 idle",
        "851 of the 1,000 sums",
        "612 pixels",
        "24 out-of-bounds writes and 48 out-of-bounds reads",
        "index 1000, from block 3, thread 232",
        "104 out-of-bounds writes and 312 out-of-bounds reads",
        "index 1200, from block (0, 1), thread (0, 14)",
        "232 write conflicts",
        "A matching output does not prove that a kernel is free of races.",
        "Each function must hold exactly its pinned statements, in the pinned order.",
        "The text must hold the five definitions and no other code.",
        "Text with a backslash, a `??` trigraph or an unterminated `/*` comment fails every check.",
        "Recompute the kernel, demo and summary sections with its own code",
        "It reads the CUDA text with `check_cuda_source`, and it compares the limits and the rule with the lab constants.",
        VECTOR_SHA256,
        GRAY_SHA256,
        CUDA_SHA256,
        REPORT_SHA256,
        "This lab does not compile or run it.",
        "no NVIDIA GPU, no CUDA toolkit and no `nvcc`",
        "it gives no NVIDIA, production or distributed evidence",
        "learner understanding",
    ):
        assert phrase in guide, phrase


def test_committed_config_and_guide_hold_no_host_specific_paths():
    for path in (CONFIG, GUIDE):
        text = path.read_text(encoding="utf-8")
        assert text.isascii()
        for marker in ("/Users/", "/home/", "/private/", "C:\\", "localhost"):
            assert marker not in text
