"""Tests that the Day 79 Triton guide states what the code and the recorded run hold."""

import json
import re
import shlex
from pathlib import Path

from ai_journey import triton_matmul_cli
from ai_journey.triton_matmul_config import PINNED_PLAN
from ai_journey.triton_matmul_kernel import (
    AUTOTUNE_CONFIGS,
    GROUPED_ORDER,
    TRITON_SOURCE,
    program_tile,
    source_sha256,
)
from ai_journey.triton_matmul_loc import counted_texts
from ai_journey.triton_matmul_performance import (
    LATER_GPU_RUN,
    METAL_RATES,
    METAL_REPORTS,
    PERFORMANCE_SHAPES,
    TRITON_GPU,
)
from ai_journey.triton_matmul_report import INTERPRETER_BOUNDARY
from ai_journey.triton_matmul_runtime import (
    CI_WHEELS,
    CORE_METADATA_SHA256,
    REQUIRES_DIST,
    REQUIRES_PYTHON,
)
from test_triton_matmul_recorded import RECORDED_REPORT_JSON, RECORDED_SHA256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/triton-matmul-lab.md"
COMMAND = ["python", "-m", "ai_journey.triton_matmul_cli"]
LABS = {"day-79": "Day 79", "day-76": "Day 76", "day-77": "Day 77"}
LANGUAGES = {
    "Triton (Python)": "Triton",
    "CUDA C++": "CUDA C++",
    "Metal Shading Language": "Metal",
}


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def documented_commands():
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", guide(), flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            for start in range(len(words)):
                if words[start : start + len(COMMAND)] == COMMAND:
                    commands.append(words[start + len(COMMAND) :])
    return commands


def label(shape):
    return f"{shape.m} x {shape.n} x {shape.k}"


def test_the_documented_commands_parse_and_chain():
    parser = triton_matmul_cli.build_parser()
    commands = documented_commands()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["loc", "check", "verify", "check"]
    assert parsed[2].report == parsed[1].output
    assert parsed[1].config == parsed[2].config == parsed[3].config
    assert (ROOT / parsed[1].config).is_file()
    assert f"`{triton_matmul_cli.PROG}`" in guide()
    for code in (
        triton_matmul_cli.EXIT_OK,
        triton_matmul_cli.EXIT_REJECTED,
        triton_matmul_cli.EXIT_INVALID,
        triton_matmul_cli.EXIT_UNAVAILABLE,
    ):
        assert f"\n- {code}: " in guide()


def test_the_guide_shows_the_k_loop_and_the_digest_of_the_pinned_text():
    (block,) = re.findall(r"```python\n(.*?)```", guide(), flags=re.DOTALL)

    assert block in TRITON_SOURCE
    assert 'acc = tl.dot(a, b, acc, input_precision="ieee")' in block
    assert f"`{source_sha256()}`" in guide()
    assert len(GROUPED_ORDER) == 9
    assert "Nine lines map each program" in guide()


def test_the_config_table_is_the_declared_list():
    text = guide()

    for config in AUTOTUNE_CONFIGS:
        row = (
            f"| `{config.name}` | {config.block_m} x {config.block_n} x "
            f"{config.block_k} | {config.num_warps} | {config.num_stages} | "
            f"{config.tile_bytes:,} |"
        )
        assert row in text, row
    assert PINNED_PLAN.configs == AUTOTUNE_CONFIGS


def test_the_grouped_order_example_is_the_cpu_model():
    order = [program_tile(pid, 10, 3, 4) for pid in range(8)]

    assert order == [(row, 0) for row in range(4)] + [(row, 1) for row in range(4)]
    assert "IDs 0 to 3 give the tiles (0, 0) to (3, 0)" in flat(guide())
    assert "IDs 4 to 7 give the tiles (0, 1) to (3, 1)" in flat(guide())


def test_six_day_74_shapes_have_a_row_stride_off_16_bytes():
    unaligned = [
        shape
        for shape in PINNED_PLAN.shapes
        if (shape.k * 4) % 16 or (shape.n * 4) % 16
    ]

    assert len(unaligned) == 6
    assert "Six of the nine Day 74 shapes have a row stride" in flat(guide())


def test_the_line_count_table_is_the_count_of_each_real_text():
    text = guide()

    for item in counted_texts():
        count = item.count()
        host = "none" if count.host is None else count.host
        row = (
            f"| {LABS[item.lab]} | {LANGUAGES[item.language]} | `{item.name}` | "
            f"{count.kernel} | {host} | {count.other} | {count.comment} | "
            f"{count.blank} |"
        )
        assert row in text, row


def test_the_correctness_table_is_the_recorded_container_run():
    report = json.loads(RECORDED_REPORT_JSON)
    text = guide()

    assert f"`{RECORDED_SHA256}`" in text
    for case in report["correctness"]["cases"]:
        shape = case["shape"]
        normal = [run for run in case["runs"] if run["family"] == "normal"]
        exact = [run for run in case["runs"] if run["family"] == "exact"]
        held = sum(run["rounded_reference_differences"] == 0 for run in exact)
        first = normal[0]
        row = (
            f"| {case['index']} | {shape['m']} x {shape['n']} x {shape['k']} | "
            f"{case['error_bound']:.3g} | {first['max_relative_error']:.3g} | "
            f"{first['bound_ratio']:.3g} | "
            f"{first['rounded_reference_differences']:,} of "
            f"{shape['m'] * shape['n']:,} | {held} of {len(exact)} |"
        )
        assert row in text, row
    environment = report["environment"]
    for fact in (
        f"CPython {environment['python']}",
        f"triton {environment['triton']}, torch {environment['torch']}, NumPy "
        f"{environment['numpy']}",
    ):
        assert fact in flat(text), fact


def test_the_errors_of_long_k_cases_are_below_the_day_76_replay_errors():
    report = json.loads(RECORDED_REPORT_JSON)
    contract = (ROOT / "docs/gpu-tiled-cuda-contract.md").read_text(encoding="utf-8")
    replay = {}
    for line in contract.splitlines():
        found = re.match(
            r"\| (\d) \| (\d+) x (\d+) x (\d+) \| `[0-9a-f]{64}` \| ([0-9.e-]+) \|",
            line,
        )
        if found:
            replay[int(found.group(1))] = float(found.group(5))

    assert sorted(replay) == list(range(9))
    for case in report["correctness"]["cases"]:
        error = case["runs"][0]["max_relative_error"]
        if case["shape"]["k"] >= 100:
            assert error < replay[case["index"]]
    assert "the two values are 3.33e-08 and 2.295e-07" in flat(guide())


def test_the_rate_tables_are_the_recorded_metal_rates_and_no_gpu_number():
    text = guide()
    order = [
        ("day-76", "matmul_tiled_32"),
        ("day-77", "matmul_s3_block_4x4"),
        ("day-77", "matmul_s7_interior"),
        ("day-76", "torch_matmul"),
        ("day-77", "torch_matmul"),
    ]

    for shape in PERFORMANCE_SHAPES:
        cells = []
        for lab, kernel in order:
            (rate,) = [
                item
                for item in METAL_RATES
                if (item.lab, item.kernel, item.shape) == (lab, kernel, shape)
            ]
            cells.append(f"{rate.median:.1f} ({rate.low:.1f} to {rate.high:.1f})")
        assert f"| {label(shape)} | " + " | ".join(cells) + " |" in text
        assert (
            f"| {label(shape)} | {TRITON_GPU} | not measured | not measured |" in text
        )
    for report in METAL_REPORTS:
        assert f"`{report.report_sha256}`" in text
        date, time, zone = report.run.split()
        assert f"{date} at {time} {zone}" in flat(text)


def test_the_ci_facts_are_the_recorded_pypi_metadata():
    text = guide()

    for wheel in CI_WHEELS:
        assert f"`{wheel.filename}` | {wheel.size:,} | `{wheel.sha256}` |" in text
    assert f"`Requires-Python: {REQUIRES_PYTHON}`" in text
    assert f"`{REQUIRES_DIST[0]}`" in text
    assert f"`{CORE_METADATA_SHA256}`" in text


def test_the_later_gpu_run_lists_the_declared_steps():
    text = guide()
    section = text.split("## What a later GPU run must do\n", 1)[1].split("\n## ", 1)[0]
    steps = re.findall(r"^\d+\. ", section, flags=re.MULTILINE)

    assert len(steps) == len(LATER_GPU_RUN) == 8
    for phrase in (
        "leave `TRITON_INTERPRET` unset",
        "`AUTOTUNE_CONFIGS` and the key M, N, K",
        "2 calls that are not timed and then 11 timed samples",
        "`sgemm_row_major` with cuBLAS",
        "1.1 or more and the IQR ranges do not overlap",
        "The tuning times are not results.",
    ):
        assert phrase in flat(section), phrase
        assert phrase.replace("`", "") in " ".join(LATER_GPU_RUN), phrase


def test_the_guide_names_the_pending_learner_work_and_the_boundary():
    text = flat(guide())

    assert "Ajay must still watch Lecture 14 and Lecture 18." in text
    assert "Ajay must still run the steps of a later GPU run" in text
    for phrase in (
        "Nothing here runs on a GPU.",
        "host-specific Apple GPU (Metal) measurement",
        "That time measures the CPU interpreter, so it is not a performance number.",
        "The repository CI has no job for this lab yet.",
    ):
        assert phrase in text, phrase


def test_the_boundary_section_names_each_field_with_its_value():
    section = guide().split("## Evidence boundary\n", 1)[1].split("\n## ", 1)[0]
    flags = {
        name: value
        for name, value in INTERPRETER_BOUNDARY.items()
        if name != "statement"
    }
    held = sorted(name for name, value in flags.items() if value)
    refused = [name for name, value in flags.items() if not value]

    assert held == [
        "metal_rates_copied_from_recorded_reports",
        "triton_interpreter_on_cpu",
    ]
    true_part, false_part = flat(section).split("These fields are false:", 1)
    for name in held:
        assert f"`{name}`" in true_part, name
    for name in refused:
        assert f"`{name}`" in false_part, name
