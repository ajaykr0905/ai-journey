"""Tests that the Day 81 guide states what the code, the plan and the model hold."""

import re
import shlex
from pathlib import Path

import pytest

from ai_journey import triton_flash_cli
from ai_journey.triton_flash_config import PINNED_PLAN
from ai_journey.triton_flash_correctness import check_case
from ai_journey.triton_flash_kernel import (
    FLASH_CONFIGS,
    TRITON_SOURCE,
    TritonFlashError,
    source_sha256,
)
from ai_journey.triton_flash_memory import MEMORY_TABLE, mebibytes
from ai_journey.triton_flash_reference import AttentionCase
from ai_journey.triton_flash_report import build_report
from test_triton_flash_correctness import StandInBackend
from test_triton_flash_emulated import EmulatedBackend, install

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/triton-flash-attention-lab.md"
COMMAND = ["python", "-m", "ai_journey.triton_flash_cli"]


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def table_rows(header):
    lines = guide().splitlines()
    start = lines.index(header) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_every_documented_command_parses_and_names_the_plan():
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", guide(), flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            for start in range(len(words)):
                if words[start : start + len(COMMAND)] == COMMAND:
                    commands.append(words[start + len(COMMAND) :])

    assert [words[0] for words in commands] == ["model", "check", "verify"]
    for words in commands:
        triton_flash_cli.build_parser().parse_args(words)
        if "--config" in words:
            path = words[words.index("--config") + 1]
            assert (ROOT / path).is_file()
    assert f"`{triton_flash_cli.PROG}`" in guide()


def test_the_guide_names_the_digest_and_quotes_the_key_loop():
    text = guide()
    loop = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL).group(1)

    assert f"`{source_sha256()}`" in text
    assert loop in TRITON_SOURCE
    assert "alpha = tl.exp(m_i - m_new)" in loop


def test_the_memory_table_is_the_model():
    header = (
        "| N | d | Naive MiB | Flash MiB | Memory ratio | Naive traffic MiB | "
        "Flash traffic MiB | Traffic ratio |"
    )
    expected = []
    for row in MEMORY_TABLE:
        item = row.to_mapping()
        expected.append(
            [
                str(row.sequence),
                str(row.head_dim),
                mebibytes(item["naive_bytes"]),
                mebibytes(item["flash_bytes"]),
                f"{item['memory_ratio']:.2f}",
                mebibytes(item["naive_traffic_bytes"]),
                mebibytes(item["flash_traffic_bytes"]),
                f"{item['traffic_ratio']:.2f}",
            ]
        )

    assert table_rows(header) == expected
    assert "It is not a measured GPU footprint." in guide()


def test_the_config_and_case_tables_are_the_declared_values():
    configs = table_rows(
        "| Config | BLOCK_M x BLOCK_N | Warps | Stages | Tile bytes, D = 64 | "
        "Tile bytes, D = 128 |"
    )
    assert configs == [
        [
            f"`{config.name}`",
            f"{config.block_m} x {config.block_n}",
            str(config.num_warps),
            str(config.num_stages),
            f"{config.tile_bytes(64):,}",
            f"{config.tile_bytes(128):,}",
        ]
        for config in FLASH_CONFIGS
    ]
    cases = table_rows("| BH | N | D | Causal |")
    assert cases == [
        [
            str(case.batch_heads),
            str(case.sequence),
            str(case.head_dim),
            "yes" if case.causal else "no",
        ]
        for case in PINNED_PLAN.cases
    ]
    assert f"with seed {PINNED_PLAN.seed}" in flat(guide())


def test_the_cited_ratios_come_from_the_stand_in_and_the_changed_text(monkeypatch):
    report = build_report(
        PINNED_PLAN,
        StandInBackend(),
        clock=lambda: 0.0,
        load_average=lambda: (0.0, 0.0, 0.0),
    )
    summary = report["summary"]
    text = flat(guide())

    assert summary["runs"] == 48 and "a check holds 48 runs" in text
    assert (
        f"ratio of the pinned plan is {summary['max_output_bound_ratio']:.4f}" in text
    )
    assert f"ratio is {summary['max_lse_bound_ratio']:.4f}" in text

    install(monkeypatch)
    changed = TRITON_SOURCE.replace(
        "alpha = tl.exp(m_i - m_new)", "alpha = tl.exp(m_new - m_new)"
    )
    with pytest.raises(TritonFlashError) as caught:
        check_case(
            EmulatedBackend(changed),
            seed=81,
            index=3,
            case=AttentionCase(1, 129, 64, True),
            configs=(FLASH_CONFIGS[3],),
        )
    ratio = float(re.search(r"is ([0-9.]+) times", str(caught.value)).group(1))
    assert 740 < ratio < 780
    assert "The factor is about 760 in the case N = 129 with D = 64." in text


def test_the_guide_keeps_the_boundary_and_the_pending_work():
    text = flat(guide())

    assert "Ajay must still watch Lecture 12 and Lecture 36." in text
    assert "Ajay must still read both papers." in text
    assert "No GPU compiled or ran the text." in text
    assert "The lab reports no Triton GPU time and tunes no config." in text
    assert "The lab implements none of these features" in text
    for word in ("CUDA run", "measured on the GPU", "occupancy", "production"):
        assert word not in text
    assert ";" not in re.sub(r"```.*?```", "", guide(), flags=re.DOTALL)


def test_no_text_says_that_a_ci_job_or_the_interpreter_ran_the_kernel():
    # The interpreter tests need a Linux job with triton. Until the workflow
    # runs them, the guide and the test module must say that no job does.
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    interpreter_tests = flat(
        (ROOT / "tests/test_triton_flash_interpreter.py").read_text(encoding="utf-8")
    )
    text = flat(guide())
    assert "test_triton_flash_" not in workflow
    assert "No CI job runs the interpreter tests yet." in text
    assert "no CI job runs them yet" in interpreter_tests
    assert "run in the Linux CI job" not in interpreter_tests
    assert "such as a CI job" not in text
    assert "read the CI log" not in text
    assert "Ajay must still add a Linux CI job for the interpreter tests" in text
    # The cited fault ratio comes from the NumPy model, not the interpreter.
    assert (
        "Under the NumPy model of `triton.language`, a text without the rescale "
        "step breaks the bound" in text
    )
    assert "The bound holds for" not in text
    assert "No interpreter run checked the bound." in text
    # The paper notes are a summary from memory, not a checked reading.
    assert "These notes come from the papers." not in text
    assert "Nobody checked them against the papers." in text
