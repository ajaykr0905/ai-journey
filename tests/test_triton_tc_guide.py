"""Tests that the Day 82 Tensor Core guide states what the code and config hold."""

import re
import shlex
from pathlib import Path

from ai_journey import triton_tc_cli
from ai_journey.triton_tc_config import load_config
from ai_journey.triton_tc_kernel import PINNED_SHA256, TC_CONFIG, VARIANTS
from ai_journey.triton_tc_model import ratio_text, run_model
from ai_journey.triton_tc_ptx import CLASSES
from ai_journey.triton_tc_report import LABELS
from ai_journey.triton_tc_ttgir import shared_bank

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "triton-tensor-core-lab.md"
CONFIG = ROOT / "config" / "day-82-triton-tc.json"
COMMAND = ["python", "-m", "ai_journey.triton_tc_cli"]
SHORT = {"a100-sxm4-80gb": "A100", "h100-sxm5-80gb": "H100"}


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def table_rows(header):
    """Return the cells of each row of the table whose header starts so."""

    lines = guide().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def documented_commands():
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", guide(), flags=re.DOTALL):
        for line in block.splitlines():
            words = shlex.split(line)
            for start in range(len(words)):
                if words[start : start + len(COMMAND)] == COMMAND:
                    commands.append(words[start + len(COMMAND) :])
    return commands


def test_the_documented_commands_parse_and_chain():
    parser = triton_tc_cli.build_parser()
    commands = documented_commands()
    parsed = [parser.parse_args(argv) for argv in commands]

    assert [argv[0] for argv in commands] == ["model", "verify", "compile", "verify"]
    assert parsed[1].report == parsed[0].output
    assert parsed[3].report == parsed[2].output
    assert parsed[3].dump == parsed[2].dump
    assert {item.config for item in parsed} == {Path("config/day-82-triton-tc.json")}
    for code in range(4):
        assert f"- {code}: " in guide()
    assert "`compile` exits 3 before it makes the output folder" in flat(guide())


def test_the_guide_names_the_pinned_text_config_and_labels():
    text = guide()
    assert PINNED_SHA256 in text
    assert f"`BLOCK_M = {TC_CONFIG.block_m}`" in text
    assert f"`BLOCK_K = {TC_CONFIG.block_k}`" in text
    for label in LABELS:
        assert f"`{label}`" in text
    assert "`TRITON_TC_DUMP_DIR`" in text
    assert "tests/test_triton_tc_compile_contract.py" in text


def test_the_variant_table_is_the_declared_variants():
    rows = table_rows("| Variant | A and B |")
    assert [
        (name.strip("`"), kind, precision.strip("`"), tensor, operand.strip("`"))
        for name, kind, precision, tensor, operand in rows
    ] == [
        (
            variant.name,
            variant.input_type,
            variant.input_precision,
            "yes" if variant.tensor_cores else "no",
            variant.mma_operand_type or "none",
        )
        for variant in VARIANTS
    ]


def test_the_rule_table_names_every_ptx_class():
    rows = table_rows("| Class | Rule |")
    named = {name.strip().strip("`") for row in rows for name in row[0].split(",")}
    assert named == set(CLASSES)


def test_the_model_table_is_the_model_output():
    model = run_model(load_config(CONFIG))
    expected = [
        [
            SHORT[device["key"]],
            str(row["size"]),
            *(
                f"{float(row['points'][path]['time_us']):.3f}"
                for path in ("fp32", "tf32", "bf16")
            ),
            ", ".join(
                row["points"][path]["bound"] for path in ("fp32", "tf32", "bf16")
            ),
            ratio_text(row["ratio"]),
        ]
        for device in model["devices"]
        for row in device["rows"]
    ]
    assert table_rows("| Device | N |") == expected
    assert "This is a model, not a measurement." in guide()


def test_the_declared_input_table_is_the_config():
    config = load_config(CONFIG)
    a100, h100 = config.device("sm_80"), config.device("sm_90")
    rows = {row[0]: row[1:3] for row in table_rows("| Input | A100 80GB SXM |")}

    assert rows["SMs"] == [str(a100.sms), str(h100.sms)]
    for path, label in (
        ("fp32", "FP32 FMA per clock per SM"),
        ("tf32", "TF32 dense FMA per clock per SM"),
        ("fp16", "FP16 dense FMA per clock per SM"),
        ("bf16", "BF16 dense FMA per clock per SM"),
    ):
        assert rows[label] == [str(a100.path(path).rate), str(h100.path(path).rate)]
    assert rows["FP32 peak, TFLOPS"] == ["19.5", "67"]
    assert rows["TF32 Tensor Core peak, TFLOPS"] == ["156", "989 with sparsity"]
    assert rows["Memory bandwidth, GB/s"] == ["2039", "3350"]
    for source in config.sources:
        assert f"<{source.url}>" in guide()


def test_the_bank_example_is_the_rule_output():
    banks = [
        shared_bank(
            row,
            0,
            columns=64,
            element_bytes=2,
            vec=8,
            per_phase=1,
            max_phase=8,
        )
        for row in range(8)
    ]
    assert "banks 0, 4, 8, 12, 16, 20, 24 and 28" in flat(guide())
    assert banks == [0, 4, 8, 12, 16, 20, 24, 28]


def test_the_guide_states_the_boundary_and_the_pending_learner_work():
    text = flat(guide())
    assert "Ajay must still watch Lecture 23 and Lecture 15." in text
    assert "Ajay must still read the CI dump." in text
    assert "Nothing here ran on an NVIDIA GPU." in text
    assert "that job had not run when this guide got its text" in text
    assert "They are not compiler output." in text
    assert "The lab measured no NVIDIA time." in text
