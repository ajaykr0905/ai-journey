"""Tests that the Day 80 Triton IR guide states what the code and the fixtures hold."""

import re
import shlex
from pathlib import Path

from ai_journey import triton_ir_cli
from ai_journey.triton_ir_compare import NEEDS_FULL_DUMPS
from ai_journey.triton_ir_manifest import RECORDED_MANIFEST_SHA256
from ai_journey.triton_ir_report import LABELS, build_report
from ai_journey.triton_matmul_kernel import source_sha256

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "triton-ir-lab.md"
FIXTURES = ROOT / "tests" / "fixtures" / "day80_triton_ir"
PTX = FIXTURES / "sm_80" / "m32_n64_k32_g8_w2_s5" / "ptx.ptx"
COMMAND = ["python", "-m", "ai_journey.triton_ir_cli"]


def guide():
    return GUIDE.read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def report():
    return build_report(FIXTURES, clock=lambda: 0.0, load_average=lambda: (0, 0, 0))


def documented_commands():
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", guide(), flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            for start in range(len(words)):
                if words[start : start + len(COMMAND)] == COMMAND:
                    commands.append(words[start + len(COMMAND) :])
    return commands


def test_the_documented_commands_parse_and_chain():
    parser = triton_ir_cli.build_parser()
    commands = documented_commands()
    parsed = [parser.parse_args(argv) for argv in commands]
    assert [argv[0] for argv in commands] == [
        "summarize",
        "verify",
        "stage-map",
        "check",
    ]
    assert parsed[1].report == parsed[0].output
    assert {item.fixtures for item in parsed} == {
        Path("tests/fixtures/day80_triton_ir")
    }
    assert (ROOT / parsed[0].fixtures / "index.json").is_file()
    text = flat(guide())
    for code in range(4):
        assert f"- {code}: " in guide()
    assert "exits 3 before it makes the output folder" in text


def test_the_guide_names_the_pinned_digests_and_the_labels():
    text = guide()
    assert source_sha256() in text
    assert RECORDED_MANIFEST_SHA256 in text
    for label in LABELS:
        assert f"`{label}`" in text


def test_the_guide_counts_are_the_counts_of_the_fixtures():
    built = report()
    families = built["ptx"]["families"]
    text = flat(guide())
    rows = {
        "fma.rn.f32": families["fma.rn.f32"],
        "cp.async` copies": families["cp.async"],
        "cp.async.commit_group": families["cp.async.commit_group"],
        "cp.async.wait_group": families["cp.async.wait_group"],
        "ld.shared": families["ld.shared"],
        "st.shared": families["st.shared"],
        "ldmatrix": families["ldmatrix"],
        "bar.sync": families["bar.sync"],
        "st.global": families["st.global"],
        "ld.global": families["ld.global"],
        "mma": families["mma"],
        "setp": families["setp"],
        "selp": families["selp"],
    }
    for name, count in rows.items():
        assert (
            f"| `{name}` | {count} |" in guide() or f"| `{name} | {count} |" in guide()
        )
    assert f"The entry holds {built['ptx']['instructions']} instructions." in text
    assert f"Thus the PTX holds {built['summary']['cp_async_all']} instructions" in text
    assert f"The TTIR holds {built['ttir']['ops']} ops." in text
    assert f"The TTGIR holds {built['ttgir']['ops']} ops." in text
    loop = built["ptx"]["loops"][0]
    assert f"on line {loop['first_line']} of `ptx.ptx`" in text
    assert f"back on line {loop['last_line']}" in text
    assert f"It holds {loop['instructions']} instructions." in text
    missing = built["stage_map"]["without_source_line"]
    assert (
        f"The TTIR has {missing['ttir_ops']} ops with no kernel line, the TTGIR has "
        f"{missing['ttgir_ops']} and the PTX has {missing['ptx_instructions']}." in text
    )


def test_the_guide_line_numbers_hold_the_named_instructions():
    lines = PTX.read_text(encoding="ascii").split("\n")
    assert lines[2216 - 1].startswith("$L__BB0_3:")
    assert lines[2222 - 1].strip() == "cp.async.wait_group \t6;"
    assert lines[2223 - 1].strip() == "bar.sync \t0;"
    assert lines[3441 - 1].strip() == "bar.sync \t0;"
    assert lines[3869 - 1].strip() == "@%p186 bra \t$L__BB0_3;"
    body = lines[2226 - 1 : 2324]
    assert sum("ld.shared.v4.b32" in line for line in body) == 96
    body = lines[2326 - 1 : 3349]
    assert sum("fma.rn.f32" in line for line in body) == 1024
    body = lines[3449 - 1 : 3859]
    assert sum("cp.async.ca.shared.global" in line for line in body) == 48
    assert lines[3864 - 1].strip() == "add.s32 \t%r2262, %r2262, 32;"


def test_the_ring_table_and_the_target_ranges_are_the_manifest():
    built = report()
    for row in built["comparison"]["compiles"]:
        if row["target"] == "sm_80":
            assert (
                f"| `{row['config']}` | {row['num_stages']} | {row['tile_bytes']} | "
                f"{row['shared_bytes']} | {row['ring_buffers']} |"
            ) in guide()
    ptx = [row["line_change"]["ptx"] for row in built["comparison"]["targets"]]
    llir = [-row["line_change"]["llir"] for row in built["comparison"]["targets"]]
    text = flat(guide())
    assert f"{min(ptx)} to {max(ptx)} more lines" in text
    assert f"{min(llir)} to {max(llir)} fewer lines" in text
    assert built["comparison"]["targets"][5]["ttgir_is_retargeted_sm_80"] is True
    assert len(NEEDS_FULL_DUMPS) == 4


def test_the_guide_keeps_the_boundary_and_the_pending_work():
    text = flat(guide())
    assert "Nothing here ran on an NVIDIA GPU." in text
    assert "No `ptxas` register report exists on this host" in text
    assert "Ajay must still watch Lecture 29 and Lecture 28." in text
    assert "Ajay must still read the committed PTX himself." in text
    assert "The repository holds one run only" in text
    lowered = text.lower()
    for claim in ("occupancy of", "tflop", "gb/s", "we measured", "ran on an h100"):
        assert claim not in lowered
    assert ";" not in re.sub(r"```.*?```|`[^`]*`", "", guide(), flags=re.DOTALL)
