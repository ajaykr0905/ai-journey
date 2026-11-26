"""The Triton half of the Day 82 README section and the progress row.

Each README command must be a guide command, and each row of the README model
table must be the output of the declared model. The README must keep the
boundary: the instruction classes come only from the Linux CI compile and
its artifact, no run of that job happened yet, and the model is not a
measurement. The progress row must keep the lectures and the learner's own
work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import triton_tc_cli
from ai_journey.triton_tc_config import load_config
from ai_journey.triton_tc_model import ratio_text, run_model

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Compile for Tensor Cores in CI and time the Apple GPU matrix path\n"
GUIDE = "docs/triton-tensor-core-lab.md"
PINNED = Path("config/day-82-triton-tc.json")
COMMAND = "ai-journey-triton-tc"
STARTS = [["python", "-m", triton_tc_cli.__name__], [COMMAND]]
TABLE = (
    "| Device | N | fp32 us | tf32 us | bf16 us | Memory-bound paths "
    "| fp32 : tf32 : bf16 |"
)
SHORT = {"a100-sxm4-80gb": "A100", "h100-sxm5-80gb": "H100"}
PATHS = ("fp32", "tf32", "bf16")


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def flat(text):
    return " ".join(text.split())


def commands(text):
    found = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            # The guide sets PYTHONPATH before the command, so skip such words.
            while words and "=" in words[0]:
                words = words[1:]
            found += [
                words[len(start) :] for start in STARTS if words[: len(start)] == start
            ]
    return found


def rows(text, header):
    lines = text.splitlines()
    found = []
    for line in lines[lines.index(header) + 2 :]:
        if not line.startswith("|"):
            break
        found.append([cell.strip() for cell in line.strip("|").split("|")])
    return found


def test_each_readme_command_is_a_guide_command_and_chains():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    guide = (ROOT / GUIDE).read_text(encoding="utf-8")
    readme = commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.triton_tc_cli:main"
    assert readme and all(argv in commands(guide) for argv in readme)
    parsed = [triton_tc_cli.build_parser().parse_args(argv) for argv in readme]
    assert [item.command for item in parsed] == ["model", "verify", "compile", "verify"]
    assert {item.config for item in parsed} == {PINNED}
    assert parsed[1].report == parsed[0].output
    assert (parsed[3].report, parsed[3].dump) == (parsed[2].output, parsed[2].dump)


def test_the_readme_model_table_is_the_declared_model():
    model = run_model(load_config(ROOT / PINNED))
    expected = [
        [
            SHORT[device["key"]],
            str(row["size"]),
            *(f"{float(row['points'][path]['time_us']):.3f}" for path in PATHS),
            ", ".join(row["memory_bound"]) or "none",
            ratio_text(row["ratio"]),
        ]
        for device in model["devices"]
        for row in device["rows"]
    ]

    assert rows(readme_section(), TABLE) == expected
    assert len(expected) == 10
    assert "This table is a model, not a measurement." in flat(readme_section())


def test_the_readme_explains_the_ratio_as_the_guide_does():
    section = flat(readme_section())
    guide = flat((ROOT / GUIDE).read_text(encoding="utf-8"))

    for phrase in (
        "16 : 2 : 1",
        "A100 shows that ratio from N = 1024.",
        "about 1983 MHz",
        "about 1830 MHz",
        "14.77 : 2.00 : 1",
    ):
        assert phrase in section and phrase in guide, phrase


def test_readme_states_the_ci_job_and_the_boundary_of_the_triton_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "Only its `compile` command needs triton.",
        "the Triton instruction classes come only from the Linux CI compile and "
        "its artifact.",
        "That job did not run yet, so this section states no Day 82 instruction "
        "count.",
        "nothing runs a PTX text or a cubin.",
        "The speedup model uses declared datasheet inputs, and it is not a "
        "measurement.",
        "CI job `triton-tc-contracts` runs the Day 82 Triton tests on Linux with "
        "triton 3.9.0 and fails on a skipped test.",
        "No run of either job happened yet.",
        "Ajay must still watch Lectures 23 and 15, read the first CI dump",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  triton-tc-contracts:\n" in workflow
    codes = (
        triton_tc_cli.EXIT_OK,
        triton_tc_cli.EXIT_REJECTED,
        triton_tc_cli.EXIT_INVALID,
        triton_tc_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)


def test_the_day_82_row_keeps_the_triton_work_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 82 |")]
    _, code, lecture, manual = [cell.strip() for cell in row.strip("|").split("|")]

    for part in (
        "One pinned Triton matmul text with four declared variants for the "
        "declared targets sm_80 and sm_90 present.",
        "An offline compile contract that keeps the compiler dump present.",
        "A declared A100 and H100 speedup model of fp32, tf32 and bf16 present.",
        "No Triton compiler output, NVIDIA GPU, CUDA or Tensor Core rate present.",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 23 and 15 pending user confirmation"
    assert manual.startswith("Learner's own reading of the first CI compile dump")
    assert "A measured fp32 and bf16 ratio on an NVIDIA GPU pending" in manual
    assert manual.endswith(" pending")
    (updated,) = re.findall(r"^Last updated: (\d{4}-\d{2}-\d{2})$", progress, re.M)
    assert updated >= "2026-11-26"
