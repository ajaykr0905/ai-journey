"""The Triton half of the Day 81 README section and the progress row.

Each README command must be a guide command, and each row of the README
memory table must be the declared model. The README must keep the boundary:
the kernel runs only in the CPU interpreter, and no run of it happened yet.
The progress row must keep the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import triton_flash_cli
from ai_journey.triton_flash_config import PINNED_PLAN, load_plan
from ai_journey.triton_flash_memory import MEMORY_TABLE, mebibytes
from ai_journey.triton_flash_reference import FAMILIES

ROOT = Path(__file__).resolve().parents[1]
HEADING = (
    "\n## Check flash attention in the Triton interpreter and time it on the "
    "Apple GPU\n"
)
GUIDE = "docs/triton-flash-attention-lab.md"
PINNED = Path("config/day-81-triton-flash.json")
COMMAND = "ai-journey-triton-flash"
STARTS = [["python", "-m", triton_flash_cli.__name__], [COMMAND]]
TABLE = "| N | d | Naive MiB | Flash MiB | Memory ratio | Traffic ratio |"


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

    assert project["project"]["scripts"][COMMAND] == "ai_journey.triton_flash_cli:main"
    assert readme and all(argv in commands(guide) for argv in readme)
    model, check, verify = (
        triton_flash_cli.build_parser().parse_args(argv) for argv in readme
    )
    assert (model.command, check.command, verify.command) == (
        "model",
        "check",
        "verify",
    )
    assert check.config == verify.config == PINNED
    assert verify.report == check.output


def test_the_readme_memory_table_is_the_declared_model():
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
                f"{item['traffic_ratio']:.2f}",
            ]
        )

    assert rows(readme_section(), TABLE) == expected
    assert "This model is not a measured GPU footprint." in flat(readme_section())


def test_the_readme_counts_the_runs_of_the_pinned_plan():
    plan = load_plan(ROOT / PINNED)
    runs = len(plan.cases) * len(plan.configs) * len(FAMILIES)
    section = flat(readme_section())

    assert plan == PINNED_PLAN
    assert runs == 48
    assert (
        f"The pinned Triton plan holds {len(plan.cases)} cases, and each case runs "
        f"with {len(plan.configs)} configs and {len(FAMILIES)} input families. "
        f"Thus a check holds {runs} runs."
    ) in section


def test_readme_states_the_ci_job_and_the_boundary_of_the_triton_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "Only its `check` command needs triton, and it runs only the Triton CPU "
        "interpreter.",
        "No interpreter run of the pinned text happened yet.",
        "the Triton kernel runs only in the Triton CPU interpreter.",
        "No GPU compiled or ran the Triton text, and the lab reports no Triton "
        "time.",
        "The Triton memory table is a model.",
        "CI job `triton-flash-contracts` runs the Triton tests in the CPU "
        "interpreter on Linux with triton 3.9.0 and fails on a skipped test.",
        "No run of either job happened yet.",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  triton-flash-contracts:\n" in workflow
    codes = (
        triton_flash_cli.EXIT_OK,
        triton_flash_cli.EXIT_REJECTED,
        triton_flash_cli.EXIT_INVALID,
        triton_flash_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)


def test_the_day_81_row_keeps_the_triton_work_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 81 |")]
    _, code, lecture, manual = [cell.strip() for cell in row.strip("|").split("|")]

    for part in (
        "One pinned Triton flash attention forward text with four declared configs",
        "declared per-head memory and traffic model of naive against flash "
        "attention present",
        "No Triton interpreter run, Triton GPU, CUDA or FlashAttention library rate "
        "present",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 12 and 36 pending user confirmation"
    assert "FlashAttention-2 and FlashAttention-3 papers" in manual
    assert "A read of the first CI interpreter log" in manual
    assert manual.endswith(" pending")
