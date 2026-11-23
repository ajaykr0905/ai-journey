"""The fused softmax half of the Day 79 README section and the progress row.

Each README command must be a guide command, and each README rate must be the
median that the guide quotes from the pinned run. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import metal_fused_cli
from ai_journey.metal_fused_config import load_config
from ai_journey.metal_fused_plan import PLAN
from ai_journey.metal_fused_report import MEASUREMENT_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Port the matmul to Triton and fuse a softmax on the Apple GPU\n"
GUIDE = "docs/metal-fused-softmax-lab.md"
TRITON_GUIDE = "docs/triton-matmul-lab.md"
PINNED = Path("config/day-79-metal-fused.json")
CEILINGS = Path("/tmp/day-74-gpu-ceilings.json")
COMMAND = "ai-journey-metal-fused"
STARTS = [["python", "-m", metal_fused_cli.__name__], [COMMAND]]
README_TABLE = "| Point | G | U0 | F1 | F2 | F3 | `torch.softmax` | Hypotheses held |"
POINT_TABLE = (
    "| Step | Median GB/s (IQR range) | Modeled GB/s | Of the ceiling "
    "| Over its base | Over U0 | Over torch | Hypothesis |"
)
SIDES = ("U0", "F1", "F2", "F3", "torch")
# Each README figure, with the guide phrase that states the measured value.
FIGURES = (
    (
        "Each cell gives the median effective GB/s of 50 samples.",
        "Each rate is the median effective GB/s of 50 samples",
    ),
    (
        "The predictions held at 18 of 20 points: 4 steps at 5 points each.",
        "The predictions held at 18 of 20 points: 4 steps at 5 points each.",
    ),
    (
        "At the four points of 4.1 to 4.2 million elements, F1 was 1.632 to 1.937 "
        "times as fast as U0.",
        "at the four points of 4.1 to 4.2 million elements, F1 was 1.632 to 1.937 "
        "times as fast as U0.",
    ),
    (
        "At 4096 x 128, F1 showed no clear difference from U0, with a ratio of "
        "1.004.",
        "F1 showed no clear difference from U0, with a ratio of 1.004.",
    ),
    (
        "At the ragged point, `torch.softmax` processed the padded matrix, about "
        "twice the elements of the rows.",
        "But `torch.softmax` processed the padded matrix, about twice the elements "
        "of the rows.",
    ),
    (
        "The report recorded load averages of 8.00, 8.05 and 8.17 before the run. "
        "It recorded 7.72, 7.99 and 8.14 after the run.",
        "The report recorded load averages of 8.00, 8.05 and 8.17 before the run. "
        "It recorded 7.72, 7.99 and 8.14 after the run.",
    ),
    (
        "All 32 outputs (8 cases, 4 steps) met the four rules in both runs.",
        "All 32 outputs (8 cases, 4 steps) met the four rules in both runs.",
    ),
)


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def guide():
    return (ROOT / GUIDE).read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def commands(text):
    found = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            while words and re.fullmatch(r"[A-Z_]+=\S*", words[0]):
                words = words[1:]
            found += [
                words[len(start) :] for start in STARTS if words[: len(start)] == start
            ]
    return found


def rows(lines, start):
    found = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        found.append([cell.strip() for cell in line.strip("|").split("|")])
    return found


def guide_points():
    """The heading, medians and held count of each point table of the guide."""

    lines = guide().splitlines()
    points = []
    for index, line in enumerate(lines):
        if line != POINT_TABLE:
            continue
        (heading,) = [
            text for text in lines[index - 2 : index] if text.startswith("#### ")
        ]
        table = {row[0]: row for row in rows(lines, index)}
        assert tuple(table) == SIDES
        medians = [table[side][1].split()[0] for side in SIDES]
        held = sum(table[step][7] == "held" for step in SIDES[:4])
        points.append((heading.removeprefix("#### "), medians, held))
    return points


def day_79_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 79 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_each_readme_command_is_a_guide_command_and_chains():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.metal_fused_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    run, verify = (metal_fused_cli.build_parser().parse_args(argv) for argv in readme)
    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert run.ceilings == verify.ceilings == CEILINGS
    assert verify.report == run.output


def test_the_readme_table_holds_the_medians_of_the_pinned_run():
    lines = readme_section().splitlines()
    table = rows(lines, lines.index(README_TABLE))
    points = guide_points()
    config = load_config(ROOT / PINNED)

    assert [step.step for step in PLAN] == list(SIDES[:4])
    assert len(table) == len(points) == len(config.benchmark.shapes) == 5
    for row, (heading, medians, held) in zip(table, points):
        point, group = row[0], row[1]
        assert heading.startswith(point.removesuffix(", 2048 rows")), heading
        assert f"(G = {group}," in heading, heading
        assert row[2:7] == medians, heading
        assert row[7] == f"{held} of 4", heading
    assert sum(held for _, _, held in points) == 18
    assert config.benchmark.repeats == 50
    assert config.comparison.min_faster_ratio == 1.1


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())
    text = flat(guide())

    for claim, measured in FIGURES:
        assert claim in section, claim
        assert measured in text, measured
    assert (
        "All values in the softmax table and in the two paragraphs after it come "
        "from host-specific Apple GPU (Metal) measurements on one shared host. They "
        "are not portable figures."
    ) in section


def test_readme_states_the_exit_codes_ci_jobs_and_boundary_of_the_labs():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "or that it cannot run Metal kernels through PyTorch.",
        "No CUDA kernel ran, the labs used no NVIDIA GPU, and they did not measure "
        "cuBLAS.",
        "PyTorch does not name the kernel of `torch.softmax`.",
        "The labs read no hardware counter, so each statement about a cache is an "
        "inference from timings.",
        "CI job `triton-matmul-contracts` runs the Triton tests in the triton 3.9.0 "
        "CPU interpreter on Linux.",
        "CI job `metal-fused-contracts` runs the fused softmax tests on Linux with "
        "CPU torch.",
        f"({GUIDE})",
        f"({TRITON_GUIDE})",
    ):
        assert phrase in section, phrase
    for job in ("triton-matmul-contracts", "metal-fused-contracts"):
        assert f"\n  {job}:\n" in workflow, job
    codes = (
        metal_fused_cli.EXIT_OK,
        metal_fused_cli.EXIT_REJECTED,
        metal_fused_cli.EXIT_INVALID,
        metal_fused_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
    for flag in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "triton_gpu_kernel_executed",
        "vendor_kernel_known",
        "hardware_counters_used",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag


def test_the_day_79_row_keeps_the_lectures_and_learner_evidence_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8").splitlines()
    day, code, lecture, manual = day_79_row()
    days = [line.split("|")[1].strip() for line in progress if line.startswith("| ")]

    assert days[days.index("78") + 1] == day == "79"
    for part in (
        "One pinned Triton matmul text with six declared configs",
        "checked in the Triton CPU interpreter",
        "a line count of the Triton, CUDA and Metal matmul texts",
        "Four Metal softmax steps from unfused to online",
        "one Apple GPU run that times each step beside torch.softmax",
        "installed commands, and CI gates present",
        "No Triton GPU, CUDA or cuBLAS rate present",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 14 and 18 pending user confirmation"
    assert manual.startswith("Learner-run Triton port")
    assert manual.endswith("on an NVIDIA or AMD GPU pending")
    (updated,) = re.findall(
        r"^Last updated: (\d{4}-\d{2}-\d{2})$", "\n".join(progress), re.M
    )
    assert updated >= "2026-11-23"
    for name in (GUIDE, TRITON_GUIDE):
        assert (ROOT / name).is_file()
