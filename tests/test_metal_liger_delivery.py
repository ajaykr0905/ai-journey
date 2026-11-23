"""The fused kernel half of the Day 80 README section and the progress row.

Each README command must be a guide command, and each README cell must be the
value that the guide quotes from the pinned run. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import metal_liger_cli
from ai_journey.metal_liger_config import load_config
from ai_journey.metal_liger_report import MEASUREMENT_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Read the Triton compiler output and fuse LLM layers on the Apple GPU\n"
GUIDE = "docs/metal-liger-lab.md"
PINNED = Path("config/day-80-metal-liger.json")
CEILINGS = Path("/tmp/day-74-gpu-ceilings.json")
COMMAND = "ai-journey-metal-liger"
STARTS = [["python", "-m", metal_liger_cli.__name__], [COMMAND]]
README_TABLE = (
    "| Family | Shape | Kernel median (IQR), us | Kernel GB/s | Of the ceiling "
    "| PyTorch median (IQR), us | Ratio, verdict |"
)
GUIDE_TABLE = (
    "| Family | Shape | Calls | Kernel median (IQR), us | Kernel GB/s (range) "
    "| Of the ceiling | PyTorch median (IQR), us | PyTorch GB/s (range) "
    "| Ratio, verdict |"
)
# Each README figure, with the guide phrase that states the measured value.
FIGURES = (
    ("The predictions held at 8 of 8 points.",) * 2,
    ("All 12 cases met the three rules in both runs.",) * 2,
    (
        "the ceiling of 124.61 GB/s from a fresh Day 74 run.",
        "Its ceiling is 124.61 GB/s",
    ),
    (
        "131072000 bytes at 32000 columns and 205852672 bytes at 50257 columns.",
        "| 1024 | 32000 | 32768000 | 131072000 (125 MiB) | 4096 |",
    ),
    (
        "These are counts of a model, not measured allocations.",
        "They are counts of a model.",
    ),
    (
        "The Day 80 report recorded load averages of 7.73, 8.02 and 7.68 before "
        "the run. It recorded 7.10, 7.87 and 7.63 after the run.",
        "The Day 80 report recorded load averages of 7.73, 8.02 and 7.68 before "
        "the run. It recorded 7.10, 7.87 and 7.63 after the run.",
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
    readme = commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.metal_liger_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    run, verify = (metal_liger_cli.build_parser().parse_args(argv) for argv in readme)
    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert run.ceilings == verify.ceilings == CEILINGS
    assert verify.report == run.output


def test_the_readme_table_holds_the_cells_of_the_pinned_run():
    readme = rows(readme_section(), README_TABLE)
    pinned = rows(guide(), GUIDE_TABLE)
    config = load_config(ROOT / PINNED)

    assert len(readme) == len(pinned) == 8
    assert sum(len(shapes) for shapes in config.benchmark.shapes.values()) == 8
    for mine, measured in zip(readme, pinned):
        assert mine[:2] == measured[:2]
        assert mine[2] == measured[3]
        assert mine[3] == measured[4].split()[0]
        assert mine[4:6] == measured[5:7]
        assert mine[6] == measured[8]
    assert config.comparison.min_faster_ratio == 1.1


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())
    text = flat(guide())

    for claim, measured in FIGURES:
        assert claim in section, claim
        assert measured in text, measured
    assert (
        "All values in the fused kernel table and in the paragraph after it come "
        "from host-specific Apple GPU (Metal) measurements on one shared host. They "
        "are not portable figures."
    ) in section


def test_readme_states_the_ci_job_and_boundary_of_the_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "or that it cannot run Metal kernels through PyTorch.",
        "The fused kernel times come from one Apple GPU through Metal on one host.",
        "The lab ran no Liger Kernel code and no backward pass.",
        "PyTorch does not name the kernels of its sequences.",
        "The labs read no hardware counter.",
        "CI job `metal-liger-contracts` runs the fused kernel tests on Linux with "
        "CPU torch.",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  metal-liger-contracts:\n" in workflow
    codes = (
        metal_liger_cli.EXIT_OK,
        metal_liger_cli.EXIT_REJECTED,
        metal_liger_cli.EXIT_INVALID,
        metal_liger_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
    for flag in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "triton_gpu_kernel_executed",
        "liger_kernel_code_executed",
        "backward_pass_measured",
        "hardware_counters_used",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag


def test_the_day_80_row_keeps_the_fused_kernel_work_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 80 |")]
    _, code, lecture, manual = [cell.strip() for cell in row.strip("|").split("|")]

    for part in (
        "Three fused Metal forward kernels for RMSNorm, SwiGLU and cross-entropy",
        "one Apple GPU run that times each kernel beside its PyTorch sequence",
        "no CUDA, Triton GPU or Liger Kernel rate present",
    ):
        assert part in code, part
    assert "Lectures 29 and 28 pending user confirmation" in lecture
    assert "own check of the Liger Kernel facts against Lecture 28" in manual
    assert "learner-run fused kernels" in manual
    assert manual.endswith(" pending")
    assert (ROOT / GUIDE).is_file()
