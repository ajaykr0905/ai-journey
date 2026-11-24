"""The Metal half of the Day 81 README section and the progress row.

Each README command must be a guide command, and each README cell must be the
value that the guide quotes from the pinned run. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import metal_flash_cli
from ai_journey.metal_flash_config import load_config
from ai_journey.metal_flash_report import MEASUREMENT_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = (
    "\n## Check flash attention in the Triton interpreter and time it on the "
    "Apple GPU\n"
)
GUIDE = "docs/metal-flash-attention-lab.md"
PINNED = Path("config/day-81-metal-flash.json")
CEILINGS = Path("/tmp/day-74-gpu-ceilings.json")
COMMAND = "ai-journey-metal-flash"
STARTS = [["python", "-m", metal_flash_cli.__name__], [COMMAND]]
README_TIMES = (
    "| Shape | Mode | N0 ms | F1 ms | PyTorch ms | F1 over N0 | F1 over PyTorch |"
)
GUIDE_TIMES = (
    "| Shape | Mode | N0 ms | F1 ms | PyTorch ms | GFLOP/s N0 / F1 / PyTorch "
    "| F1 of FMA | F1 over N0 | F1 over PyTorch | N0 over PyTorch |"
)
README_MEMORY = (
    "| Shape | N0 allocated | N0 driver peak | F1 allocated | F1 driver peak "
    "| PyTorch allocated | PyTorch driver peak |"
)
GUIDE_MEMORY = (
    "| Shape | N0 allocated | N0 driver peak | N0 by construction | F1 allocated "
    "| F1 driver peak | PyTorch allocated | PyTorch driver peak |"
)
# Each README figure, with the guide phrase that states the measured value.
FIGURES = (
    (
        "The predictions held at 30 of 30 step results.",
        "The predictions held at 30 of 30 step results.",
    ),
    (
        "F1 was faster than N0 at all 10 points, and slower than the PyTorch call "
        "at all 10 points.",
        "The verdicts were `faster` for F1 over N0 at all 10 points. They were "
        "`slower` for N0 and for F1 over the PyTorch call at all 10 points.",
    ),
    (
        "F1 ran at about 0.2 of the FMA ceiling of 4665.5 GFLOP/s from a fresh "
        "Day 74 run.",
        "about 0.2 of the FMA ceiling.",
    ),
    ("The FMA ceiling is 4665.5 GFLOP/s.",) * 2,
    (
        "The driver count grows in heaps of about 1 GiB.",
        "The driver count grew in heaps of about 1 GiB,",
    ),
    (
        "For the PyTorch call, no counter showed an N x N buffer. This result "
        "does not prove that no such buffer existed,",
        "For the PyTorch call, no counter showed an N x N buffer.",
    ),
    (
        "The Day 81 report recorded load averages of 8.87, 8.36 and 11.07 before "
        "the run. It recorded 11.94, 9.21 and 11.24 after the run.",
        "The Day 81 report recorded load averages of 8.87, 8.36 and 11.07 before "
        "the run. It recorded 11.94, 9.21 and 11.24 after the run.",
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

    assert project["project"]["scripts"][COMMAND] == "ai_journey.metal_flash_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    run, verify = (metal_flash_cli.build_parser().parse_args(argv) for argv in readme)
    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert run.ceilings == verify.ceilings == CEILINGS
    assert verify.report == run.output


def test_the_readme_timing_table_holds_the_cells_of_the_pinned_run():
    readme = rows(readme_section(), README_TIMES)
    pinned = rows(guide(), GUIDE_TIMES)
    config = load_config(ROOT / PINNED)

    points = len(config.benchmark.shapes) * len(config.benchmark.modes)
    assert len(readme) == len(pinned) == points == 10
    for mine, measured in zip(readme, pinned):
        assert mine == measured[:5] + measured[7:9]
    assert config.comparison.min_faster_ratio == 1.1


def test_the_readme_memory_table_holds_the_cells_of_the_pinned_run():
    readme = rows(readme_section(), README_MEMORY)
    pinned = rows(guide(), GUIDE_MEMORY)

    assert len(readme) == len(pinned) == 5
    for mine, measured in zip(readme, pinned):
        assert mine == measured[:3] + measured[4:]


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())
    text = flat(guide())

    for mine, measured in FIGURES:
        assert mine in section or mine in text, mine
        assert measured in text, measured
    assert "Each point holds 32768 tokens with a head dimension of 64." in section


def test_readme_states_the_ci_job_and_boundary_of_the_metal_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "or that it cannot run Metal kernels through PyTorch.",
        "The Metal times and memory readings come from one Apple GPU through "
        "Metal on one host.",
        "The labs used no NVIDIA GPU and ran no CUDA kernel.",
        "The labs read no hardware counter.",
        "All values in the two Metal tables come from host-specific Apple GPU "
        "(Metal) measurements on one shared host.",
        "CI job `metal-flash-contracts` runs the Metal tests on Linux with CPU "
        "torch.",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  metal-flash-contracts:\n" in workflow
    codes = (
        metal_flash_cli.EXIT_OK,
        metal_flash_cli.EXIT_REJECTED,
        metal_flash_cli.EXIT_INVALID,
        metal_flash_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
    for flag in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "triton_gpu_kernel_executed",
        "backward_pass_measured",
        "hardware_counters_used",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag


def test_the_day_81_row_keeps_the_metal_work_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 81 |")]
    _, code, lecture, manual = [cell.strip() for cell in row.strip("|").split("|")]

    for part in (
        "A naive and a flash float32 Metal attention kernel",
        "Apple GPU run that checks, measures the memory of and times each "
        "kernel beside torch.nn.functional.scaled_dot_product_attention present",
        "Sealed reports, independent verifiers, installed commands, and CI gates "
        "present",
    ):
        assert part in code, part
    assert "Lectures 12 and 36 pending user confirmation" in lecture
    assert "Own check of the guide facts against Lecture 12" in manual
    assert manual.endswith(" pending")
