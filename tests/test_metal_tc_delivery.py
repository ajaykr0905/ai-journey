"""The Metal half of the Day 82 README section and the progress row.

Each README command must be a guide command, and each README cell must be the
value that the guide quotes from the pinned run. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import metal_tc_cli
from ai_journey.metal_tc_config import load_config
from ai_journey.metal_tc_report import MEASUREMENT_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Compile for Tensor Cores in CI and time the Apple GPU matrix path\n"
GUIDE = "docs/metal-matrix-precision-lab.md"
PINNED = Path("config/day-82-metal-tc.json")
CEILINGS = Path("/tmp/day-74-gpu-ceilings.json")
COMMAND = "ai-journey-metal-tc"
STARTS = [["python", "-m", metal_tc_cli.__name__], [COMMAND]]
# The guide holds two tables with this header: the times, then the rates.
TIMES = "| Side | 512 | 1024 | 2048 | 4096 |"
SIDES = [
    "S1",
    "M1",
    "M2",
    "M3",
    "PyTorch float32",
    "PyTorch float16",
    "PyTorch bfloat16",
]
# Each README figure, with the guide phrase that states the measured value.
FIGURES = (
    ("M1 over S1 is 1.44, 1.45, 1.41 and 1.25.",) * 2,
    ("Its ratios are 1.46, 1.50, 1.22 and 1.03.", "It is 1.46, 1.50, 1.22 and 1.03."),
    (
        "M3 over M1 is 1.04 or less at each size, so the gain comes from the matrix "
        "path, not from bfloat.",
    )
    * 2,
    (
        "The float16 and bfloat16 PyTorch calls ran 1.09 to 1.14 times as fast as "
        "the float32 call.",
    )
    * 2,
    ("The predictions held at 31 of 36 step results.",) * 2,
    ("The FMA ceiling is 4665.5 GFLOP/s,", "The FMA ceiling is 4665.5 GFLOP/s."),
    (
        "the bandwidth ceiling is 107.71 GB/s.",
        "The bandwidth ceiling is 107.71 GB/s.",
    ),
    (
        "The Day 82 report recorded load averages of 7.09, 5.19 and 5.35 before "
        "the run. It recorded 5.86, 5.02 and 5.29 after the run.",
    )
    * 2,
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

    assert project["project"]["scripts"][COMMAND] == "ai_journey.metal_tc_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    run, verify = (metal_tc_cli.build_parser().parse_args(argv) for argv in readme)
    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert run.ceilings == verify.ceilings == CEILINGS
    assert verify.report == run.output


def test_the_readme_timing_table_holds_the_cells_of_the_pinned_run():
    readme = rows(readme_section(), TIMES)
    pinned = rows(guide(), TIMES)
    config = load_config(ROOT / PINNED)

    assert readme == pinned
    assert [row[0] for row in readme] == SIDES
    assert len(config.benchmark.sizes) == 4
    for row in readme:
        assert all(re.fullmatch(r"\d+\.\d{3} \(\d+\.\d{3}\)", cell) for cell in row[1:])
    assert "Each time is the median time of one call in milliseconds" in flat(
        readme_section()
    )


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())
    text = flat(guide())

    for mine, measured in FIGURES:
        assert mine in section, mine
        assert measured in text, measured


def test_readme_states_the_ci_job_and_boundary_of_the_metal_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "or that it cannot run Metal kernels through PyTorch.",
        "The Metal numbers come from one Apple GPU through Metal on one host.",
        "The labs used no NVIDIA GPU and ran no CUDA kernel.",
        "The labs read no hardware counter.",
        "All values in the Metal table come from host-specific Apple GPU (Metal) "
        "measurements on one shared host.",
        "These bf16 over fp32 ratios are not Tensor Core ratios, because Apple "
        "GPUs have no NVIDIA Tensor Cores.",
        "CI job `metal-tc-contracts` runs the Metal tests on Linux with CPU torch.",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  metal-tc-contracts:\n" in workflow
    codes = (
        metal_tc_cli.EXIT_OK,
        metal_tc_cli.EXIT_REJECTED,
        metal_tc_cli.EXIT_INVALID,
        metal_tc_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
    for flag in (
        "tensor_cores_used",
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "triton_gpu_kernel_executed",
        "hardware_counters_used",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag


def test_the_day_82_row_keeps_the_metal_work_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 82 |")]
    _, code, lecture, manual = [cell.strip() for cell in row.strip("|").split("|")]

    for part in (
        "A scalar and three matrix-path Metal matmul kernels in float32, half and "
        "bfloat",
        "One Apple GPU run that checks and times each kernel beside torch.matmul in "
        "three dtypes present.",
        "Sealed reports, independent verifiers, installed commands, and CI gates "
        "present.",
    ):
        assert part in code, part
    assert "Lectures 23 and 15 pending user confirmation" in lecture
    assert (
        "Own check of the Tensor Core and CUTLASS facts against Lectures 23" in manual
    )
    assert manual.endswith(" pending")
