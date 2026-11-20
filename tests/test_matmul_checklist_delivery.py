"""The README section and the progress row of the Day 77 labs.

Each README command must be a guide command, and each README ratio must be
the ratio that the checklist guide quotes from the pinned run. Each other
README number must be a value that a guide states. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import (
    gpu_checklist_cli,
    matmul_checklist_bound_cli,
    matmul_checklist_cli,
)
from ai_journey.gpu_checklist_check import (
    BLOCKED_CHECK_NAMES,
    CHECK_SHAPES,
    COARSENED_CHECK_NAMES,
)
from ai_journey.gpu_checklist_cuda import PINNED_BLOCKED, PINNED_COARSENED
from ai_journey.matmul_checklist_bound import MEASUREMENT_BOUNDARY as BOUND_BOUNDARY
from ai_journey.matmul_checklist_config import load_config
from ai_journey.matmul_checklist_report import MEASUREMENT_BOUNDARY
from ai_journey.matmul_checklist_steps import checklist_mapping

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Apply a GPU performance checklist and record the speedup of each step\n"
CHECKLIST_GUIDE = "docs/matmul-checklist-lab.md"
BOUND_GUIDE = "docs/gpu-performance-checklist.md"
CUDA_GUIDE = "docs/gpu-checklist-cuda-contract.md"
PINNED = Path("config/day-77-matmul-checklist.json")
COMMAND = "ai-journey-matmul-checklist"
CHECKLIST = [["python", "-m", matmul_checklist_cli.__name__], [COMMAND]]
CUDA = [["python", "-m", gpu_checklist_cli.__name__]]
README_TABLE = (
    "| Step | Item | Kernel | Base kernel | 1024 x 1024 x 1024 | 2048 x 2048 x 2048 "
    "| 4096 x 4096 x 4096 | 2000 x 3000 x 1000 |"
)
PLAN_TABLE = "| Step | Item | Status | Base kernel | New kernel | Predicted verdicts |"
POINT_TABLE = (
    "| Step | Kernel | Median GFLOP/s (IQR range) | Over its base | Over S0 "
    "| Of `torch.matmul` | Hypothesis |"
)
# Each README figure, with the guide phrase that states the measured value.
FIGURES = {
    CHECKLIST_GUIDE: (
        (
            "Every hypothesis entered the repository before the first timed run of "
            "a Day 77 kernel.",
            "Every hypothesis entered the repository before the first timed run of "
            "a Day 77 kernel.",
        ),
        (
            "Each median comes from 11 samples.",
            "Each rate is the median GFLOP/s of 11 samples.",
        ),
        (
            "The steps form one chain, so each ratio measures one change on top of "
            "the changes before it.",
            "The steps form one chain, so each ratio measures one change on top of "
            "the changes before it.",
        ),
        (
            "The predictions held at 24 of 36 points.",
            "The predictions held at 24 of 36 points:",
        ),
        (
            "They failed most for S4 and for the 8 x 8 groups, at all four points "
            "each.",
            "They failed most for S4 and for the 8 x 8 groups, at all four points "
            "each.",
        ),
        (
            "S7 was the fastest kernel at each point. It reached 1312.0 to 1825.8 "
            "GFLOP/s, 3.70 to 6.16 times the S0 rate. That is 0.518 to 0.584 of the "
            "`torch.matmul` rate at the same point.",
            "S7 was the fastest kernel at each point. It reached 1312.0 to 1825.8 "
            "GFLOP/s, 3.70 to 6.16 times the S0 rate. That is 0.518 to 0.584 of the "
            "`torch.matmul` rate at the same point.",
        ),
        (
            "The report recorded load averages of 19.21, 21.17 and 23.01 before the "
            "run and 16.63, 20.39 and 22.64 after it.",
            "The report recorded load averages of 19.21, 21.17 and 23.01 before the "
            "run. It recorded 16.63, 20.39 and 22.64 after the run.",
        ),
        (
            "The `uptime` command printed 16.11, 20.65 and 22.85 before the lock.",
            "The `uptime` command printed 16.11, 20.65 and 22.85 before the lock,",
        ),
    ),
    BOUND_GUIDE: (
        (
            "The rule calls a ratio faster or slower only when it is 1.1 or more in "
            "that direction and the IQR ranges do not overlap.",
            "The ratio is 1.1 or more in that direction, and the IQR ranges of the "
            "two sides do not overlap.",
        ),
        ("The FMA ceiling was 4835.1 GFLOP/s.", "The FMA ceiling was 4835.1 GFLOP/s,"),
        (
            "The highest FMA share of a kernel was 0.378, for S7 at 4096 cubed. "
            "`torch.matmul` reached 0.465 to 0.728.",
            "The highest FMA share of a kernel was 0.378, for S7 at 4096 cubed. "
            "`torch.matmul` reached 0.465 to 0.728.",
        ),
        (
            "The rule named 16 of the 44 points `latency or occupancy-bound`, a "
            "residual name, and the other 28 points `indeterminate`.",
            "The rule named 16 of the 44 points `latency or occupancy-bound`. That "
            "name is a residual.",
        ),
        ("the other 28 points `indeterminate`.", "the other 28 points `indeterminate`"),
        (
            "The `uptime` command printed 17.24, 15.91 and 16.75 before the ceiling "
            "run.",
            "The `uptime` command printed 17.24, 15.91 and 16.75 before the ceiling "
            "run.",
        ),
    ),
    CUDA_GUIDE: (
        (
            "The CUDA check runs 11 checks on each coarsened text and 12 checks on "
            "each blocked text, at each of 18 Day 74 shapes.",
            "The coarsened kernel has 11 checks, and the blocked kernel has 12 "
            "checks",
        ),
        (
            "The CPU block simulator gives the replay bits in all 45 runs.",
            "All 45 runs hold the bits of `fma_replay` for their case.",
        ),
    ),
}


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def guide(name):
    return (ROOT / name).read_text(encoding="utf-8")


def flat(text):
    return " ".join(text.split())


def commands(text, starts):
    """The arguments of each bash line in ``text`` that begins with a start."""

    found = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            # Drop an environment assignment such as PYTHONPATH=src.
            while words and re.fullmatch(r"[A-Z_]+=\S*", words[0]):
                words = words[1:]
            found += [
                words[len(start) :] for start in starts if words[: len(start)] == start
            ]
    return found


def table(text, header, start=0):
    lines = text.splitlines()
    rows = []
    for line in lines[lines.index(header, start) + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def shape_names():
    return [
        f"{shape.m} x {shape.n} x {shape.k}"
        for shape in load_config(ROOT / PINNED).benchmark.shapes
    ]


def point_ratios():
    """The cell "Over its base" of each step kernel in each point table."""

    lines = guide(CHECKLIST_GUIDE).splitlines()
    ratios = {}
    for index, line in enumerate(lines):
        if line == POINT_TABLE:
            (heading,) = [
                text for text in lines[index - 2 : index] if text.startswith("### ")
            ]
            rows = table("\n".join(lines), POINT_TABLE, index)
            ratios[heading.removeprefix("### ")] = {
                row[1].strip("`"): row[3] for row in rows if row[3]
            }
    return ratios


def day_77_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 77 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_each_readme_command_is_a_guide_command():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section = readme_section()

    assert (
        project["project"]["scripts"][COMMAND] == "ai_journey.matmul_checklist_cli:main"
    )
    for starts, name in ((CHECKLIST, CHECKLIST_GUIDE), (CUDA, CUDA_GUIDE)):
        readme = commands(section, starts)
        steps = commands(guide(name), starts)
        assert readme and all(argv in steps for argv in readme), name
    # The README names the bound commands and leaves their order to the guide.
    bound = ["python", "-m", matmul_checklist_bound_cli.__name__]
    assert f"`{' '.join(bound)}`" in section
    assert not commands(section, [bound])
    assert commands(guide(BOUND_GUIDE), [bound])


def test_the_readme_checks_the_outputs_of_its_runs():
    section = readme_section()
    run, verify = (
        matmul_checklist_cli.build_parser().parse_args(argv)
        for argv in commands(section, CHECKLIST)
    )
    sources, check = (
        gpu_checklist_cli.build_parser().parse_args(argv)
        for argv in commands(section, CUDA)
    )

    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert verify.report == run.output
    assert (sources.command, check.command) == ("sources", "check")
    assert check.source_dir == sources.output_dir


def test_the_readme_table_holds_the_step_ratios_of_the_pinned_run():
    rows = table(readme_section(), README_TABLE)
    plan = table(guide(CHECKLIST_GUIDE), PLAN_TABLE)
    ratios = point_ratios()
    shapes = shape_names()
    declared = [
        (step["step"], step["base"], entry["variant"])
        for step in checklist_mapping()["steps"]
        for entry in step["predictions"]
    ]

    assert README_TABLE.strip("|").split("|")[4:] == [f" {name} " for name in shapes]
    assert list(ratios) == shapes
    assert rows == [
        [
            step,
            item,
            new,
            base,
            *(ratios[shape][new.strip("`")] for shape in shapes),
        ]
        for step, item, _, base, new, _ in plan
    ]
    assert [(row[0], row[3].strip("`"), row[2].strip("`")) for row in rows] == (
        declared
    )


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())

    for name, figures in FIGURES.items():
        text = flat(guide(name))
        for claim, measured in figures:
            assert claim in section, claim
            assert measured in text, (name, measured)
    config = load_config(ROOT / PINNED)
    assert config.benchmark.repeats == 11
    assert config.comparison.min_faster_ratio == 1.1
    assert (
        "All values in the table and in the three paragraphs after it come from "
        "host-specific Apple GPU (Metal) measurements on one shared host. They are "
        "not portable figures."
    ) in section


def test_each_pinned_cuda_text_passes_each_check_at_each_shape(tmp_path, capsys):
    assert gpu_checklist_cli.main(["sources", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert gpu_checklist_cli.main(["check", "--source-dir", str(tmp_path)]) == 0
    lines = {
        line.split()[0].removeprefix("cuda_check="): line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("cuda_check=")
    }

    assert len(CHECK_SHAPES) == 18
    for kernels, names in (
        (PINNED_COARSENED, COARSENED_CHECK_NAMES),
        (PINNED_BLOCKED, BLOCKED_CHECK_NAMES),
    ):
        total = len(names) * len(CHECK_SHAPES)
        for kernel in kernels:
            line = lines[f"{kernel.name}.cu"]
            assert f" pinned=true checks={total}/{total} shapes=18 " in line, line
    assert (len(COARSENED_CHECK_NAMES), len(BLOCKED_CHECK_NAMES)) == (11, 12)
    assert "Each pinned text passes each check." in flat(readme_section())


def test_readme_states_the_exit_codes_ci_jobs_and_boundary_of_the_labs():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "The bound command also exits 2 for reports that do not pair.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "The CUDA command needs no GPU, so it uses only the codes 0, 1 and 2.",
        "PyTorch does not name the kernel of `torch.matmul`",
        "No CUDA kernel ran, the labs used no NVIDIA GPU, and they did not measure "
        "cuBLAS.",
        "Nothing compiled or ran the CUDA texts.",
        "The labs read no hardware counter, so they did not measure occupancy.",
        "Each bottleneck name is an inference from timings and measured ceilings, "
        "not from hardware counters.",
        "CI job `matmul-checklist-contracts` runs the checklist and bound tests on "
        "Linux with CPU torch.",
        "CI job `nvidia-preflight-contracts` runs the CUDA contract tests.",
        f"({CHECKLIST_GUIDE})",
        f"({BOUND_GUIDE})",
        f"({CUDA_GUIDE})",
    ):
        assert phrase in section, phrase
    for job in ("matmul-checklist-contracts", "nvidia-preflight-contracts"):
        assert f"\n  {job}:\n" in workflow, job
    for cli in (matmul_checklist_cli, matmul_checklist_bound_cli):
        codes = (cli.EXIT_OK, cli.EXIT_REJECTED, cli.EXIT_INVALID, cli.EXIT_UNAVAILABLE)
        assert codes == (0, 1, 2, 3)
    assert (gpu_checklist_cli.EXIT_OK, gpu_checklist_cli.EXIT_REJECTED) == (0, 1)
    assert gpu_checklist_cli.EXIT_INVALID == 2
    assert not hasattr(gpu_checklist_cli, "EXIT_UNAVAILABLE")
    for flag in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "cublas_measured",
        "hardware_counters_used",
        "occupancy_measured",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag
        assert BOUND_BOUNDARY[flag] is False, flag
    assert MEASUREMENT_BOUNDARY["vendor_kernel_known"] is False
    assert BOUND_BOUNDARY["inference_from_timings"] is True


def test_the_day_77_row_keeps_the_lectures_and_learner_evidence_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8").splitlines()
    day, code, lecture, manual = day_77_row()
    rows = [line.split("|")[1].strip() for line in progress if line.startswith("| ")]

    assert rows[rows.index("76") + 1] == day == "77"
    for part in (
        "A declared performance checklist with one hypothesis for each step",
        "seven new Metal matmul kernels from one block template",
        "One Apple GPU run with the speedup of each step",
        "a bound from measured ceilings and a traffic model",
        "CUDA texts checked without a compiler",
        "an installed command",
        "CI gates present",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 8 and 2 pending user confirmation"
    assert manual.startswith("Learner-printed checklist")
    assert manual.endswith("an NVIDIA run of the CUDA texts pending")
    assert "Last updated: 2026-11-20" in progress
    for guide_name in (CHECKLIST_GUIDE, BOUND_GUIDE, CUDA_GUIDE):
        assert (ROOT / guide_name).is_file()
