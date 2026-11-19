"""The README section and the progress row of the Day 76 labs.

Each README command must be a guide command, each README number must be the
value that a guide measured, and the progress row must keep the lectures and
the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import gpu_tiled_cli, matmul_tiled_cli, matmul_tiled_gap_cli
from ai_journey.matmul_tiled_comparison import CUBLAS
from ai_journey.matmul_tiled_config import load_config
from ai_journey.matmul_tiled_gap import MEASUREMENT_BOUNDARY as GAP_BOUNDARY
from ai_journey.matmul_tiled_report import MEASUREMENT_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Tile the matmul and compare the naive, tiled and vendor rates\n"
TILED_GUIDE = "docs/matmul-tiled-lab.md"
CUDA_GUIDE = "docs/gpu-tiled-cuda-contract.md"
PINNED = Path("config/day-76-matmul-tiled.json")
COMMAND = "ai-journey-matmul-tiled"
TILED = [["python", "-m", matmul_tiled_cli.__name__], [COMMAND]]
CUDA = [["python", "-m", gpu_tiled_cli.__name__]]
README_TABLE = "| M x N x K | Naive | Tiled | Vendor | cuBLAS |"
GUIDE_TABLE = "| M x N x K | Naive | Tiled (tile 32 at each point) | Vendor | cuBLAS |"
# Each README figure, with the guide phrase that states the measured value.
FIGURES = {
    TILED_GUIDE: (
        (
            "Each value is the median GFLOP/s of 11 samples",
            "Each cell gives the median GFLOP/s of 11 samples.",
        ),
        (
            "Tile 32 was slower at 4 shapes",
            "The best tile, 32, was slower at 4 points.",
        ),
        (
            "the rule found no clear difference at 4096 x 4096 x 4096.",
            "At 4096 x 4096 x 4096, the rule found no clear difference.",
        ),
        (
            "`torch.matmul` was 6.7 to 11.1 times as fast as the best tile.",
            "`torch.matmul` was 6.7 to 11.1 times as fast as the best tile.",
        ),
        (
            "The report recorded load averages of 9.93, 11.62 and 14.06 before the "
            "run and 9.70, 11.40 and 13.90 after it.",
            "The report recorded load averages of 9.93, 11.62 and 14.06 before the "
            "run. It recorded 9.70, 11.40 and 13.90 after the run.",
        ),
        (
            "The gap run gave `latency or occupancy-bound` as the top name of the "
            "naive kernel and of each tile.",
            "The top name of the naive kernel and of each tile was `latency or "
            "occupancy-bound`.",
        ),
        (
            "Each kernel used at most 0.082 of the measured FMA ceiling",
            "Each kernel used at most 0.082 of the FMA ceiling.",
        ),
        ("ceiling of 4836.8 GFLOP/s.", "The FMA ceiling was 4836.8 GFLOP/s"),
        (
            "The `uptime` command printed 9.81, 10.52 and 13.03 before the ceiling "
            "run.",
            "The `uptime` command printed 9.81, 10.52 and 13.03 before the ceiling "
            "run.",
        ),
    ),
    CUDA_GUIDE: (
        ("all 297 CUDA checks", "cuda_checks=297/297"),
        ("all 63 cuBLAS checks pass.", "cublas_checks=63/63"),
        ("gives the replay bits in all 27 runs.", "replay_matches=27/27"),
        (
            "`2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a30158f06ffae`",
            "gpu_tiled_report=2419f0bfedcffd5eaa363e07640023e4e81a4bcfc752a415fd8a"
            "30158f06ffae",
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


def table(text, header):
    lines = text.splitlines()
    rows = []
    for line in lines[lines.index(header) + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def day_76_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 76 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_each_readme_command_is_a_guide_command():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section = readme_section()

    assert project["project"]["scripts"][COMMAND] == "ai_journey.matmul_tiled_cli:main"
    for starts, name in ((TILED, TILED_GUIDE), (CUDA, CUDA_GUIDE)):
        readme = commands(section, starts)
        steps = commands(guide(name), starts)
        assert readme and all(argv in steps for argv in readme), name
    # The README names the gap commands and leaves their order to the guide.
    assert f"`python -m {matmul_tiled_gap_cli.__name__}`" in section
    assert not commands(section, [["python", "-m", matmul_tiled_gap_cli.__name__]])


def test_the_readme_verifies_the_outputs_of_its_runs():
    section = readme_section()
    run, verify = (
        matmul_tiled_cli.build_parser().parse_args(argv)
        for argv in commands(section, TILED)
    )
    cuda_run, cuda_verify = (
        gpu_tiled_cli.build_parser().parse_args(argv)
        for argv in commands(section, CUDA)
    )

    assert (run.command, verify.command) == ("run", "verify")
    assert run.config == verify.config == PINNED
    assert verify.report == run.output
    assert (cuda_run.command, cuda_verify.command) == ("run", "verify")
    assert cuda_run.config is None and cuda_verify.config is None
    assert cuda_verify.report == cuda_run.output


def test_the_readme_table_holds_the_guide_medians_of_the_pinned_shapes():
    rows = table(readme_section(), README_TABLE)
    measured = [
        [re.sub(r" \(\S+ to \S+\)$", "", cell) for cell in row]
        for row in table(guide(TILED_GUIDE), GUIDE_TABLE)
    ]
    shapes = load_config(ROOT / PINNED).benchmark.shapes

    assert rows == measured
    assert [row[0] for row in rows] == [
        f"{shape.m} x {shape.n} x {shape.k}" for shape in shapes
    ]
    assert {row[-1] for row in rows} == {"not measured"}
    assert CUBLAS.startswith("not measured")


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())

    for name, figures in FIGURES.items():
        text = flat(guide(name))
        for claim, measured in figures:
            assert claim in section, claim
            assert measured in text, (name, measured)
    assert load_config(ROOT / PINNED).benchmark.repeats == 11
    assert (
        "All values in the table and in these two paragraphs are host-specific "
        "Apple GPU (Metal) measurements from one shared host, not portable figures."
    ) in section


def test_readme_states_the_exit_codes_ci_jobs_and_boundary_of_the_labs():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "The gap command also exits 2 for reports that do not pair.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "The CUDA command needs no GPU, so it uses only the codes 0, 1 and 2.",
        "PyTorch does not name the kernel of `torch.matmul`",
        "No CUDA kernel ran, the labs used no NVIDIA GPU, and they did not measure "
        "cuBLAS.",
        "Nothing compiled or ran the CUDA and cuBLAS texts.",
        "Each bottleneck name is an inference from timings and measured ceilings, "
        "not from hardware counters.",
        "CI job `matmul-tiled-contracts` runs the tiled and gap tests on Linux with "
        "CPU torch.",
        "CI job `nvidia-preflight-contracts` runs the CUDA contract tests.",
        f"({TILED_GUIDE})",
        f"({CUDA_GUIDE})",
    ):
        assert phrase in section, phrase
    for job in ("matmul-tiled-contracts", "nvidia-preflight-contracts"):
        assert f"\n  {job}:\n" in workflow, job
    for cli in (matmul_tiled_cli, matmul_tiled_gap_cli):
        codes = (cli.EXIT_OK, cli.EXIT_REJECTED, cli.EXIT_INVALID, cli.EXIT_UNAVAILABLE)
        assert codes == (0, 1, 2, 3)
    assert (gpu_tiled_cli.EXIT_OK, gpu_tiled_cli.EXIT_REJECTED) == (0, 1)
    assert gpu_tiled_cli.EXIT_INVALID == 2
    assert not hasattr(gpu_tiled_cli, "EXIT_UNAVAILABLE")
    for flag in ("cuda_kernel_executed", "nvidia_gpu_used", "cublas_measured"):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag
        assert GAP_BOUNDARY[flag] is False, flag
    assert MEASUREMENT_BOUNDARY["vendor_kernel_known"] is False
    assert GAP_BOUNDARY["inference_from_timings"] is True
    assert GAP_BOUNDARY["hardware_counters_used"] is False


def test_the_day_76_row_keeps_the_lectures_and_learner_evidence_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8").splitlines()
    day, code, lecture, manual = day_76_row()
    rows = [line.split("|")[1].strip() for line in progress if line.startswith("| ")]

    assert rows[rows.index("75") + 1] == day == "76"
    for part in (
        "threadgroup memory",
        "torch.matmul numbers from one Apple GPU run",
        "gap to the measured ceilings",
        "cuBLAS SGEMM texts checked without a compiler",
        "an installed command",
        "CI gates present",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 5 and 16 pending user confirmation"
    assert manual.startswith("Learner-run tiled kernel")
    assert manual.endswith("run on an NVIDIA GPU pending")
    for guide_name in (TILED_GUIDE, CUDA_GUIDE):
        assert (ROOT / guide_name).is_file()
