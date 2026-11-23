"""The README section and the progress row of the Day 78 labs.

Each README command must be a guide command, and each cell of the four README
tables must be the value that a guide quotes from its pinned run. Each other
README number must be a value that a guide states. The progress row must keep
the lectures and the learner's own work pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import gpu_ceiling_cli, gpu_reduce_cli, gpu_reduce_cuda_cli
from ai_journey import gpu_scan_cli
from ai_journey.gpu_reduce_config import load_config as load_reduce_config
from ai_journey.gpu_reduce_cuda import PINNED_KERNELS
from ai_journey.gpu_reduce_cuda_check import CHECK_SIZES, check_names
from ai_journey.gpu_reduce_plan import PLAN as REDUCE_PLAN
from ai_journey.gpu_reduce_report import MEASUREMENT_BOUNDARY as REDUCE_BOUNDARY
from ai_journey.gpu_scan_config import load_config as load_scan_config
from ai_journey.gpu_scan_plan import PLAN as SCAN_PLAN
from ai_journey.gpu_scan_report import MEASUREMENT_BOUNDARY as SCAN_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Speed up a sum reduction step by step and compare three scans\n"
REDUCE_GUIDE = "docs/gpu-reduction-lab.md"
CUDA_GUIDE = "docs/gpu-reduce-cuda-contract.md"
SCAN_GUIDE = "docs/gpu-scan-lab.md"
REDUCE_PINNED = Path("config/day-78-gpu-reduce.json")
SCAN_PINNED = Path("config/day-78-gpu-scan.json")
# The words that begin each command: its python -m form and its console script.
CEILING = [["python", "-m", gpu_ceiling_cli.__name__], ["ai-journey-gpu-ceilings"]]
REDUCE = [["python", "-m", gpu_reduce_cli.__name__], ["ai-journey-gpu-reduce"]]
SCAN = [["python", "-m", gpu_scan_cli.__name__], ["ai-journey-gpu-scan"]]
CUDA = [["python", "-m", gpu_reduce_cuda_cli.__name__]]
# The README tables and the guide tables that hold their values.
REDUCE_RATES = "| n | R0 | R1 | R2 | R3 | R4 | R5 | `torch.sum` |"
REDUCE_STEPS = (
    "| Step | Kernel | Base | 65536 | 1048576 | 16777216 | 67108864 | 10000019 |"
)
REDUCE_PLAN_TABLE = "| Step | Kernel | Base | small | medium | large |"
REDUCE_POINT = (
    "| Step | Kernel | Median GB/s (IQR range) | Of the ceiling | Over its base "
    "| Over R0 | Of `torch.sum` | Hypothesis |"
)
SCAN_RATES = "| Length | S0 | S1 | S2 | `torch.cumsum` | `stream_copy` |"
# The plan table of the scan guide and the ratio table of the README share it.
SCAN_STEPS = "| Step | Kernel | Base | 2^16 | 2^20 | 2^24 | 5000003 |"
SCAN_POINT = (
    "| Side | Median GB/s (IQR range) | Over its base | Over S0 "
    "| Over `torch.cumsum` | Of the copy rate | Hypothesis |"
)
SCAN_SIDES = ("S0", "S1", "S2", "`torch.cumsum`", "`stream_copy`")
# Each README figure, with the guide phrases that state the measured value.
FIGURES = {
    REDUCE_GUIDE: (
        (
            "Each hypothesis of both labs entered the repository before the first "
            "timed run of its kernels.",
            (
                "Every hypothesis entered the repository before the first timed run "
                "of a Day 78 kernel.",
            ),
        ),
        (
            "Each value is the median GB/s of 14 samples, with 4 n bytes for each "
            "call.",
            (
                "the range from the IQR of its 14 samples.",
                "A rate is decimal GB/s: 4 n bytes for each call,",
            ),
        ),
        (
            "The steps form one chain, so each ratio measures one change on top of "
            "the changes before it.",
            (
                "The steps form one chain, so each ratio measures one change on top "
                "of the changes before it.",
            ),
        ),
        (
            "The predictions held at 23 of 30 points.",
            ("The predictions held at 23 of 30 points:",),
        ),
        (
            "At 2^16, R0 was slower than `torch.sum`, and R2, R3 and R4 were each "
            "faster than their bases. The plan said that the host dispatch sets the "
            "time there.",
            (
                "At 2^16, R0 was slower than `torch.sum`, and R2, R3 and R4 were each "
                "faster than their bases. The plan said that the host dispatch sets "
                "the time there.",
            ),
        ),
        (
            "The fastest kernel reached 2.891 to 14.707 times the R0 rate.",
            ("It reached 2.891 to 14.707 times the R0 rate.",),
        ),
        (
            "The ceiling of the fresh Day 74 run was 87.77 GB/s, from `stream_copy` "
            "at 256 MiB.",
            ("The ceiling is 87.77 GB/s, from `stream_copy` at 256 MiB.",),
        ),
        (
            "From 2^24, R2 to R5 reached 1.067 to 1.254 of the ceiling. The ceiling "
            "is the best median of the Day 74 kernels, not a hardware limit.",
            (
                "From 2^24, R2 to R5 reached 1.067 to 1.254 of the ceiling,",
                "The ceiling is the best median of those kernels, not a hardware "
                "limit.",
            ),
        ),
        (
            "At the ragged size of 10000019, `torch.sum` gave 18.7 GB/s. PyTorch "
            "does not report which kernel `torch.sum` ran, so the lab cannot tell "
            "why.",
            (
                "At the ragged size of 10000019, `torch.sum` gave 18.7 GB/s.",
                "PyTorch does not report which kernel `torch.sum` ran, so the lab "
                "cannot tell why.",
            ),
        ),
        (
            "All 78 outputs stayed within their bounds: 13 cases and 6 kernels. In "
            "the three exact cases, every kernel gave the exact sum. R3 kept the "
            "bits of R2 in all 13 cases and at all 5 sizes, as the plan predicted.",
            (
                "All 78 outputs stayed within their bounds: 13 cases and 6 kernels.",
                "In the three exact cases, every kernel gave the exact sum.",
                "R3 kept the bits of R2 in all 13 cases and at all 5 sizes, as the "
                "plan predicted.",
            ),
        ),
        (
            "The reduction report recorded load averages of 12.58, 11.47 and 12.76 "
            "before the run and 12.51, 11.51 and 12.74 after it. The `uptime` "
            "command printed 12.40, 11.38 and 12.76 before the ceiling run.",
            (
                "The reduction report recorded load averages of 12.58, 11.47 and "
                "12.76 before the run, and 12.51, 11.51 and 12.74 after the run.",
                "The `uptime` command printed 12.40, 11.38 and 12.76 before the "
                "ceiling run,",
            ),
        ),
    ),
    SCAN_GUIDE: (
        (
            "Each hypothesis of both labs entered the repository before the first "
            "timed run of its kernels. The scan plan uses one fact from an untimed "
            "correctness check, and the scan guide states it.",
            (
                "the repository received each hypothesis before the first timed run.",
                "Before the plan entered the repository, two untimed scripts ran "
                "scans on the development GPU.",
                "The plan uses one fact from the second script:",
            ),
        ),
        (
            "Each value is the median GB/s of 11 samples, with 8 bytes for each "
            "element.",
            (
                "Each rate is the median GB/s of 11 samples, with 8 bytes for each "
                "element.",
            ),
        ),
        (
            "S2 was the fastest kernel at each point. The predictions held at 5 of "
            "12 points.",
            (
                "S2 was the fastest kernel at each point. The predictions held at 5 "
                "of 12 points:",
            ),
        ),
        (
            "At 2^24 and at 5000003, S0 was 10.2 and 8.7 times as fast as "
            "`torch.cumsum`.",
            (
                "At 2^24 and at 5000003, S0 was 10.2 and 8.7 times as fast as "
                "`torch.cumsum`.",
            ),
        ),
        (
            "At 2^20, the IQR range of `torch.cumsum` was 11.58 to 92.55 GB/s, a "
            "factor of 8.",
            (
                "the IQR range of `torch.cumsum` was 11.58 to 92.55 GB/s,",
                "Their IQR range covers a factor of 8.",
            ),
        ),
        (
            "Each run of each kernel kept its sentinel tail and stayed within its "
            "bound. S0 and S1 held the bits of their CPU replays in each of the 8 "
            "normal cases.",
            (
                "Each run of each kernel kept its sentinel tail and stayed within its "
                "bound.",
                "S0 and S1 held the bits of their CPU replays in each of the 8 normal "
                "cases.",
            ),
        ),
        (
            "`torch.cumsum` gave other bits from run to run in the normal cases of "
            "65536, 1048577 and 5000003 elements.",
            (
                "`torch.cumsum` gave other bits from run to run in the normal cases "
                "of 65536, 1048577 and 5000003 elements.",
            ),
        ),
        (
            "The scan report recorded load averages of 10.46, 10.89 and 12.69 "
            "before the run and 11.51, 11.12 and 12.74 after it.",
            (
                "The report recorded load averages of 10.46, 10.89 and 12.69 before "
                "the run. It recorded 11.51, 11.12 and 12.74 after the run.",
            ),
        ),
    ),
    CUDA_GUIDE: (
        (
            "The CUDA check runs 13 checks on the R1 text and 14 checks on the "
            "shuffle text, at each of 14 sizes.",
            (
                "The R1 text has 13 checks, and the shuffle text has 14 checks:",
                "It checks each file for the 14 sizes",
            ),
        ),
        (
            "The simulator predicts 36 results, and each result is within its bound "
            "and has no race.",
            ("kernels=2 inputs=18 within_bounds=36/36 race_free=36/36",),
        ),
        (
            "At all 8 inputs of the recorded Metal report, each simulated result has "
            "the bits of its Metal twin.",
            (
                "At all 8 recorded inputs, `reduce_r1_shared` gives the bits of "
                "`reduce_r1_sequential`. At all 8 inputs, `reduce_warp_shuffle` gives "
                "the bits of `reduce_r2_first_add` and `reduce_r3_simd_shuffle`.",
            ),
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


def code(name):
    return f"`{name}`"


def length_label(n):
    return f"2^{n.bit_length() - 1}" if n & (n - 1) == 0 else str(n)


def reduce_points():
    """The rows of each size table of the reduction guide, by n and by step."""

    lines = guide(REDUCE_GUIDE).splitlines()
    points = {}
    for index, line in enumerate(lines):
        if line == REDUCE_POINT:
            (heading,) = [
                text for text in lines[index - 2 : index] if text.startswith("### ")
            ]
            n = re.fullmatch(r"### n = (\d+), class \w+", heading).group(1)
            rows = table("\n".join(lines), REDUCE_POINT, index)
            points[n] = {row[0]: row for row in rows}
    return points


def scan_points():
    """The rows of each sweep table of the scan guide, by length and by side."""

    lines = guide(SCAN_GUIDE).splitlines()
    points = {}
    for index, line in enumerate(lines):
        if line == SCAN_POINT:
            (label,) = [
                text.split(" elements, ", 1)[0]
                for text in lines[index - 2 : index]
                if " calls in each sample:" in text
            ]
            rows = table("\n".join(lines), SCAN_POINT, index)
            points[label] = {row[0].split(" ", 1)[0]: row for row in rows}
    return points


def median(cell):
    """The median of a cell such as ``31.93 (31.62 to 32.73)``."""

    value, spread = cell.split(" ", 1)
    assert re.fullmatch(r"\(\d+\.\d+ to \d+\.\d+\)", spread), cell
    return value


def day_78_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 78 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_each_readme_command_is_a_guide_command():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section = readme_section()

    for starts in (CEILING, REDUCE, SCAN):
        (_, _, module), (command,) = starts
        assert scripts["project"]["scripts"][command] == f"{module}:main"
    for starts, name in (
        (CEILING, REDUCE_GUIDE),
        (REDUCE, REDUCE_GUIDE),
        (SCAN, SCAN_GUIDE),
        (CUDA, CUDA_GUIDE),
    ):
        readme = commands(section, starts)
        steps = commands(guide(name), starts)
        assert readme and all(argv in steps for argv in readme), name
    # The Metal commands of the README use the installed console scripts.
    for starts in (CEILING, REDUCE, SCAN):
        assert not commands(section, starts[:1]), starts[0]


def test_the_readme_commands_chain_their_inputs_and_outputs():
    section = readme_section()
    (ceiling,) = commands(section, CEILING)
    made = gpu_ceiling_cli.build_parser().parse_args(ceiling)
    run, verify = (
        gpu_reduce_cli.build_parser().parse_args(argv)
        for argv in commands(section, REDUCE)
    )
    scan_run, scan_verify = (
        gpu_scan_cli.build_parser().parse_args(argv) for argv in commands(section, SCAN)
    )
    sources, check, simulate = (
        gpu_reduce_cuda_cli.build_parser().parse_args(argv)
        for argv in commands(section, CUDA)
    )

    assert (made.command, run.command, verify.command) == ("run", "run", "verify")
    assert run.ceilings == verify.ceilings == made.output
    assert verify.report == run.output
    assert run.config == verify.config == simulate.config == REDUCE_PINNED
    assert (scan_run.command, scan_verify.command) == ("run", "verify")
    assert scan_verify.report == scan_run.output
    assert scan_run.config == scan_verify.config == SCAN_PINNED
    assert (sources.command, check.command, simulate.command) == (
        "sources",
        "check",
        "simulate",
    )
    assert check.source_dir == sources.output_dir
    for path in (made.config, REDUCE_PINNED, SCAN_PINNED):
        assert (ROOT / path).is_file(), path


def test_the_reduction_tables_hold_the_pinned_run_of_the_guide():
    section = readme_section()
    sizes = [str(n) for n in load_reduce_config(ROOT / REDUCE_PINNED).benchmark.sizes]
    points = reduce_points()
    plan = table(guide(REDUCE_GUIDE), REDUCE_PLAN_TABLE)

    assert table(section, REDUCE_RATES) == table(guide(REDUCE_GUIDE), REDUCE_RATES)
    assert [row[0] for row in table(section, REDUCE_RATES)] == sizes
    assert REDUCE_STEPS.strip("|").split("|")[3:] == [f" {n} " for n in sizes]
    assert list(points) == sizes
    assert table(section, REDUCE_STEPS) == [
        [step, kernel, base, *(points[n][step][4] for n in sizes)]
        for step, kernel, base, *_ in plan
    ]
    assert [row[:2] for row in plan] == [
        [entry.step, code(entry.kernel)] for entry in REDUCE_PLAN
    ]


def test_the_scan_tables_hold_the_pinned_run_of_the_guide():
    section = readme_section()
    lengths = [
        length_label(n) for n in load_scan_config(ROOT / SCAN_PINNED).benchmark.lengths
    ]
    points = scan_points()
    plan = table(guide(SCAN_GUIDE), SCAN_STEPS)

    assert SCAN_STEPS.strip("|").split("|")[3:] == [f" {n} " for n in lengths]
    assert list(points) == lengths
    assert table(section, SCAN_RATES) == [
        [label, *(median(points[label][side][1]) for side in SCAN_SIDES)]
        for label in lengths
    ]
    assert table(section, SCAN_STEPS) == [
        [step, kernel, base, *(points[label][step][2] for label in lengths)]
        for step, kernel, base, *_ in plan
    ]
    assert [row[:2] for row in plan] == [
        [entry.step, code(entry.kernel)] for entry in SCAN_PLAN
    ]
    assert [points[lengths[0]][entry.step][0] for entry in SCAN_PLAN] == [
        f"{entry.step} {code(entry.kernel)}" for entry in SCAN_PLAN
    ]


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())

    for name, figures in FIGURES.items():
        text = flat(guide(name))
        for claim, measured in figures:
            assert claim in section, claim
            for phrase in measured:
                assert phrase in text, (name, phrase)
    reduce = load_reduce_config(ROOT / REDUCE_PINNED)
    scan = load_scan_config(ROOT / SCAN_PINNED)
    assert (reduce.benchmark.repeats, scan.benchmark.repeats) == (14, 11)
    assert reduce.comparison.min_faster_ratio == scan.comparison.min_faster_ratio
    assert scan.comparison.min_faster_ratio == 1.1
    assert (
        "The four tables and the paragraphs between the first table and this one "
        "give host-specific Apple GPU (Metal) measurements. They come from one "
        "shared host, and they are not portable figures. One run gives each ratio, "
        "so another run can give another verdict."
    ) in section


def test_each_pinned_cuda_text_passes_each_check_at_each_size(tmp_path, capsys):
    assert gpu_reduce_cuda_cli.main(["sources", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert gpu_reduce_cuda_cli.main(["check", "--source-dir", str(tmp_path)]) == 0
    lines = {
        line.split()[0].removeprefix("cuda_check="): line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("cuda_check=")
    }

    assert len(CHECK_SIZES) == 14
    assert [len(check_names(kernel)) for kernel in PINNED_KERNELS] == [13, 14]
    for kernel in PINNED_KERNELS:
        total = len(check_names(kernel)) * len(CHECK_SIZES)
        line = lines[f"{kernel.name}.cu"]
        assert f" pinned=true checks={total}/{total} sizes=14 " in line, line
    assert "Each pinned text passes each check." in flat(readme_section())


def test_readme_states_the_exit_codes_ci_jobs_and_boundary_of_the_labs():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "Ajay must still watch both lectures and check the labs against them.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "The CUDA command needs no GPU, so it uses only the codes 0, 1 and 2.",
        "SIMD-group functions are Metal's counterpart of CUDA warp shuffles, not "
        "the same instruction.",
        "PyTorch does not name the kernels of `torch.sum` and `torch.cumsum`, so "
        "each vendor number names the call, not a kernel.",
        "No CUDA kernel ran, and the labs used no NVIDIA GPU.",
        "Nothing compiled or ran the CUDA texts.",
        "The labs read no hardware counter, so each bound name is an inference from "
        "timings and measured rates.",
        "CI job `gpu-reduce-scan-contracts` runs the reduction, CUDA and scan tests "
        "on Linux with CPU torch.",
        "It also requires each installed Apple GPU command to exit 3 and write no "
        "report.",
        "The reduction report must also match the recorded Day 74 ceiling.",
        "CI job `nvidia-preflight-contracts` runs the same tests.",
        f"({REDUCE_GUIDE})",
        f"({CUDA_GUIDE})",
        f"({SCAN_GUIDE})",
    ):
        assert phrase in section, phrase
    for job in ("gpu-reduce-scan-contracts", "nvidia-preflight-contracts"):
        assert f"\n  {job}:\n" in workflow, job
    for cli in (gpu_reduce_cli, gpu_scan_cli):
        codes = (cli.EXIT_OK, cli.EXIT_REJECTED, cli.EXIT_INVALID, cli.EXIT_UNAVAILABLE)
        assert codes == (0, 1, 2, 3)
    cuda = gpu_reduce_cuda_cli
    assert (cuda.EXIT_OK, cuda.EXIT_REJECTED, cuda.EXIT_INVALID) == (0, 1, 2)
    assert not hasattr(cuda, "EXIT_UNAVAILABLE")
    for boundary, warp in (
        (REDUCE_BOUNDARY, "warp_level_claim"),
        (SCAN_BOUNDARY, "warp_shuffle_executed"),
    ):
        for flag in (
            "cuda_kernel_executed",
            "nvidia_gpu_used",
            warp,
            "vendor_kernel_known",
            "hardware_counters_used",
            "occupancy_measured",
        ):
            assert boundary[flag] is False, flag
        assert boundary["apple_gpu_metal_this_host"] is True


def test_the_day_78_row_keeps_the_lectures_and_learner_evidence_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8").splitlines()
    day, code_status, lecture, manual = day_78_row()
    rows = [line.split("|")[1].strip() for line in progress if line.startswith("| ")]
    (updated,) = [line for line in progress if line.startswith("Last updated: ")]

    assert rows[rows.index("77") + 1] == day == "78"
    for part in (
        "Six float32 Metal sum reductions, R0 to R5",
        "three float32 Metal inclusive scans",
        "each with one hypothesis committed before timing",
        "One Apple GPU run of each lab",
        "CUDA texts of the R1 shared-memory and warp-shuffle reductions checked "
        "without a compiler",
        "installed commands",
        "CI gates present",
    ):
        assert part in code_status, part
    assert lecture == "GPU MODE Lectures 9 and 20 pending user confirmation"
    assert manual.startswith("Learner-written reduction and scan kernels")
    assert manual.endswith("an NVIDIA run of the CUDA texts pending")
    # A later day updates the file again, so the date may only move forward.
    assert updated.removeprefix("Last updated: ") >= "2026-11-21"
    for guide_name in (REDUCE_GUIDE, CUDA_GUIDE, SCAN_GUIDE):
        assert (ROOT / guide_name).is_file()
