"""The Triton half of the Day 79 README section.

Each README command must be a guide command, and each row of the README line
count table must be the count of a real text. Each other README figure of the
Triton check must be a value that the guide and the recorded report state.
"""

import json
import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import triton_matmul_cli
from ai_journey.triton_matmul_loc import counted_texts
from ai_journey.triton_matmul_report import INTERPRETER_BOUNDARY
from test_triton_matmul_recorded import RECORDED_REPORT_JSON

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Port the matmul to Triton and fuse a softmax on the Apple GPU\n"
GUIDE = "docs/triton-matmul-lab.md"
PINNED = Path("config/day-79-triton-matmul.json")
COMMAND = "ai-journey-triton-matmul"
STARTS = [["python", "-m", triton_matmul_cli.__name__], [COMMAND]]
LOC_TABLE = (
    "| Lab | Language | Text | Kernel lines | Host lines | Other | Comment | Blank |"
)
LABS = {"day-79": "Day 79", "day-76": "Day 76", "day-77": "Day 77"}
LANGUAGES = {
    "Triton (Python)": "Triton",
    "CUDA C++": "CUDA C++",
    "Metal Shading Language": "Metal",
}


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


def table(text, header):
    lines = text.splitlines()
    rows = []
    for line in lines[lines.index(header) + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_each_readme_command_is_a_guide_command_and_chains():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.triton_matmul_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    loc, check, verify = (
        triton_matmul_cli.build_parser().parse_args(argv) for argv in readme
    )
    assert (loc.command, check.command, verify.command) == ("loc", "check", "verify")
    assert check.config == verify.config == PINNED
    assert verify.report == check.output


def test_each_readme_line_count_row_is_the_count_of_a_real_text():
    rows = table(readme_section(), LOC_TABLE)
    counted = {}
    for item in counted_texts():
        count = item.count()
        host = "none" if count.host is None else count.host
        counted[item.name] = (
            f"| {LABS[item.lab]} | {LANGUAGES[item.language]} | `{item.name}` | "
            f"{count.kernel} | {host} | {count.other} | {count.comment} | "
            f"{count.blank} |"
        )

    assert len(rows) == 6
    assert table(guide(), LOC_TABLE)[0] == rows[0]
    for row in rows:
        name = row.split("|")[3].strip().strip("`")
        assert row == counted[name], row
        assert row in table(guide(), LOC_TABLE), row


def test_each_readme_figure_of_the_check_is_the_recorded_run():
    section = flat(readme_section())
    text = flat(guide())
    report = json.loads(RECORDED_REPORT_JSON)

    for claim, measured in (
        (
            "Nine of the 33 Triton kernel lines are the grouped order, which no CUDA "
            "text has.",
            "The Triton kernel has 33 code lines. Nine of them are the grouped order, "
            "which no CUDA text has.",
        ),
        (
            "One Triton text serves all six configs.",
            "One Triton text serves all six configs.",
        ),
        (
            "It did 108 runs: 9 cases, 6 configs and 2 families. All 54 normal runs "
            "kept the Day 74 bound, and all 54 exact runs held the exact bits.",
            "It did 108 runs: 9 cases, 6 configs and 2 families. All 54 normal runs "
            "kept the bound, and all 54 exact runs held the exact bits.",
        ),
        (
            "The largest share of the bound was 0.516, at 1 x 1 x 1.",
            "| 0 | 1 x 1 x 1 | 5.96e-08 | 3.07e-08 | 0.516 |",
        ),
    ):
        assert claim in section, claim
        assert measured in text, measured
    summary = report["summary"]
    assert (summary["cases"], summary["configs"], summary["runs"]) == (9, 6, 108)
    assert summary["normal_runs_within_bound"] == 54
    assert summary["exact_runs_with_exact_bits"] == 54
    assert f"{summary['max_bound_ratio']:.3f}" == "0.516"
    assert (summary["triton_kernel_lines"], summary["triton_host_lines"]) == (33, 9)
    assert "These runs are CPU correctness checks, not performance numbers." in (
        section
    )


def test_the_readme_keeps_the_triton_boundary():
    section = flat(readme_section())

    for phrase in (
        "The lab gives no Triton GPU rate, no CUDA rate and no cuBLAS rate, because "
        "this host has no NVIDIA or AMD GPU.",
        "no Triton kernel ran on a GPU, and the labs tuned no config.",
        "Code 3 means that the host cannot run the Triton CPU interpreter",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    for flag in (
        "gpu_executed",
        "triton_compiled_for_gpu",
        "autotuned",
        "cuda_compiled",
        "cublas_called",
        "nvidia_or_amd_gpu_used",
    ):
        assert INTERPRETER_BOUNDARY[flag] is False, flag
    codes = (
        triton_matmul_cli.EXIT_OK,
        triton_matmul_cli.EXIT_REJECTED,
        triton_matmul_cli.EXIT_INVALID,
        triton_matmul_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
