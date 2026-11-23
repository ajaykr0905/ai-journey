"""The Triton IR half of the Day 80 README section and the progress row.

Each README command must be a guide command, and each README count must be a
count of the committed fixtures. The progress row must keep the lectures and
the learner's own reading of the PTX pending.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import triton_ir_cli
from ai_journey.triton_ir_report import BOUNDARIES, build_report

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Read the Triton compiler output and fuse LLM layers on the Apple GPU\n"
GUIDE = "docs/triton-ir-lab.md"
FIXTURES = Path("tests/fixtures/day80_triton_ir")
COMMAND = "ai-journey-triton-ir"
STARTS = [["python", "-m", triton_ir_cli.__name__], [COMMAND]]
FAMILY_TABLE = "| Family | Count |"


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
    found = []
    for line in lines[lines.index(header) + 2 :]:
        if not line.startswith("|"):
            break
        found.append(tuple(cell.strip() for cell in line.strip("|").split("|")))
    return found


def report():
    return build_report(
        ROOT / FIXTURES, clock=lambda: 0.0, load_average=lambda: (0.0, 0.0, 0.0)
    )


def day_80_row():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    (row,) = [line for line in progress.splitlines() if line.startswith("| 80 |")]
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_each_readme_command_is_a_guide_command_and_chains():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = [argv for argv in commands(readme_section()) if argv]
    parser = triton_ir_cli.build_parser()

    assert project["project"]["scripts"][COMMAND] == "ai_journey.triton_ir_cli:main"
    assert readme and all(argv in commands(guide()) for argv in readme)
    summarize, verify, mapping = (parser.parse_args(argv) for argv in readme)
    assert (summarize.command, verify.command, mapping.command) == (
        "summarize",
        "verify",
        "stage-map",
    )
    assert summarize.fixtures == verify.fixtures == mapping.fixtures == FIXTURES
    assert verify.report == summarize.output


def test_the_readme_counts_are_the_counts_of_the_committed_ptx():
    built = report()
    families = built["ptx"]["families"]
    loop = built["ptx"]["loops"][0]
    section = flat(readme_section())
    readme_rows = table(readme_section(), FAMILY_TABLE)
    guide_rows = set(table(guide(), FAMILY_TABLE))

    assert readme_rows and set(readme_rows) <= guide_rows
    names = {"`cp.async` copies": "cp.async"}
    for name, count in readme_rows:
        key = names.get(name, name.strip("`"))
        assert int(count) == families[key], name
    assert f"The entry holds {built['ptx']['instructions']} instructions." in section
    assert (
        f"The K loop starts on line {loop['first_line']} of `ptx.ptx` and ends on "
        f"line {loop['last_line']}. One pass holds {loop['instructions']} "
        "instructions in this order:"
    ) in section
    assert loop["families"]["fma.rn.f32"] == 1024
    assert f"{loop['families']['fma.rn.f32']} `fma.rn.f32` instructions" in section
    assert "`cp.async.wait_group 6`" in section
    assert built["pipeline"]["num_stages"] == 5
    assert built["pipeline"]["buffers_per_operand"] == 4
    assert built["ttgir"]["mma_encoding"] is False
    assert built["pipeline"]["mma_instructions"] == 0


def test_each_readme_statement_about_the_texts_is_in_the_guide():
    section = flat(readme_section())
    text = flat(guide())

    for claim, stated in (
        (
            'The kernel calls `tl.dot` with `input_precision="ieee"`, so Triton '
            "took the FMA path.",
            "Thus Triton took the FMA path.",
        ),
        (
            "In all 12 compiles, the shared bytes equal `(num_stages - 1)` times "
            "the tile bytes.",
            "The manifest shows the ring rule for all 12 compiles.",
        ),
        (
            "The TTIR is identical for `sm_80` and `sm_90` in all 6 configs.",
            "The TTIR is identical for both targets.",
        ),
        (
            "No `ptxas` report exists, so the lab states no register use and no "
            "occupancy.",
            "so the lab states no register use and no occupancy.",
        ),
    ):
        assert claim in section, claim
        assert stated in text, stated


def test_readme_states_the_exit_codes_ci_job_and_boundary_of_the_lab():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "Code 1 also marks a `check` whose regenerated text or metadata differs "
        "from the dump.",
        "Code 3 means that the host cannot compile with triton,",
        "the Triton IR and PTX are compiler output for the declared NVIDIA "
        "targets `sm_80` and `sm_90`. No PTX or cubin ran, and the labs used no "
        "NVIDIA GPU.",
        "CI job `triton-ir-contracts` runs the installed `check` on an x86_64 "
        "Linux runner and compares each stage with the aarch64 dump.",
        f"({GUIDE})",
    ):
        assert phrase in section, phrase
    assert "\n  triton-ir-contracts:\n" in workflow
    codes = (
        triton_ir_cli.EXIT_OK,
        triton_ir_cli.EXIT_REJECTED,
        triton_ir_cli.EXIT_INVALID,
        triton_ir_cli.EXIT_UNAVAILABLE,
    )
    assert codes == (0, 1, 2, 3)
    for boundary in BOUNDARIES.values():
        assert boundary["compiler_output_only"] is True
        for flag in ("nvidia_gpu_used", "ptx_executed", "cubin_executed"):
            assert boundary[flag] is False, flag


def test_the_day_80_row_keeps_the_lectures_and_learner_evidence_pending():
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8").splitlines()
    day, code, lecture, manual = day_80_row()
    days = [line.split("|")[1].strip() for line in progress if line.startswith("| ")]

    assert days[days.index("79") + 1] == day == "80"
    for part in (
        "The recorded TTIR, TTGIR and PTX of the Day 79 Triton matmul",
        "for the declared targets sm_80 and sm_90, read without triton",
        "one K loop pass read line by line",
        "installed commands, and CI gates present",
        "No PTX ran on a GPU",
    ):
        assert part in code, part
    assert lecture == "GPU MODE Lectures 29 and 28 pending user confirmation"
    assert manual.startswith("Learner's own reading of the committed PTX")
    assert manual.endswith("on an NVIDIA GPU pending")
    (updated,) = re.findall(
        r"^Last updated: (\d{4}-\d{2}-\d{2})$", "\n".join(progress), re.M
    )
    assert updated >= "2026-11-23"
    assert (ROOT / GUIDE).is_file()
