import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import kernel_launch_cli
from ai_journey.kernel_launch_report import EVIDENCE_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/kernel-launch-lab.md"
HEADING = "\n## Simulate CUDA kernel launches and time Metal kernels on an Apple GPU\n"
COMMAND = "ai-journey-kernel-launch"
MODULE_FORM = ["PYTHONPATH=src", "python", "-m", "ai_journey.kernel_launch_cli"]
PINNED = "config/day-73-kernel-launch.json"


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def documented_commands(text):
    """Each documented launch-lab command as the arguments of the installed script."""

    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(MODULE_FORM)] == MODULE_FORM:
                words = [COMMAND, *words[len(MODULE_FORM) :]]
            if words and words[0] == COMMAND:
                commands.append(words[1:])
    return commands


def pinned_line():
    """The summary line that the guide states for the pinned plan."""

    guide = GUIDE.read_text(encoding="utf-8")
    (line,) = re.findall(r"```text\n(.*?)\n```", guide, flags=re.DOTALL)
    return line


def test_readme_runs_the_guides_launch_commands_as_the_installed_script():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.kernel_launch_cli:main"
    assert readme == documented_commands(GUIDE.read_text(encoding="utf-8"))
    assert [argv[0] for argv in readme] == ["run", "verify"]
    run, verify = (kernel_launch_cli.build_parser().parse_args(argv) for argv in readme)
    assert run.config == verify.config == Path(PINNED)
    assert verify.report == run.output


def test_documented_commands_print_the_pinned_line_the_readme_states(tmp_path, capsys):
    def localized(argv):
        return [
            str(ROOT / PINNED)
            if word == PINNED
            else str(tmp_path / Path(word).name)
            if word.startswith("/tmp/")
            else word
            for word in argv
        ]

    run, verify = documented_commands(readme_section())
    line = pinned_line()

    assert kernel_launch_cli.main(localized(run)) == 0
    assert capsys.readouterr().out == f"{line}\n"
    assert kernel_launch_cli.main(localized(verify)) == 0
    assert capsys.readouterr().out == f"verified_{line}\n"
    fields = dict(word.split("=", 1) for word in line.split())
    passed, checks = fields["cuda_checks"].split("/")
    assert fields["runs"] == fields["runs_matching_reference"]
    assert passed == checks
    section = " ".join(readme_section().split())
    for phrase in (
        f"all {fields['runs']} runs of the guarded kernels match exact references",
        f"the unguarded kernels make {fields['out_of_bounds_writes_caught']} "
        f"out-of-bounds writes and {fields['write_conflicts_caught']} write conflicts",
        f"{checks} structural checks pass on it without a compiler",
        f"report digest `{fields['kernel_launch_report']}`",
    ):
        assert phrase in section, phrase


def test_readme_states_the_launch_exit_codes_ci_job_and_boundary():
    section = " ".join(readme_section().split())

    for phrase in (
        "[kernel-launch lab guide](docs/kernel-launch-lab.md)",
        "`ai-journey-kernel-launch` and `ai-journey-kernel-metal` exit 0 on success, "
        "1 when a report fails verification, and 2 for usage errors, unreadable or "
        "invalid inputs, or a refused output",
        "the kernel-launch lab runs no GPU, compiles no CUDA source and measures no "
        "time",
        "Its CUDA C++ text is source for a later run on an NVIDIA GPU.",
        "CI job `kernel-launch-cpu` runs and verifies the pinned plan from an "
        "installed wheel without torch, NumPy or matplotlib.",
    ):
        assert phrase in section, phrase
    for flag in (
        "gpu_executed",
        "cuda_compiled",
        "nvidia_hardware_used",
        "timing_measured",
    ):
        assert EVIDENCE_BOUNDARY[flag] is False, flag
    assert (
        "The CUDA text is source for a later NVIDIA run."
        in (EVIDENCE_BOUNDARY["statement"])
    )
    assert (
        kernel_launch_cli.EXIT_OK,
        kernel_launch_cli.EXIT_REJECTED,
        kernel_launch_cli.EXIT_INVALID,
    ) == (0, 1, 2)
