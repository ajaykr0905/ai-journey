import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import matmul_model_cli
from ai_journey.matmul_model_report import EVIDENCE_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/matmul-model-lab.md"
HEADING = "\n## Model a naive matmul and compare it with measured Apple GPU ceilings\n"
COMMAND = "ai-journey-matmul-model"
MODULE_FORM = ["PYTHONPATH=src", "python", "-m", "ai_journey.matmul_model_cli"]
PINNED = "config/day-74-matmul-model.json"


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    """Each documented model-lab command as the arguments of the installed script."""

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


def test_readme_runs_the_guides_model_commands_as_the_installed_script():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.matmul_model_cli:main"
    assert readme == documented_commands(GUIDE.read_text(encoding="utf-8"))
    assert [argv[0] for argv in readme] == ["run", "verify"]
    run, verify = (matmul_model_cli.build_parser().parse_args(argv) for argv in readme)
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

    assert matmul_model_cli.main(localized(run)) == 0
    assert capsys.readouterr().out == f"{line}\n"
    assert matmul_model_cli.main(localized(verify)) == 0
    assert capsys.readouterr().out == f"verified_{line}\n"
    fields = dict(word.split("=", 1) for word in line.split())
    passed, checks = fields["cuda_checks"].split("/")
    assert fields["runs"] == fields["runs_matching_reference"]
    assert passed == checks
    section = flat(readme_section())
    for phrase in (
        f"all {fields['runs']} model runs match the reference bit for bit",
        f"all {checks} CUDA checks pass",
        f"report digest `{fields['matmul_model_report']}`",
    ):
        assert phrase in section, phrase


def test_readme_states_the_model_exit_codes_ci_job_and_boundary():
    section = flat(readme_section())

    for phrase in (
        "[matmul model lab guide](docs/matmul-model-lab.md)",
        "All five commands exit 0 on success and 1 when a report fails "
        "verification. They exit 2 for usage errors, unreadable or invalid inputs, "
        "or a refused output.",
        "the model lab runs no GPU and compiles no CUDA source.",
        "Its CUDA C text is source for a later run on an NVIDIA GPU.",
        "CI job `matmul-model-cpu` runs and verifies the pinned plan from an "
        "installed wheel without torch, NumPy or matplotlib.",
    ):
        assert phrase in section, phrase
    for flag in (
        "gpu_executed",
        "cuda_compiled",
        "nvidia_hardware_used",
        "timing_measured",
        "ceilings_measured",
    ):
        assert EVIDENCE_BOUNDARY[flag] is False, flag
    assert (
        matmul_model_cli.EXIT_OK,
        matmul_model_cli.EXIT_REJECTED,
        matmul_model_cli.EXIT_INVALID,
    ) == (0, 1, 2)
