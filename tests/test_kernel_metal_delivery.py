import itertools
import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import kernel_metal_cli
from ai_journey.kernel_metal_config import load_config
from ai_journey.kernel_metal_io import load_report
from ai_journey.kernel_metal_report import (
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_SCOPE,
    SYNTHETIC_SCOPE,
)
from test_kernel_metal_recorded import RecordedBackend
from test_kernel_metal_runtime import requires_metal

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/kernel-metal-lab.md"
HEADING = "\n## Simulate CUDA kernel launches and time Metal kernels on an Apple GPU\n"
COMMAND = "ai-journey-kernel-metal"
MODULE_FORM = ["python", "-m", "ai_journey.kernel_metal_cli"]
PINNED = "config/day-73-kernel-metal.json"


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def flat(text):
    return " ".join(text.split())


def documented_commands(text):
    """Each documented Metal-lab command as the arguments of the installed script."""

    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(MODULE_FORM)] == MODULE_FORM:
                words = [COMMAND, *words[len(MODULE_FORM) :]]
            if words and words[0] == COMMAND:
                commands.append(words[1:])
    return commands


def localized_commands(tmp_path):
    """The README commands with the pinned config and the output under tmp_path."""

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
    return localized(run), localized(verify)


def counts():
    work = load_config(ROOT / PINNED).planned_work()
    return work.correctness_cases, work.benchmark_points


def test_readme_runs_the_guides_metal_commands_as_the_installed_script():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.kernel_metal_cli:main"
    assert readme == documented_commands(GUIDE.read_text(encoding="utf-8"))
    assert [argv[0] for argv in readme] == ["run", "verify"]
    run, verify = (kernel_metal_cli.build_parser().parse_args(argv) for argv in readme)
    assert run.config == verify.config == Path(PINNED)
    assert verify.report == run.output


def test_documented_commands_operate_the_pinned_config_with_a_stand_in(
    tmp_path, capsys
):
    # A stand-in backend and clock let every host operate the documented
    # commands; the report then says that its timings are synthetic.
    run, verify = localized_commands(tmp_path)
    clock = itertools.count(0, 1_000).__next__
    correctness, benchmark = counts()

    assert kernel_metal_cli.main(run, backend=RecordedBackend(), clock=clock) == 0
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(Path(run[run.index("--output") + 1]))
    assert lines[-1] == (
        f"kernel_metal_synthetic_timing={report['report_sha256']} "
        f"correctness_cases={correctness} benchmark_points={benchmark} "
        f"scope={SYNTHETIC_SCOPE} measured=false cuda=false nvidia=false"
    )
    assert all(line.startswith("synthetic_timing ") for line in lines[:-1])
    assert kernel_metal_cli.main(verify) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


@requires_metal
def test_documented_commands_measure_and_verify_on_this_apple_gpu(tmp_path, capsys):
    run, verify = localized_commands(tmp_path)
    correctness, benchmark = counts()

    assert kernel_metal_cli.main(run) == 0
    lines = capsys.readouterr().out.splitlines()
    report, _ = load_report(Path(run[run.index("--output") + 1]))
    assert lines[-1] == (
        f"kernel_metal_measurement={report['report_sha256']} "
        f"correctness_cases={correctness} benchmark_points={benchmark} "
        f"scope={MEASUREMENT_SCOPE} cuda=false nvidia=false"
    )
    assert all(line.startswith("apple_gpu_metal_measurement ") for line in lines[:-1])
    # The README states that every case was exact in the recorded runs.
    summary = report["summary"]
    assert summary["rgb_to_gray_metal_mismatched_pixels"] == 0
    assert summary["rgb_to_gray_max_abs_difference_levels"] == 0
    assert summary["vector_add_exact_cases"] == len(report["correctness"]["vector_add"])
    assert kernel_metal_cli.main(verify) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_readme_states_the_metal_results_exit_codes_ci_job_and_boundary():
    section = flat(readme_section())
    guide = flat(GUIDE.read_text(encoding="utf-8"))
    boundary = flat(MEASUREMENT_BOUNDARY["statement"])
    correctness, _ = counts()

    for phrase in (
        "[Metal kernel lab guide](docs/kernel-metal-lab.md)",
        f"all {correctness} correctness cases were exact",
        "For `ai-journey-kernel-metal`, code 2 also marks a run that failed a check, "
        "had a device error or exceeded its time budget.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "CI job `kernel-metal-contracts` runs the Metal lab tests on Linux, where the "
        "GPU tests skip. It also requires the installed `run` to exit 3 and write no "
        "report.",
    ):
        assert phrase in section, phrase
    # The README repeats the guide's three-run figures and the report boundary.
    for figure in ("12.8 to 23.6 times faster", "between 0.85 and 1.24"):
        assert figure in section and figure in guide, figure
    for claim in (
        "host-specific Apple GPU (Metal) measurements",
        "do not measure or predict CUDA, NVIDIA, distributed or production "
        "performance.",
    ):
        assert claim in section and claim in boundary, claim
    for flag in (
        "cuda_kernel_executed",
        "nvidia_gpu_used",
        "portable_performance_claim",
    ):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag
    assert MEASUREMENT_BOUNDARY["host_specific"] is True
    assert (
        kernel_metal_cli.EXIT_OK,
        kernel_metal_cli.EXIT_REJECTED,
        kernel_metal_cli.EXIT_INVALID,
        kernel_metal_cli.EXIT_UNAVAILABLE,
    ) == (0, 1, 2, 3)
