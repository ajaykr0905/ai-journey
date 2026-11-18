"""The README section of the four Day 75 labs, tied to the guides and the code.

Each README command must be a guide command of an installed script, the
bottleneck command must read the reports that the README runs write, and
each figure must appear in the guide that measured it.
"""

import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import (
    gpu_timing_cli,
    matmul_bottleneck_cli,
    profile_trace_cli,
    sim_profile_cli,
)
from ai_journey.matmul_bottleneck_report import MEASUREMENT_BOUNDARY
from ai_journey.profile_trace_report import MEASURED_BOUNDARY as TRACE_BOUNDARY
from test_matmul_bottleneck_gpu_timing import gap_timing_report, write_gap_inputs

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Profile the naive matmul and name its top bottleneck\n"
# Each installed command, its CLI module and the guide that documents it.
COMMANDS = {
    "ai-journey-sim-profile": (sim_profile_cli, "docs/simulator-profiling.md"),
    "ai-journey-profile-trace": (profile_trace_cli, "docs/profile-trace-lab.md"),
    "ai-journey-gpu-timing": (gpu_timing_cli, "docs/gpu-timing-lab.md"),
    "ai-journey-matmul-bottleneck": (
        matmul_bottleneck_cli,
        "docs/matmul-bottleneck-lab.md",
    ),
}
# Each README figure, with the guide phrase that states the measured value.
FIGURES = {
    "docs/profile-trace-lab.md": (
        (
            "It had a share of 0.685 of the iteration host time.",
            "879.4 ms of iteration host time, a share of 0.685.",
        ),
        (
            "The load averages were 11.00, 12.79 and 12.49 before that capture.",
            "The load averages were 11.00, 12.79 and 12.49 before the capture.",
        ),
    ),
    "docs/gpu-timing-lab.md": (
        ("median of 64.9 us and 69.6 us in two runs.", "| 1 | 64.9 us |"),
        ("median of 64.9 us and 69.6 us in two runs.", "| 2 | 69.6 us |"),
        (
            "`torch.matmul` reached 0.834 and 0.850 of the 4618 GFLOP/s FMA ceiling",
            "| `torch.matmul` | 3850.2, 3926.7 | 0.834, 0.850 |",
        ),
        (
            "The load averages were 9.87, 13.62 and 12.65 before run 1.",
            "| 1 | 08:32:09 to 08:33:27 | 9.87, 13.62, 12.65 |",
        ),
    ),
    "docs/matmul-bottleneck-lab.md": (
        (
            "`latency or occupancy-bound` as the top name of each naive kernel, at "
            "4 of 6 shapes.",
            "The top name of each kernel was `latency or occupancy-bound`, at 4 of 6 "
            "shapes.",
        ),
        (
            "At 128 cubed, both kernels were `dispatch-bound`.",
            "| 128 cubed | `dispatch-bound` (0.880) | `dispatch-bound` (0.817) |",
        ),
        (
            "The load averages were 8.90, 8.80 and 10.84 before the joined run.",
            "The `uptime` load averages were 8.90, 8.80 and 10.84 before the ceiling "
            "run",
        ),
    ),
    "docs/simulator-profiling.md": (
        (
            "the only repeatable gain, 1.29 to 1.40 in each run.",
            "Round 3 gave the only large and repeatable gain: 1.29 to 1.40 in each "
            "run.",
        ),
        ("overall speedups of 2.079 and 1.508", "| All | | 2.079 | 1.508 |"),
        (
            "at a 1-minute load of 14.7 to 18.4.",
            "The 1-minute load was 17.2 to 18.4 in session 2 and 14.7 to 17.4 in "
            "session 1.",
        ),
    ),
}


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def flat(text):
    return " ".join(text.split())


def documented(text, command):
    """The commands of one lab in ``text`` as the arguments of its installed script."""

    module = ["python", "-m", COMMANDS[command][0].__name__]
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(module)] == module:
                words = [command, *words[len(module) :]]
            if words and words[0] == command:
                commands.append(words[1:])
    return commands


def parsed(command):
    module = COMMANDS[command][0]
    return [
        module.build_parser().parse_args(argv)
        for argv in documented(readme_section(), command)
    ]


def test_each_readme_command_is_a_guide_command_of_an_installed_script():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    for command, (module, guide) in COMMANDS.items():
        assert project["project"]["scripts"][command] == f"{module.__name__}:main"
        assert module.PROG == command
        readme = documented(readme_section(), command)
        steps = documented((ROOT / guide).read_text(encoding="utf-8"), command)
        assert readme and all(argv in steps for argv in readme), command
        assert f"`{command}`" in readme_section(), command


def test_the_readme_verifies_and_classifies_the_outputs_of_its_runs():
    capture, trace_verify = parsed("ai-journey-profile-trace")
    timing_run, timing_verify = parsed("ai-journey-gpu-timing")
    (classify,) = parsed("ai-journey-matmul-bottleneck")
    (sim_run,) = parsed("ai-journey-sim-profile")

    assert (capture.command, trace_verify.command) == ("capture", "verify")
    assert trace_verify.report.parent == capture.output_dir
    assert trace_verify.config == capture.config
    assert timing_verify.report == timing_run.output
    assert timing_verify.config == timing_run.config
    assert classify.command == "classify"
    assert classify.timing == timing_run.output
    assert sim_run.stage == "round-4"


def test_the_readme_classify_command_joins_a_gpu_timing_report(tmp_path, capsys):
    report = gap_timing_report()
    inputs = write_gap_inputs(tmp_path, report)
    local = {
        "--matmul-report": str(inputs.matmul_report),
        "--ceiling-report": str(inputs.ceiling_report),
        "--gap-report": str(inputs.gap_report),
        "--timing": str(inputs.timing),
        "--output": str(tmp_path / "bottleneck.json"),
        "--config": str(ROOT / "config/day-75-matmul-bottleneck.json"),
    }
    (argv,) = documented(readme_section(), "ai-journey-matmul-bottleneck")
    argv = [
        local[argv[index - 1]] if index and argv[index - 1] in local else word
        for index, word in enumerate(argv)
    ]

    assert matmul_bottleneck_cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("matmul_bottleneck_synthetic_timing=")
    assert "sources_checked=true" in lines[-1]


def test_each_readme_figure_is_the_value_its_guide_measured():
    section = flat(readme_section())

    for guide, figures in FIGURES.items():
        text = flat((ROOT / guide).read_text(encoding="utf-8"))
        for claim, measured in figures:
            assert claim in section, claim
            assert measured in text, (guide, measured)
    assert (
        "All values in these three paragraphs are host-specific Apple GPU (Metal) "
        "measurements from one shared host, not portable figures."
    ) in section


def test_readme_states_the_exit_codes_ci_jobs_and_boundary_of_the_labs():
    section = flat(readme_section())
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for phrase in (
        "The bottleneck command also exits 2 for inputs that do not pair.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "each bottleneck name is an inference from timings",
        "The labs read no hardware counter and used no Nsight and no Instruments.",
        "do not measure or predict CUDA, NVIDIA, distributed or production "
        "performance.",
        "CI job `profile-sim-cpu` runs the trace and sim-profile tests without torch.",
        "CI job `gpu-timing-contracts` runs the timing and bottleneck tests on Linux "
        "with CPU torch.",
    ):
        assert phrase in section, phrase
    for job in ("profile-sim-cpu", "gpu-timing-contracts"):
        assert f"\n  {job}:\n" in workflow, job
    for module in COMMANDS.values():
        cli = module[0]
        assert (cli.EXIT_OK, cli.EXIT_REJECTED, cli.EXIT_INVALID) == (0, 1, 2)
        assert f"({module[1]})" in section
    for cli in (profile_trace_cli, gpu_timing_cli):
        assert cli.EXIT_UNAVAILABLE == 3
    assert MEASUREMENT_BOUNDARY["inference_from_timings"] is True
    for flag in ("hardware_counters_used", "nsight_used", "instruments_used"):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag
    assert TRACE_BOUNDARY["hardware_counters_read"] is False
    assert "CUDA, NVIDIA, distributed or production performance" in flat(
        TRACE_BOUNDARY["statement"]
    )
