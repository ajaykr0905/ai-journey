import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import (
    gpu_ceiling_cli,
    matmul_metal_cli,
    matmul_roofline_cli,
    metal_launch_sweep_cli,
)
from ai_journey.gpu_ceiling_report import MEASUREMENT_BOUNDARY as CEILING_BOUNDARY
from ai_journey.matmul_metal_report import MEASUREMENT_BOUNDARY as MATMUL_BOUNDARY
from ai_journey.matmul_roofline_gap import MEASUREMENT_BOUNDARY
from test_matmul_roofline_gap import write_sources
from test_matmul_roofline_recorded import RECORDED_GAP

ROOT = Path(__file__).resolve().parents[1]
HEADING = "\n## Model a naive matmul and compare it with measured Apple GPU ceilings\n"
# Each installed command, its CLI module and the guide that documents it.
COMMANDS = {
    "ai-journey-matmul-metal": (matmul_metal_cli, "docs/matmul-metal-lab.md"),
    "ai-journey-gpu-ceilings": (gpu_ceiling_cli, "docs/gpu-ceilings-lab.md"),
    "ai-journey-metal-launch-sweep": (
        metal_launch_sweep_cli,
        "docs/metal-launch-sweep.md",
    ),
    "ai-journey-matmul-roofline": (matmul_roofline_cli, "docs/matmul-roofline-gap.md"),
}
NAIVE = ("matmul_x_columns", "matmul_x_rows")


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
        entry = project["project"]["scripts"][command]
        assert entry == f"{module.__name__}:main", command
        assert module.PROG == command
        readme = documented(readme_section(), command)
        steps = documented((ROOT / guide).read_text(encoding="utf-8"), command)
        assert all(argv in steps for argv in readme), command
        assert f"`{command}`" in readme_section(), command
    assert [
        argv[0] for argv in documented(readme_section(), "ai-journey-matmul-roofline")
    ] == [
        "gap",
        "verify",
    ]


def test_the_readme_gap_reads_the_outputs_of_the_readme_runs():
    (metal_run,) = parsed("ai-journey-matmul-metal")
    (ceiling_run,) = parsed("ai-journey-gpu-ceilings")
    gap, verify = parsed("ai-journey-matmul-roofline")

    assert (metal_run.command, ceiling_run.command) == ("run", "run")
    assert gap.matmul_report == verify.matmul_report == metal_run.output
    assert gap.ceiling_report == verify.ceiling_report == ceiling_run.output
    assert verify.report == gap.output


def test_the_readme_gap_commands_operate_on_recorded_reports(tmp_path, capsys):
    matmul, ceilings = write_sources(tmp_path)
    (metal_run,) = parsed("ai-journey-matmul-metal")
    (ceiling_run,) = parsed("ai-journey-gpu-ceilings")
    local = {
        str(metal_run.output): str(matmul),
        str(ceiling_run.output): str(ceilings),
    }
    gap, verify = (
        [
            local.get(word, str(tmp_path / Path(word).name))
            if word.startswith("/tmp/")
            else word
            for word in argv
        ]
        for argv in documented(readme_section(), "ai-journey-matmul-roofline")
    )

    assert matmul_roofline_cli.main(gap) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("matmul_roofline_measurement=")
    assert "sources_checked=true" in lines[-1]
    assert matmul_roofline_cli.main(verify) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"


def test_readme_states_the_measured_gap_of_the_recorded_reports():
    section = flat(readme_section())
    summary = RECORDED_GAP["summary"]
    points = RECORDED_GAP["points"]
    best = summary["best_fma_ceiling_share"]
    columns = points[best["matmul_x_columns"]["index"]]["sides"]["matmul_x_columns"]
    torch = points[best["torch_matmul"]["index"]]["sides"]["torch_matmul"]
    no_reuse = summary["no_reuse_roof_share_range"]
    gap_guide = flat((ROOT / "docs/matmul-roofline-gap.md").read_text(encoding="utf-8"))

    for phrase in (
        'In this section, "peak" means a ceiling that the gpu-ceilings lab '
        "measured on this host.",
        "the naive `matmul_x_columns` kernel reached at most "
        f"{columns['median_gflops_per_second']:.0f} GFLOP/s.",
        f"That rate is {columns['fma_ceiling_share']:.3f} of the measured FMA "
        f"ceiling of {RECORDED_GAP['ceilings']['fma_gflops_per_second']:.0f} GFLOP/s.",
        f"`torch.matmul` reached {torch['median_gflops_per_second']:.0f} GFLOP/s, "
        f"{torch['fma_ceiling_share']:.3f} of the same ceiling.",
        f"At all {summary['points']} sizes, both naive kernels ran "
        f"{no_reuse['matmul_x_rows'][0]:.2f} to {no_reuse['matmul_x_columns'][1]:.2f} "
        "times above the no-reuse bandwidth roof.",
        "The 1-minute load averages were 7.98 to 10.72 before and after the two "
        "runs.",
    ):
        assert phrase in section, phrase
    assert summary["points_above_no_reuse_roof"] == {side: 9 for side in NAIVE}
    # The README range runs from the lowest share of either kernel to the highest.
    assert (
        min(no_reuse["matmul_x_rows"][0], no_reuse["matmul_x_columns"][0])
        == (no_reuse["matmul_x_rows"][0])
    )
    assert (
        max(no_reuse["matmul_x_rows"][1], no_reuse["matmul_x_columns"][1])
        == (no_reuse["matmul_x_columns"][1])
    )
    # The gap guide states each load average that the README sums up.
    for phrase in (
        "The `uptime` load averages were 10.72, 9.61 and 8.81 before the ceiling run.",
        "The `uptime` load averages were 7.98, 9.06 and 8.98 before the matmul-metal "
        "run and 8.24, 8.97 and 8.95 after it.",
    ):
        assert phrase in gap_guide, phrase


def test_readme_states_the_exit_codes_ci_job_and_boundary_of_the_metal_labs():
    section = flat(readme_section())

    for phrase in (
        "The gap command also exits 2 for two reports that do not pair.",
        "For the three Metal commands, code 2 also marks a run that failed a check, "
        "had a device error or exceeded its time budget.",
        "Code 3 means that the host cannot run Metal kernels through PyTorch.",
        "CI job `metal-roofline-contracts` runs the tests of the other four labs on "
        "Linux, where the GPU tests skip. It also requires each installed Metal "
        "`run` to exit 3 and write no report.",
        "[matmul-metal lab guide](docs/matmul-metal-lab.md)",
        "[gpu-ceilings lab guide](docs/gpu-ceilings-lab.md)",
        "[launch-sweep guide](docs/metal-launch-sweep.md)",
        "[gap guide](docs/matmul-roofline-gap.md)",
    ):
        assert phrase in section, phrase
    for claim, boundary in (
        ("host-specific Apple GPU (Metal) measurements", MATMUL_BOUNDARY),
        (
            "do not measure or predict CUDA, NVIDIA, distributed or production "
            "performance.",
            CEILING_BOUNDARY,
        ),
        ("not a vendor peak", MEASUREMENT_BOUNDARY),
    ):
        assert claim in section and claim in flat(boundary["statement"]), claim
    for module in (matmul_metal_cli, gpu_ceiling_cli, metal_launch_sweep_cli):
        assert (module.EXIT_OK, module.EXIT_REJECTED, module.EXIT_INVALID) == (0, 1, 2)
        assert module.EXIT_UNAVAILABLE == 3
    assert (
        matmul_roofline_cli.EXIT_OK,
        matmul_roofline_cli.EXIT_REJECTED,
        matmul_roofline_cli.EXIT_INVALID,
    ) == (0, 1, 2)
