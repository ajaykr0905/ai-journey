import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import memory_access_cli
from ai_journey.memory_access_report import (
    ANALYSIS_BOUNDARY,
    MEASUREMENT_BOUNDARY,
    MEASUREMENT_SCOPE,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/memory-access-model.md"
COMMAND = "ai-journey-memory-access"
HEADING = "\n## Model GPU memory access and time CPU matmul baselines\n"
PLAN = "config/day-74-memory-access-plan.json"


def readme_section():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(HEADING, 1)[1].split("\n## ", 1)[0]


def documented_commands(text):
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words and words[0] == COMMAND:
                commands.append(words[1:])
    return commands


def test_readme_separates_analytical_results_from_cpu_measurements():
    section = " ".join(readme_section().split())

    for phrase in (
        "[memory-access model guide](docs/memory-access-model.md)",
        "`analyze` is analytical and prints `measured=false gpu_executed=false`",
        "`benchmark` is the only measurement",
        f"`{MEASUREMENT_SCOPE}`",
        "declared from NVIDIA's specifications table",
        "cannot re-observe a timing",
        "Exit code 0 means success, 1 a report that failed verification, and 2",
        "`config/day-74-memory-access-ci-smoke.json`",
    ):
        assert phrase in section, phrase
    analytical = "No GPU, CUDA kernel, cache, or device timing was executed or measured"
    measured = (
        "do not measure or predict GPU, CUDA, tensor-core, distributed, or "
        "production performance"
    )
    assert analytical in section and analytical in ANALYSIS_BOUNDARY["statement"]
    assert measured in section and measured in MEASUREMENT_BOUNDARY["statement"]
    assert ANALYSIS_BOUNDARY["gpu_executed"] is MEASUREMENT_BOUNDARY["gpu_executed"]
    assert MEASUREMENT_BOUNDARY["gpu_executed"] is False
    assert (
        memory_access_cli.EXIT_OK,
        memory_access_cli.EXIT_REJECTED,
        memory_access_cli.EXIT_INVALID,
    ) == (0, 1, 2)


def test_documented_commands_parse_and_the_analysis_verifies(tmp_path, capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.memory_access_cli:main"
    assert [argv[0] for argv in readme] == ["analyze", "verify", "benchmark", "verify"]
    assert documented_commands(GUIDE.read_text(encoding="utf-8")) == readme
    parsed = [memory_access_cli.build_parser().parse_args(argv) for argv in readme]
    assert {str(arguments.config) for arguments in parsed} == {PLAN}
    assert parsed[1].report == parsed[0].output
    assert parsed[3].report == parsed[2].output

    def localized(argv):
        outputs = {str(parsed[0].output), str(parsed[2].output)}
        return [
            str(ROOT / PLAN)
            if word == PLAN
            else str(tmp_path / Path(word).name)
            if word in outputs
            else word
            for word in argv
        ]

    assert memory_access_cli.main(localized(readme[0])) == 0
    analyzed = capsys.readouterr().out.strip()
    assert re.fullmatch(
        r"memory_access_analysis=[0-9a-f]{64} patterns=16 transposes=4 "
        r"tiled_estimates=9 measured=false gpu_executed=false",
        analyzed,
    )
    assert memory_access_cli.main(localized(readme[1])) == 0
    assert capsys.readouterr().out.strip() == f"verified_{analyzed}"
