import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import memory_hierarchy_cli
from ai_journey.memory_hierarchy_report import MEASUREMENT_BOUNDARY, MEASUREMENT_SCOPE

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/memory-hierarchy-lab.md"
COMMAND = "ai-journey-memory-hierarchy"
HEADING = "\n## Measure the CPU memory hierarchy\n"
PINNED = "config/day-71-memory-hierarchy.json"
SMOKE = "config/day-71-memory-hierarchy-ci-smoke.json"


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


def test_readme_states_what_the_cpu_run_measures_and_its_boundary():
    section = " ".join(readme_section().split())
    boundary = " ".join(MEASUREMENT_BOUNDARY["statement"].split())

    for phrase in (
        "[memory-hierarchy lab guide](docs/memory-hierarchy-lab.md)",
        "prints lines labeled `cpu_host_measurement`",
        f"`scope={MEASUREMENT_SCOPE}` and `gpu_executed=false`",
        "a stated, unmeasured 100 ns latency",
        "cannot re-observe a timing",
        "Exit code 0 means success, 1 a report that failed verification, and 2",
        f"`{SMOKE}`",
        "its timings are a smoke check, not a baseline",
    ):
        assert phrase in section, phrase
    for claim in (
        "Bandwidths count the operand bytes each NumPy call reads and writes, not "
        "hardware traffic, and include NumPy call overhead.",
        "No cache size is detected or claimed, and nothing here measures or "
        "predicts GPU, CUDA, distributed, or production performance.",
    ):
        assert claim in section and claim in boundary, claim
    for flag in ("gpu_executed", "latency_measured", "cache_sizes_detected"):
        assert MEASUREMENT_BOUNDARY[flag] is False, flag
    assert (
        memory_hierarchy_cli.EXIT_OK,
        memory_hierarchy_cli.EXIT_REJECTED,
        memory_hierarchy_cli.EXIT_INVALID,
    ) == (0, 1, 2)


def test_documented_commands_run_and_verify_through_the_packaged_cli(tmp_path, capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    scripts = project["project"]["scripts"]
    assert scripts[COMMAND] == "ai_journey.memory_hierarchy_cli:main"
    assert [argv[0] for argv in readme] == ["run", "verify"]
    assert documented_commands(GUIDE.read_text(encoding="utf-8")) == readme
    parser = memory_hierarchy_cli.build_parser()
    run, verify = (parser.parse_args(argv) for argv in readme)
    assert str(run.config) == str(verify.config) == PINNED
    assert verify.report == run.output

    # The pinned run measures 208 MiB of arrays for seconds, so the documented
    # commands operate the CI smoke config instead, as the CI job does.
    def localized(argv):
        return [
            str(ROOT / SMOKE)
            if word == PINNED
            else str(tmp_path / run.output.name)
            if word == str(run.output)
            else word
            for word in argv
        ]

    assert memory_hierarchy_cli.main(localized(readme[0])) == 0
    lines = capsys.readouterr().out.splitlines()
    labels = [line.split(" ", 1)[0] for line in lines[:-1]]
    assert labels == ["cpu_host_measurement"] * 9 + ["derived"] * 2
    assert re.fullmatch(
        r"memory_hierarchy_measurement=[0-9a-f]{64} stream_measurements=8 "
        rf"access_patterns=4 scope={MEASUREMENT_SCOPE} gpu_executed=false",
        lines[-1],
    ), lines[-1]
    assert memory_hierarchy_cli.main(localized(readme[1])) == 0
    assert capsys.readouterr().out == f"verified_{lines[-1]}\n"
