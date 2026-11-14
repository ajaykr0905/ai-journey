import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import simt_cli
from ai_journey.simt_report import SIMT_EVIDENCE_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/simt-execution-model.md"
COMMAND = "ai-journey-simt"
HEADING = "\n## Model SIMT execution limits analytically\n"
COUNTS = "devices=3 sweep_cells=40 memory_rows=18 arithmetic_rows=12"


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


def test_readme_labels_the_report_as_analytical_with_cited_inputs():
    section = " ".join(readme_section().split())

    for phrase in (
        "[SIMT execution model guide](docs/simt-execution-model.md)",
        "Everything in the report is analytical",
        "declared inputs",
        "CUDA C++ Programming Guide v12.6",
        "cuda_occupancy.h",
        "`measured=false gpu_executed=false`",
        "return 0 on success and 1 for an invalid plan, report or path",
    ):
        assert phrase in section, phrase
    for count in COUNTS.split():
        assert f"`{count}`" in section
    claim = "No GPU, CUDA kernel, or latency was executed or measured"
    assert claim in section and claim in SIMT_EVIDENCE_BOUNDARY["statement"]
    assert SIMT_EVIDENCE_BOUNDARY["gpu_executed"] is False
    assert (simt_cli.EXIT_OK, simt_cli.EXIT_INVALID) == (0, 1)


def test_documented_commands_build_and_verify_with_the_packaged_cli(tmp_path, capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = documented_commands(readme_section())

    assert project["project"]["scripts"][COMMAND] == "ai_journey.simt_cli:main"
    assert [argv[0] for argv in readme] == ["build", "verify"]
    assert documented_commands(GUIDE.read_text(encoding="utf-8")) == readme
    build, verify = readme
    report = build[build.index("--output") + 1]
    assert verify[verify.index("--report") + 1] == report

    def localized(argv):
        local = str(tmp_path / Path(report).name)
        return [local if word == report else word for word in argv]

    config = build[build.index("--config") + 1]
    build = localized(build)
    build[build.index("--config") + 1] = str(ROOT / config)
    assert simt_cli.main(build) == 0
    built = capsys.readouterr().out.strip()
    assert re.fullmatch(
        rf"simt_report=[0-9a-f]{{64}} {COUNTS} measured=false gpu_executed=false",
        built,
    )
    assert simt_cli.main(localized(verify)) == 0
    assert capsys.readouterr().out.strip() == f"verified_{built}"
