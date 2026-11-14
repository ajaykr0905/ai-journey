import re
import shlex
import tomllib
from pathlib import Path

from ai_journey import serving_recovery_cli
from ai_journey.serving_recovery_report import EVIDENCE_BOUNDARY

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/serving-failure-recovery.md"
COMMAND = "ai-journey-serving-recovery"
HEADING = "\n## Inject serving failures and measure recovery\n"


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


def test_readme_states_operation_measurements_and_evidence_boundary():
    section = " ".join(readme_section().split())
    boundary = " ".join(EVIDENCE_BOUNDARY.split())

    for phrase in (
        "[serving failure recovery guide](docs/serving-failure-recovery.md)",
        "real measurements from one monotonic clock",
        "declared lab controls, not external targets",
        "127.0.0.1 loopback",
        "The wall-clock run duration has no independent raw record, so `verify` "
        "only checks that it covers the traced lifecycle and stays within "
        "`max_run_ms`",
    ):
        assert phrase in section, phrase
    claim = "not GPU, CUDA, distributed, multi-host or production serving evidence"
    assert claim in section and claim in boundary
    assert "makes no model-quality, capacity or cost claim" in boundary
    assert "makes no model-quality, capacity or cost claim" in section
    assert (
        "Exit code 0 means verified and recovered, 1 rejected evidence, 2 invalid "
        "input or an incomplete run, and 3 verified but not recovered"
    ) in section
    assert (
        serving_recovery_cli.EXIT_RECOVERED,
        serving_recovery_cli.EXIT_VERIFICATION_FAILED,
        serving_recovery_cli.EXIT_INVALID,
        serving_recovery_cli.EXIT_NOT_RECOVERED,
    ) == (0, 1, 2, 3)


def test_documented_commands_parse_with_the_packaged_cli():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["scripts"][COMMAND]
    readme = documented_commands(readme_section())

    assert entry == "ai_journey.serving_recovery_cli:main"
    assert [argv[0] for argv in readme] == ["run", "verify"]
    assert documented_commands(GUIDE.read_text(encoding="utf-8")) == readme
    run, verify = (serving_recovery_cli._parser().parse_args(argv) for argv in readme)
    assert run.source == Path("data/day-19-demo-names.txt")
    assert run.config == Path("config/day-70-serving-recovery.json")
    assert (ROOT / run.source).is_file() and (ROOT / run.config).is_file()
    assert verify.run_dir == run.run_dir
