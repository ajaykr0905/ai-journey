"""The installed command of the Day 80 Triton IR lab.

Only ``check`` needs triton, and it imports triton when it makes its
backend. Triton has no macOS wheel, so the entry point must load, summarize
and verify on a host without triton or torch, and ``check`` must stop there
with exit code 3 before it writes anything.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import triton_ir_cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-triton-ir"
GUIDE = ROOT / "docs/triton-ir-lab.md"
FIXTURES = ROOT / "tests" / "fixtures" / "day80_triton_ir"
# A None entry in sys.modules makes every import of that name fail.
BLOCKED = ("triton", "torch", "matplotlib")
WITHOUT = """
import importlib
import sys
for name in sys.argv[1].split(","):
    sys.modules[name] = None
module, attribute = sys.argv[2].split(":")
sys.exit(getattr(importlib.import_module(module), attribute)(sys.argv[3:]))
"""


def scripts():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["scripts"]


def flat(path):
    return " ".join(path.read_text(encoding="utf-8").split())


def without_triton(*argv):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), scripts()[COMMAND], *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_console_script_resolves_to_the_triton_ir_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.triton_ir_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is triton_ir_cli.main
    assert triton_ir_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{summarize,stage-map,check,verify}" in usage


def test_console_script_summarizes_and_verifies_without_triton_or_torch(tmp_path):
    report = tmp_path / "report.json"

    summarized = without_triton(
        "summarize", "--fixtures", str(FIXTURES), "--output", str(report)
    )
    verified = without_triton(
        "verify", "--report", str(report), "--fixtures", str(FIXTURES)
    )

    assert summarized.returncode == triton_ir_cli.EXIT_OK, summarized.stderr
    assert summarized.stdout.startswith("triton_ir manifest triton=3.9.0 ")
    assert verified.returncode == triton_ir_cli.EXIT_OK, verified.stderr
    sealed = summarized.stdout.splitlines()[-1].split()[0]
    assert sealed.startswith("triton_ir_recorded=")
    assert verified.stdout.split()[0] == f"verified_{sealed}"


def test_console_script_check_stops_without_triton_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "check.json"

    checked = without_triton(
        "check", "--fixtures", str(FIXTURES), "--output", str(output)
    )

    assert checked.returncode == triton_ir_cli.EXIT_UNAVAILABLE, checked.stderr
    assert checked.stdout == ""
    assert checked.stderr.startswith("triton-ir check unavailable: ")
    assert "(triton_not_importable)" in checked.stderr
    assert not output.parent.exists()


def test_the_guide_names_the_installed_command():
    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in flat(GUIDE)
