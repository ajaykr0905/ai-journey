"""The installed command of the Day 81 Triton flash attention lab.

Only ``check`` needs triton, and it imports triton when it makes its
backend. Triton has no macOS wheel, so the entry point must load, print the
model and verify on a host without triton or torch. There ``check`` must
stop with exit code 3 before it writes anything.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import triton_flash_cli
from ai_journey.triton_flash_memory import MEMORY_TABLE
from test_triton_flash_cli import check

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-triton-flash"
GUIDE = ROOT / "docs/triton-flash-attention-lab.md"
PINNED = ROOT / "config/day-81-triton-flash.json"
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


def test_console_script_resolves_to_the_triton_flash_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.triton_flash_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is triton_flash_cli.main
    assert triton_flash_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{model,check,verify}" in usage


def test_console_script_models_and_verifies_without_triton_or_torch(tmp_path, capsys):
    report = tmp_path / "out" / "report.json"
    assert check(tmp_path, report) == triton_flash_cli.EXIT_OK
    capsys.readouterr()

    modeled = without_triton("model")
    verified = without_triton("verify", "--report", str(report))

    assert modeled.returncode == triton_flash_cli.EXIT_OK, modeled.stderr
    assert modeled.stdout.splitlines() == triton_flash_cli.model_lines()
    assert len(modeled.stdout.splitlines()) == len(MEMORY_TABLE) + 1
    assert verified.returncode == triton_flash_cli.EXIT_OK, verified.stderr
    assert verified.stdout.startswith("verified_triton_flash_synthetic=")


def test_console_script_check_stops_without_triton_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "check.json"

    checked = without_triton("check", "--config", str(PINNED), "--output", str(output))

    assert checked.returncode == triton_flash_cli.EXIT_UNAVAILABLE, checked.stderr
    assert checked.stdout == ""
    assert checked.stderr.startswith("triton-flash check unavailable: ")
    assert "(triton_not_importable)" in checked.stderr
    assert not output.parent.exists()


def test_the_guide_names_the_installed_command():
    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in flat(GUIDE)
