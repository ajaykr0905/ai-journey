"""The installed command of the Day 80 fused kernel lab.

Only ``run`` needs torch, and it imports torch when it makes its runtime, so
the entry point must load on a host without torch. ``verify`` never imports
torch, so the installed command also checks the recorded report there.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import metal_liger_cli
from test_metal_liger_recorded import RECORDED_REPORT_JSON, recorded

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-metal-liger"
GUIDE = ROOT / "docs/metal-liger-lab.md"
# A None entry in sys.modules makes every import of that name fail.
BLOCKED = ("torch", "matplotlib", "triton")
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


def without_torch(*argv):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), scripts()[COMMAND], *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_console_script_resolves_to_the_metal_liger_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.metal_liger_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is metal_liger_cli.main
    assert metal_liger_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{run,verify}" in usage


def test_console_script_loads_and_verifies_without_torch(tmp_path):
    report = tmp_path / "recorded.json"
    report.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    usage = without_torch("--help")
    verified = without_torch("verify", "--report", str(report))

    assert usage.returncode == 0, usage.stderr
    assert usage.stdout.startswith(f"usage: {COMMAND} ")
    assert verified.returncode == metal_liger_cli.EXIT_OK, verified.stderr
    assert verified.stdout.startswith(
        f"verified_metal_liger_measurement={recorded()['report_sha256']} "
    )


def test_the_guide_names_the_installed_command():
    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in flat(GUIDE)
