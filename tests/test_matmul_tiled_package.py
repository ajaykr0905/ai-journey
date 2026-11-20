"""The installed command of the tiled matmul lab.

Only ``run`` needs torch, and it imports torch when it runs, so the entry
point must load on a host without torch. The gap commands have no console
script, and the guide must say so.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import matmul_tiled_cli, matmul_tiled_gap_cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-matmul-tiled"
GUIDE = ROOT / "docs/matmul-tiled-lab.md"
# A None entry in sys.modules makes every import of that name fail.
BLOCKED = ("torch", "matplotlib")
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


def test_console_script_resolves_to_the_matmul_tiled_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.matmul_tiled_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is matmul_tiled_cli.main
    assert matmul_tiled_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{run,verify}" in usage


def test_console_script_loads_without_torch_or_matplotlib():
    entry = scripts()[COMMAND]
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), entry, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(f"usage: {COMMAND} ")


def test_the_guide_names_the_installed_command_and_python_m_for_the_gap():
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())

    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in guide
    assert matmul_tiled_gap_cli.PROG not in scripts()
    assert (
        f"the program name `{matmul_tiled_gap_cli.PROG}`. The package has no "
        "console script for these commands, so run the module with `python -m`."
    ) in guide
