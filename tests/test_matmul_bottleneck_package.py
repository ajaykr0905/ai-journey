import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import matmul_bottleneck_cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-matmul-bottleneck"
GUIDE = ROOT / "docs/matmul-bottleneck-lab.md"
# A None entry in sys.modules makes every import of that name fail. The command reads reports only, so it never needs torch.
BLOCKED = ("torch", "matplotlib")
WITHOUT = """
import importlib
import sys
for name in sys.argv[1].split(","):
    sys.modules[name] = None
module, attribute = sys.argv[2].split(":")
sys.exit(getattr(importlib.import_module(module), attribute)(sys.argv[3:]))
"""


def entry_point():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["scripts"][COMMAND]


def test_console_script_resolves_to_the_matmul_bottleneck_cli(capsys):
    entry = entry_point()

    assert entry == "ai_journey.matmul_bottleneck_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is matmul_bottleneck_cli.main
    assert matmul_bottleneck_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{classify,verify}" in usage


def test_console_script_loads_without_torch_or_matplotlib():
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), entry_point(), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(f"usage: {COMMAND} ")


def test_the_guide_names_the_installed_command():
    guide = " ".join(GUIDE.read_text(encoding="utf-8").split())

    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in guide
