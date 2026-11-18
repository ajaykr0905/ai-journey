import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import gpu_timing_cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-gpu-timing"
GUIDE = ROOT / "docs/gpu-timing-lab.md"
# A None entry in sys.modules makes every import of that name fail. Only run needs torch, and it imports torch when it runs.
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


def test_console_script_resolves_to_the_gpu_timing_cli(capsys):
    entry = entry_point()

    assert entry == "ai_journey.gpu_timing_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is gpu_timing_cli.main
    assert gpu_timing_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{run,verify}" in usage


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
