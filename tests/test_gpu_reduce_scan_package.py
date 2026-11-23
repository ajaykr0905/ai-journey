"""The installed commands of the Day 78 reduction and scan labs.

Only ``run`` needs torch, and it imports torch when it runs, so each entry
point must load on a host without torch. The CUDA commands of the reduction
lab have no console script, and their guide must say so.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import gpu_reduce_cli, gpu_reduce_cuda_cli, gpu_scan_cli

ROOT = Path(__file__).resolve().parents[1]
# Each console script: its CLI module and the guide that names it.
COMMANDS = {
    "ai-journey-gpu-reduce": (gpu_reduce_cli, "docs/gpu-reduction-lab.md"),
    "ai-journey-gpu-scan": (gpu_scan_cli, "docs/gpu-scan-lab.md"),
}
CUDA_GUIDE = ROOT / "docs/gpu-reduce-cuda-contract.md"
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


def flat(path):
    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("command", COMMANDS)
def test_each_console_script_resolves_to_its_lab_cli(command, capsys):
    module, _ = COMMANDS[command]
    entry = scripts()[command]

    assert entry == f"{module.__name__}:main"
    name, attribute = entry.split(":")
    main = getattr(importlib.import_module(name), attribute)
    assert main is module.main
    assert module.PROG == command
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {command} ")
    assert "{run,verify}" in usage


@pytest.mark.parametrize("command", COMMANDS)
def test_each_console_script_loads_without_torch_or_matplotlib(command):
    entry = scripts()[command]
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), entry, "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(f"usage: {command} ")


@pytest.mark.parametrize("command", COMMANDS)
def test_each_guide_names_its_installed_command(command):
    _, guide = COMMANDS[command]
    text = flat(ROOT / guide)

    assert (
        f"The parser uses the program name `{command}`. An installed wheel gives "
        "the same commands under that name."
    ) in text
    assert "console script" not in text


def test_the_cuda_guide_keeps_python_m_because_it_has_no_console_script():
    assert gpu_reduce_cuda_cli.PROG == "ai-journey-gpu-reduce-cuda"
    assert gpu_reduce_cuda_cli.PROG not in scripts()
    assert f"{gpu_reduce_cuda_cli.__name__}:main" not in scripts().values()
    assert (
        f"the program name `{gpu_reduce_cuda_cli.PROG}`. The package has no console "
        "script for this lab, so the commands use `python -m`."
    ) in flat(CUDA_GUIDE)
