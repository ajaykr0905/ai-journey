import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import telemetry_metrics_cli, telemetry_traces_cli

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "ai-journey-telemetry-metrics": (telemetry_metrics_cli, "{export,verify}"),
    "ai-journey-telemetry-traces": (telemetry_traces_cli, "{export,dashboard,verify}"),
}
# A None entry in sys.modules makes every import of that name fail, as on the
# telemetry-cpu runner, which installs none of these packages.
WITHOUT_HEAVY_PACKAGES = """
import importlib
import sys
for name in ("torch", "numpy", "matplotlib"):
    sys.modules[name] = None
module, attribute = sys.argv[1].split(":")
sys.exit(getattr(importlib.import_module(module), attribute)(sys.argv[2:]))
"""


def entry_point(command):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["scripts"][command]


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_console_script_resolves_to_its_telemetry_cli(command, capsys):
    cli, subcommands = COMMANDS[command]
    entry = entry_point(command)

    assert entry == f"{cli.__name__}:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is cli.main
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {command} ")
    assert subcommands in usage


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_console_script_loads_without_torch_numpy_or_matplotlib(command):
    completed = subprocess.run(
        [sys.executable, "-c", WITHOUT_HEAVY_PACKAGES, entry_point(command), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith(f"usage: {command} ")
