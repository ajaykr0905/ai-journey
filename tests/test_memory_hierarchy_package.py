import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import memory_hierarchy_cli

ROOT = Path(__file__).resolve().parents[1]
SMOKE = "config/day-71-memory-hierarchy-ci-smoke.json"
# Setting sys.modules["torch"] to None makes every import of torch fail.
WITHOUT_TORCH = """
import sys
sys.modules["torch"] = None
from ai_journey.memory_hierarchy_cli import main
sys.exit(main(sys.argv[1:]))
"""


def test_console_script_resolves_to_the_memory_hierarchy_cli(capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["scripts"]["ai-journey-memory-hierarchy"]

    assert entry == "ai_journey.memory_hierarchy_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is memory_hierarchy_cli.main
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: ai-journey-memory-hierarchy ")
    assert "{run,verify}" in usage


def test_run_and_verify_work_where_torch_cannot_be_imported(tmp_path):
    report = str(tmp_path / "report.json")
    config = str(ROOT / SMOKE)
    for arguments, prefix in (
        (["run", "--config", config, "--output", report], "cpu_host_measurement "),
        (["verify", "--report", report, "--config", config], "verified_"),
    ):
        completed = subprocess.run(
            [sys.executable, "-c", WITHOUT_TORCH, *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith(prefix), completed.stdout
