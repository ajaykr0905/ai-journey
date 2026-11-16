import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import memory_access_cli

ROOT = Path(__file__).resolve().parents[1]
PLAN = "config/day-74-memory-access-plan.json"
SMOKE = "config/day-74-memory-access-ci-smoke.json"


def test_console_script_resolves_to_the_memory_access_cli(capsys):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry = project["project"]["scripts"]["ai-journey-memory-access"]

    assert entry == "ai_journey.memory_access_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is memory_access_cli.main
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: ai-journey-memory-access ")
    assert "{analyze,benchmark,verify}" in usage


# Modules that analyze and verify reports; only ``benchmark`` may need torch.
TORCH_FREE_MODULES = (
    "ai_journey.memory_access_cli",
    "ai_journey.memory_access_plan",
    "ai_journey.memory_access_report",
    "ai_journey.memory_access_report_io",
    "ai_journey.memory_access_verify",
)
NEWLY_IMPORTED = """
import json, sys
before = set(sys.modules)
import {module}
print(json.dumps(sorted(set(sys.modules) - before)))
"""
# Setting a sys.modules entry to None makes every import of that name fail.
WITHOUT_TORCH_OR_NUMPY = """
import sys
sys.modules["torch"] = None
sys.modules["numpy"] = None
from ai_journey.memory_access_cli import main
sys.exit(main(sys.argv[1:]))
"""


def run_fresh(*arguments):
    return subprocess.run(
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_analysis_and_verification_modules_load_neither_torch_nor_numpy():
    for module in TORCH_FREE_MODULES:
        completed = run_fresh("-c", NEWLY_IMPORTED.format(module=module))
        assert completed.returncode == 0, completed.stderr
        loaded = set(json.loads(completed.stdout))

        assert module in loaded
        assert {"torch", "numpy"} & loaded == set(), module


def test_analyze_and_verify_run_where_torch_and_numpy_cannot_be_imported(
    tmp_path, capsys
):
    plan = str(ROOT / PLAN)
    smoke = str(ROOT / SMOKE)
    expected = tmp_path / "expected.json"
    measurement = tmp_path / "measurement.json"
    analysis = tmp_path / "analysis.json"
    assert (
        memory_access_cli.main(["analyze", "--config", plan, "--output", str(expected)])
        == 0
    )
    benchmark = ["benchmark", "--config", smoke, "--output", str(measurement)]
    assert memory_access_cli.main(benchmark) == 0
    capsys.readouterr()

    for arguments, prefix in (
        (
            ["analyze", "--config", plan, "--output", str(analysis)],
            "memory_access_analysis=",
        ),
        (
            ["verify", "--report", str(analysis), "--config", plan],
            "verified_memory_access_analysis=",
        ),
        (
            ["verify", "--report", str(measurement), "--config", smoke],
            "verified_memory_access_measurement=",
        ),
    ):
        completed = run_fresh("-c", WITHOUT_TORCH_OR_NUMPY, *arguments)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith(prefix), completed.stdout
    assert analysis.read_bytes() == expected.read_bytes()
