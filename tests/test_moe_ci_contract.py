"""Keep the installed public experiment and its replay in the required CI path."""

import importlib
import json
from pathlib import Path
import shlex
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_moe_console_script_resolves_to_executable_cli(capsys):
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry = package["project"]["scripts"]["ai-journey-moe-experiment"]
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "replay" in capsys.readouterr().out


def test_moe_job_checks_real_paths_and_replays_an_isolated_installed_public_run():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  moe-cpu-lab:\n", 1)[1].split(
        "  nvidia-preflight-contracts:\n", 1
    )[0]
    assert "ruff check src/ai_journey/moe*.py tests/test_moe*.py" in job
    assert "python -m build" in job and '/moe-installed"' in job
    assert 'export PATH="$RUNNER_TEMP/moe-installed/bin:$PATH"' in job
    assert "ai_journey.__file__.startswith" in job
    for operation in ("plan", "public", "verify", "replay"):
        assert f"ai-journey-moe-experiment {operation}" in job
    assert "6e328e8195c04a297c1e375d21fe663be1ce6178/en_pud-ud-test.conllu" in job
    test_lines = [
        line.strip() for line in job.splitlines() if "python -m pytest" in line
    ]
    assert test_lines
    for line in test_lines:
        for pattern in shlex.split(line)[4:]:
            assert list(ROOT.glob(pattern)), f"missing CI test input: {pattern}"


def test_readiness_includes_the_complete_expert_runtime_and_verification_surface():
    required = json.loads((ROOT / "config/deployment_readiness.json").read_text())[
        "required_files"
    ]
    for pattern in ("src/ai_journey/moe*.py", "tests/test_moe*.py"):
        for path in ROOT.glob(pattern):
            assert path.relative_to(ROOT).as_posix() in required
    assert "docs/moe-core.md" in required
    assert "docs/moe-experiments.md" in required
