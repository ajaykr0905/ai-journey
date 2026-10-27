"""The opt-in prefix operator is gated through an isolated installed wheel."""

import importlib
import json
from pathlib import Path
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readiness_module_exposes_execution_and_artifact_verification(capsys):
    main = importlib.import_module("ai_journey.prefix_readiness").main
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--installed-root" in help_text and "--verify-report" in help_text


def test_ci_operates_installed_prefix_service_and_rechecks_retained_evidence():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    job = workflow.split("  prefix-cache-cpu:\n", 1)[1].split(
        "  grouped-query-cpu:\n", 1
    )[0]
    for required in (
        "python -m build",
        "ai_journey.__file__.startswith",
        "python -m ai_journey.prefix_readiness --installed-root",
        "python -m ai_journey.prefix_readiness --verify-report",
        '--artifact-dir "$RUNNER_TEMP/prefix-evidence"',
    ):
        assert required in job
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line.strip())[4:]:
                assert list(ROOT.glob(pattern)), f"missing CI input: {pattern}"


def test_readiness_requires_the_complete_prefix_execution_and_evidence_surface():
    required = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())[
        "required_files"
    ]
    for pattern in ("src/ai_journey/prefix*.py", "tests/test_prefix*.py"):
        for path in ROOT.glob(pattern):
            assert path.relative_to(ROOT).as_posix() in required
    assert "docs/prefix-kv-cache.md" in required
