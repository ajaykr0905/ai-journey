import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness


def test_readiness_requires_complete_project1_operator():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    required = set(config["required_files"])
    expected = {
        "config/day-64-project1-run.json",
        "docs/project1-run-operator.md",
        "src/ai_journey/project1_contract.py",
        "src/ai_journey/project1_journal.py",
        "src/ai_journey/project1_bundle.py",
        "src/ai_journey/project1_operator.py",
        "src/ai_journey/project1_cli.py",
        "tests/test_project1_ci_contract.py",
    }

    assert expected <= required
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
