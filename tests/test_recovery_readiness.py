import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness


def test_readiness_requires_complete_worker_recovery_operator():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    required = set(config["required_files"])
    expected = {
        "config/day-70-worker-recovery.json",
        "docs/project1-release.md",
        "src/ai_journey/recovery_experiment.py",
        "src/ai_journey/recovery_verify.py",
        "src/ai_journey/recovery_cli.py",
        "tests/test_recovery_experiment.py",
        "tests/test_recovery_delivery.py",
    }

    assert expected <= required
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
