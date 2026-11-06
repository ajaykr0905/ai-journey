from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness, load_config  # noqa: E402


def test_deployment_readiness_requires_the_complete_evaluation_harness():
    config = load_config(ROOT / "config/deployment_readiness.json")
    required = set(config["required_files"])
    expected = {
        "config/lambada-test.json",
        "docs/evaluation-harness.md",
        "src/ai_journey/evaluation_model.py",
        "src/ai_journey/evaluation_suite.py",
        "src/ai_journey/evaluation_suite_evidence.py",
        "src/ai_journey/evaluation_suite_cli.py",
        "src/ai_journey/lambada.py",
        "src/ai_journey/lambada_evidence.py",
        "src/ai_journey/lambada_cli.py",
        "tests/test_evaluation_ci_contract.py",
        "tests/test_lambada_config.py",
    }
    assert expected <= required
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
