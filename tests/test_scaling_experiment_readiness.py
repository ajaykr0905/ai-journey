import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_requires_the_operational_scaling_experiment_surface():
    required = set(
        json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )
    expected = {
        "config/day-62-scaling-experiment-fixture.json",
        "docs/scaling-experiment-protocol.md",
        "src/ai_journey/scaling_experiment_protocol.py",
        "src/ai_journey/scaling_experiment_budget.py",
        "src/ai_journey/scaling_experiment_identity.py",
        "src/ai_journey/scaling_experiment_observation.py",
        "src/ai_journey/scaling_experiment_analysis.py",
        "src/ai_journey/scaling_experiment_report.py",
        "src/ai_journey/scaling_experiment_verify.py",
        "src/ai_journey/scaling_experiment_io.py",
        "src/ai_journey/scaling_experiment_cli.py",
        "tests/test_scaling_experiment_package.py",
        "tests/test_scaling_experiment_ci_contract.py",
        "tests/test_scaling_experiment_docs.py",
        "tests/test_scaling_experiment_readiness.py",
    }

    assert expected <= required
    assert all((ROOT / path).is_file() for path in expected)
