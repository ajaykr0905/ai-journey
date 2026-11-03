import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_requires_every_operational_scaling_layer():
    required = set(
        json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )
    expected = {
        "config/day-60-scaling-plan.json",
        "docs/scaling-law-planning.md",
        "src/ai_journey/scaling_laws.py",
        "src/ai_journey/scaling_budget.py",
        "src/ai_journey/scaling_schedule.py",
        "src/ai_journey/scaling_corpus.py",
        "src/ai_journey/scaling_protocol.py",
        "src/ai_journey/scaling_report.py",
        "src/ai_journey/scaling_verify.py",
        "src/ai_journey/scaling_io.py",
        "src/ai_journey/scaling_cli.py",
        "tests/test_scaling_ci_contract.py",
        "tests/test_scaling_docs.py",
        "tests/test_scaling_readiness.py",
    }

    assert expected <= required
    assert all((ROOT / path).is_file() for path in expected)
