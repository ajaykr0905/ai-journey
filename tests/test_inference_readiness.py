import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readiness_requires_every_operational_inference_layer():
    required = set(
        json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )
    expected = {
        "config/day-61-inference-plan.json",
        "docs/inference-capacity-planning.md",
        "src/ai_journey/inference_kv.py",
        "src/ai_journey/inference_capacity.py",
        "src/ai_journey/inference_paging.py",
        "src/ai_journey/inference_work.py",
        "src/ai_journey/inference_batching.py",
        "src/ai_journey/inference_speculative.py",
        "src/ai_journey/inference_protocol.py",
        "src/ai_journey/inference_report.py",
        "src/ai_journey/inference_seal.py",
        "src/ai_journey/inference_verify.py",
        "src/ai_journey/inference_io.py",
        "src/ai_journey/inference_cli.py",
        "tests/test_inference_ci_contract.py",
        "tests/test_inference_docs.py",
        "tests/test_inference_readiness.py",
    }

    assert expected <= required
    assert all((ROOT / path).is_file() for path in expected)
