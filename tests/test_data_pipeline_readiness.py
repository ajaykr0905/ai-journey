import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_requires_every_operational_data_pipeline_layer():
    required = set(
        json.loads((ROOT / "config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )

    for path in (
        "config/day-59-data-pipeline.json",
        "config/day-59-data-pipeline-acceptance.json",
        "docs/day-59-data-pipeline.md",
        "src/ai_journey/data_pipeline.py",
        "src/ai_journey/data_pipeline_io.py",
        "src/ai_journey/data_pipeline_verify.py",
        "src/ai_journey/data_pipeline_acceptance.py",
        "src/ai_journey/data_pipeline_cli.py",
        "tests/test_data_pipeline_ci_contract.py",
    ):
        assert path in required
        assert (ROOT / path).is_file()
