import json
from pathlib import Path


def test_deployment_readiness_tracks_every_finetune_source_and_test():
    required = set(
        json.loads(Path("config/deployment_readiness.json").read_text())[
            "required_files"
        ]
    )
    expected = {
        str(path)
        for pattern in (
            "src/ai_journey/sft_finetune_*.py",
            "tests/test_sft_finetune_*.py",
        )
        for path in Path().glob(pattern)
    }
    expected.update({"config/day-68-sft-finetune.json", "docs/sft-finetuning.md"})
    assert expected <= required
