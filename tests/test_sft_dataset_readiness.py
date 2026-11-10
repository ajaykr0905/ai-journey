import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_readiness_requires_every_sft_dataset_contract():
    readiness = json.loads((ROOT / "config/deployment_readiness.json").read_text())
    required = set(readiness["required_files"])
    expected = {
        "config/day-67-sft-dataset.json",
        *{
            str(path.relative_to(ROOT))
            for path in (ROOT / "src/ai_journey").glob("sft_dataset_*.py")
        },
        *{
            str(path.relative_to(ROOT))
            for path in (ROOT / "tests").glob("test_sft_dataset_*.py")
        },
    }
    assert expected <= required


def test_every_registered_sft_dataset_path_exists_and_is_a_regular_file():
    readiness = json.loads((ROOT / "config/deployment_readiness.json").read_text())
    paths = [
        ROOT / path
        for path in readiness["required_files"]
        if "sft_dataset" in path or "day-67-sft-dataset" in path
    ]
    assert paths
    assert all(path.is_file() and not path.is_symlink() for path in paths)
