import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readiness_requires_the_complete_minhash_operator_surface():
    readiness = json.loads(
        (ROOT / "config" / "deployment_readiness.json").read_text(encoding="utf-8")
    )
    required = set(readiness["required_files"])

    assert "config/day-66-minhash-dedup.json" in required
    assert "src/ai_journey/minhash_dedup.py" in required
    assert "src/ai_journey/minhash_verify.py" in required
    assert "src/ai_journey/minhash_cli.py" in required
    assert "tests/test_minhash_ci_contract.py" in required
    assert "tests/test_minhash_readiness.py" in required


def test_all_required_minhash_paths_exist():
    readiness = json.loads(
        (ROOT / "config" / "deployment_readiness.json").read_text(encoding="utf-8")
    )

    missing = [
        path
        for path in readiness["required_files"]
        if "minhash" in path.lower() and not (ROOT / path).is_file()
    ]
    assert missing == []
