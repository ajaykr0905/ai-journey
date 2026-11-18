import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/matmul_model_*.py"
TESTS = "tests/test_matmul_model_*.py"
PINNED = "config/day-74-matmul-model.json"
GUIDE = "docs/matmul-model-lab.md"


def test_readiness_requires_every_matmul_model_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {PINNED, GUIDE}

    assert {
        "src/ai_journey/matmul_model_cli.py",
        "tests/test_matmul_model_pinned.py",
        "tests/test_matmul_model_package.py",
        "tests/test_matmul_model_ci_contract.py",
        "tests/test_matmul_model_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
