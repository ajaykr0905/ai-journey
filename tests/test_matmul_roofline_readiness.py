import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/matmul_roofline_*.py"
TESTS = "tests/test_matmul_roofline_*.py"
GUIDE = "docs/matmul-roofline-gap.md"


def test_readiness_requires_every_matmul_roofline_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE}

    assert {
        "src/ai_journey/matmul_roofline_cli.py",
        "tests/test_matmul_roofline_recorded.py",
        "tests/test_matmul_roofline_package.py",
        "tests/test_matmul_roofline_ci_contract.py",
        "tests/test_matmul_roofline_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
