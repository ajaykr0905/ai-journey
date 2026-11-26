import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/metal_tc_*.py"
TESTS = "tests/test_metal_tc_*.py"
GUIDE = "docs/metal-matrix-precision-lab.md"
# The pinned config, and the Day 74 config of a fresh gpu-ceilings run.
CONFIGS = ("config/day-82-metal-tc.json", "config/day-74-gpu-ceilings.json")


def test_readiness_requires_every_metal_tc_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS}

    assert {
        "src/ai_journey/metal_tc_cli.py",
        "src/ai_journey/metal_tc_source.py",
        "src/ai_journey/metal_tc_verify.py",
        "tests/test_metal_tc_recorded.py",
        "tests/test_metal_tc_package.py",
        "tests/test_metal_tc_ci_contract.py",
        "tests/test_metal_tc_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
