import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/metal_flash_*.py"
TESTS = "tests/test_metal_flash_*.py"
GUIDE = "docs/metal-flash-attention-lab.md"
# The pinned config, and the Day 74 config of a fresh gpu-ceilings run.
CONFIGS = ("config/day-81-metal-flash.json", "config/day-74-gpu-ceilings.json")


def test_readiness_requires_every_metal_flash_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS}

    assert {
        "src/ai_journey/metal_flash_cli.py",
        "src/ai_journey/metal_flash_source.py",
        "src/ai_journey/metal_flash_verify.py",
        "tests/test_metal_flash_recorded.py",
        "tests/test_metal_flash_package.py",
        "tests/test_metal_flash_ci_contract.py",
        "tests/test_metal_flash_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
