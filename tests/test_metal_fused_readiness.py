import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/metal_fused_*.py"
TESTS = "tests/test_metal_fused_*.py"
GUIDE = "docs/metal-fused-softmax-lab.md"
# The pinned config, and the Day 74 config of a fresh gpu-ceilings run.
CONFIGS = ("config/day-79-metal-fused.json", "config/day-74-gpu-ceilings.json")


def test_readiness_requires_every_metal_fused_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS}

    assert {
        "src/ai_journey/metal_fused_cli.py",
        "src/ai_journey/metal_fused_source.py",
        "src/ai_journey/metal_fused_verify.py",
        "tests/test_metal_fused_recorded.py",
        "tests/test_metal_fused_triton.py",
        "tests/test_metal_fused_package.py",
        "tests/test_metal_fused_ci_contract.py",
        "tests/test_metal_fused_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
