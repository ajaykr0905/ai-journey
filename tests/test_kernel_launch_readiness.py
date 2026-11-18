import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/kernel_launch_*.py"
TESTS = "tests/test_kernel_launch_*.py"
PINNED = "config/day-73-kernel-launch.json"
GUIDE = "docs/kernel-launch-lab.md"


def test_readiness_requires_every_kernel_launch_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {PINNED, GUIDE}

    assert {
        "src/ai_journey/kernel_launch_cli.py",
        "tests/test_kernel_launch_package.py",
        "tests/test_kernel_launch_ci_contract.py",
        "tests/test_kernel_launch_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
