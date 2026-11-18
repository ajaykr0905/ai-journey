import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/gpu_ceiling_*.py"
TESTS = "tests/test_gpu_ceiling_*.py"
PINNED = "config/day-74-gpu-ceilings.json"
GUIDE = "docs/gpu-ceilings-lab.md"


def test_readiness_requires_every_gpu_ceiling_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {PINNED, GUIDE}

    assert {
        "src/ai_journey/gpu_ceiling_cli.py",
        "tests/test_gpu_ceiling_recorded.py",
        "tests/test_gpu_ceiling_package.py",
        "tests/test_gpu_ceiling_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
