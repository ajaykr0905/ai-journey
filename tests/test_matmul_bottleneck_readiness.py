import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/matmul_bottleneck_*.py"
TESTS = "tests/test_matmul_bottleneck_*.py"
GUIDE = "docs/matmul-bottleneck-lab.md"
CONFIGS = (
    "config/day-75-matmul-bottleneck.json",
    "config/day-75-matmul-metal-cubes.json",
)


def test_readiness_requires_every_matmul_bottleneck_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS}

    assert {
        "src/ai_journey/matmul_bottleneck_cli.py",
        "src/ai_journey/matmul_bottleneck_gpu_timing.py",
        "tests/test_matmul_bottleneck_recorded.py",
        "tests/test_matmul_bottleneck_package.py",
        "tests/test_matmul_bottleneck_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
