import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/matmul_tiled_*.py"
TESTS = "tests/test_matmul_tiled_*.py"
GUIDE = "docs/matmul-tiled-lab.md"
# The pinned tiled config, and the ceiling and rule configs of the gap commands.
CONFIGS = (
    "config/day-76-matmul-tiled.json",
    "config/day-74-gpu-ceilings.json",
    "config/day-75-matmul-bottleneck.json",
)


def test_readiness_requires_every_matmul_tiled_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS}

    assert {
        "src/ai_journey/matmul_tiled_cli.py",
        "src/ai_journey/matmul_tiled_gap_cli.py",
        "tests/test_matmul_tiled_recorded.py",
        "tests/test_matmul_tiled_gap_recorded.py",
        "tests/test_matmul_tiled_package.py",
        "tests/test_matmul_tiled_ci_contract.py",
        "tests/test_matmul_tiled_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
