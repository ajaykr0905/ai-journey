import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/matmul_checklist_*.py"
TESTS = "tests/test_matmul_checklist_*.py"
GUIDES = ("docs/matmul-checklist-lab.md", "docs/gpu-performance-checklist.md")
# The pinned checklist config, and the ceiling and rule configs of the bound.
CONFIGS = (
    "config/day-77-matmul-checklist.json",
    "config/day-74-gpu-ceilings.json",
    "config/day-75-matmul-bottleneck.json",
)


def test_readiness_requires_every_matmul_checklist_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {*GUIDES, *CONFIGS}

    assert {
        "src/ai_journey/matmul_checklist_cli.py",
        "src/ai_journey/matmul_checklist_bound_cli.py",
        "tests/test_matmul_checklist_recorded.py",
        "tests/test_matmul_checklist_bound_recorded.py",
        "tests/test_matmul_checklist_package.py",
        "tests/test_matmul_checklist_ci_contract.py",
        "tests/test_matmul_checklist_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
