import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/gpu_checklist_*.py"
TESTS = "tests/test_gpu_checklist_*.py"
GUIDE = "docs/gpu-checklist-cuda-contract.md"
# The pinned kernels and the 18 check shapes live in code, so the lab has no
# config file of its own. A test holds those shapes to the Day 74 Metal config.
SHAPES = "config/day-74-matmul-metal.json"


def test_readiness_requires_every_gpu_checklist_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, SHAPES}

    assert {
        "src/ai_journey/gpu_checklist_cli.py",
        "src/ai_journey/gpu_checklist_check.py",
        "src/ai_journey/gpu_checklist_cuda.py",
        "tests/test_gpu_checklist_guide.py",
        "tests/test_gpu_checklist_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
