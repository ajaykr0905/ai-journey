import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

# Each Day 78 lab: its guides and the configs that its documented commands
# read. The reduction prefix also holds the CUDA texts of that lab and the
# tests that both labs share, tests/test_gpu_reduce_scan_*.py.
LABS = {
    "gpu_reduce": (
        ("docs/gpu-reduction-lab.md", "docs/gpu-reduce-cuda-contract.md"),
        ("config/day-78-gpu-reduce.json", "config/day-74-gpu-ceilings.json"),
    ),
    "gpu_scan": (("docs/gpu-scan-lab.md",), ("config/day-78-gpu-scan.json",)),
}


@pytest.mark.parametrize("lab", LABS)
def test_readiness_requires_every_file_of_each_day_78_lab(lab):
    guides, configs = LABS[lab]
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (f"src/ai_journey/{lab}_*.py", f"tests/test_{lab}_*.py")
        for path in ROOT.glob(pattern)
    }
    expected |= {*guides, *configs}

    assert {
        f"src/ai_journey/{lab}_cli.py",
        f"src/ai_journey/{lab}_verify.py",
        f"tests/test_{lab}_guide.py",
        f"tests/test_{lab}_recorded.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
