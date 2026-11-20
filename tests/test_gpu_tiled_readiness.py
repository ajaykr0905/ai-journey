import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/gpu_tiled_*.py"
TESTS = "tests/test_gpu_tiled_*.py"
# The pinned plan lives in gpu_tiled_config, so the lab has no config file.
GUIDE = "docs/gpu-tiled-cuda-contract.md"


def test_readiness_requires_every_gpu_tiled_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected.add(GUIDE)

    assert {
        "src/ai_journey/gpu_tiled_cli.py",
        "src/ai_journey/gpu_tiled_config.py",
        "src/ai_journey/gpu_tiled_verify.py",
        "tests/test_gpu_tiled_report.py",
        "tests/test_gpu_tiled_guide.py",
        "tests/test_gpu_tiled_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
