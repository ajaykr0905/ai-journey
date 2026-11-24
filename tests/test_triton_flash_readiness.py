import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/triton_flash_*.py"
TESTS = "tests/test_triton_flash_*.py"
GUIDE = "docs/triton-flash-attention-lab.md"
PLAN = "config/day-81-triton-flash.json"
# The Day 79 runtime whose interpreter names and reasons this lab imports.
RUNTIME = "src/ai_journey/triton_matmul_runtime.py"


def test_readiness_requires_every_triton_flash_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, PLAN, RUNTIME}

    assert {
        "src/ai_journey/triton_flash_cli.py",
        "src/ai_journey/triton_flash_kernel.py",
        "src/ai_journey/triton_flash_verify.py",
        "tests/test_triton_flash_interpreter.py",
        "tests/test_triton_flash_package.py",
        "tests/test_triton_flash_ci_contract.py",
        "tests/test_triton_flash_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
