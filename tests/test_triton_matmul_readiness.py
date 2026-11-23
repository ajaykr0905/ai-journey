import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/triton_matmul_*.py"
TESTS = "tests/test_triton_matmul_*.py"
GUIDE = "docs/triton-matmul-lab.md"
# The pinned plan, and the Day 74 config whose correctness shapes it names.
CONFIGS = ("config/day-79-triton-matmul.json", "config/day-74-matmul-metal.json")
# The guides whose pinned digests and rates the line counts and rates cite.
CITED = (
    "docs/matmul-tiled-lab.md",
    "docs/matmul-checklist-lab.md",
    "docs/gpu-tiled-cuda-contract.md",
    "docs/gpu-checklist-cuda-contract.md",
)


def test_readiness_requires_every_triton_matmul_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, *CONFIGS, *CITED}

    assert {
        "src/ai_journey/triton_matmul_cli.py",
        "src/ai_journey/triton_matmul_kernel.py",
        "src/ai_journey/triton_matmul_verify.py",
        "tests/test_triton_matmul_interpreter.py",
        "tests/test_triton_matmul_recorded.py",
        "tests/test_triton_matmul_package.py",
        "tests/test_triton_matmul_ci_contract.py",
        "tests/test_triton_matmul_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
