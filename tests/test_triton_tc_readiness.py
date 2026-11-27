import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/triton_tc_*.py"
TESTS = "tests/test_triton_tc_*.py"
GUIDE = "docs/triton-tensor-core-lab.md"
CONFIG = "config/day-82-triton-tc.json"
# The Day 79 runtime that names the declared triton release, and the
# recorded Day 80 texts that the analyzer and the verifier read.
REUSED = (
    "src/ai_journey/triton_matmul_runtime.py",
    "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ptx.ptx",
    "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ttgir.mlir",
)


def test_readiness_requires_every_triton_tc_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS)
        for path in ROOT.glob(pattern)
    }
    expected |= {GUIDE, CONFIG, *REUSED}

    assert {
        "src/ai_journey/triton_tc_cli.py",
        "src/ai_journey/triton_tc_kernel.py",
        "src/ai_journey/triton_tc_verify.py",
        "tests/test_triton_tc_compile_contract.py",
        "tests/test_triton_tc_package.py",
        "tests/test_triton_tc_ci_contract.py",
        "tests/test_triton_tc_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
