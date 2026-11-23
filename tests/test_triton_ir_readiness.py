import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from deployment_readiness_check import check_readiness  # noqa: E402

SOURCES = "src/ai_journey/triton_ir_*.py"
TESTS = "tests/test_triton_ir_*.py"
GUIDE = "docs/triton-ir-lab.md"
# The recorded dump: its index and the 3 committed stage texts.
FIXTURES = "tests/fixtures/day80_triton_ir/**/*"
# The Day 79 kernel text whose digest the dump names.
KERNEL = "src/ai_journey/triton_matmul_kernel.py"


def test_readiness_requires_every_triton_ir_file():
    config = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())
    expected = {
        path.relative_to(ROOT).as_posix()
        for pattern in (SOURCES, TESTS, FIXTURES)
        for path in ROOT.glob(pattern)
        if path.is_file()
    }
    expected |= {GUIDE, KERNEL}

    assert {
        "src/ai_journey/triton_ir_cli.py",
        "src/ai_journey/triton_ir_ptx.py",
        "src/ai_journey/triton_ir_verify.py",
        "tests/fixtures/day80_triton_ir/index.json",
        "tests/fixtures/day80_triton_ir/sm_80/m32_n64_k32_g8_w2_s5/ptx.ptx",
        "tests/test_triton_ir_package.py",
        "tests/test_triton_ir_ci_contract.py",
        "tests/test_triton_ir_readiness.py",
    } <= expected
    assert expected <= set(config["required_files"])
    report = check_readiness(ROOT, config)
    assert report["ready"], report["checks"]
