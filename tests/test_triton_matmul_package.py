"""The installed command of the Day 79 Triton matmul lab.

Only ``check`` needs triton and torch, and it imports them when it makes its
backend. Triton has no macOS wheel, so the entry point must load, count lines
and verify on a host without triton or torch.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import triton_matmul_cli
from test_triton_matmul_recorded import RECORDED_REPORT_JSON, RECORDED_SHA256

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-triton-matmul"
GUIDE = ROOT / "docs/triton-matmul-lab.md"
PINNED = ROOT / "config/day-79-triton-matmul.json"
# A None entry in sys.modules makes every import of that name fail.
BLOCKED = ("triton", "torch", "matplotlib")
WITHOUT = """
import importlib
import sys
for name in sys.argv[1].split(","):
    sys.modules[name] = None
module, attribute = sys.argv[2].split(":")
sys.exit(getattr(importlib.import_module(module), attribute)(sys.argv[3:]))
"""


def scripts():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return project["project"]["scripts"]


def flat(path):
    return " ".join(path.read_text(encoding="utf-8").split())


def without_triton(*argv):
    return subprocess.run(
        [sys.executable, "-c", WITHOUT, ",".join(BLOCKED), scripts()[COMMAND], *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_console_script_resolves_to_the_triton_matmul_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.triton_matmul_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is triton_matmul_cli.main
    assert triton_matmul_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{loc,check,verify}" in usage


def test_console_script_counts_and_verifies_without_triton_or_torch(tmp_path):
    report = tmp_path / "recorded.json"
    report.write_text(RECORDED_REPORT_JSON + "\n", encoding="ascii")

    counted = without_triton("loc")
    verified = without_triton(
        "verify", "--report", str(report), "--config", str(PINNED)
    )

    assert counted.returncode == 0, counted.stderr
    assert counted.stdout.startswith(
        "loc lab=day-79 language=triton text=matmul_kernel kernel=33 host=9 "
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.startswith(
        f"verified_triton_matmul_interpreter={RECORDED_SHA256} "
    )


def test_the_guide_names_the_installed_command():
    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in flat(GUIDE)
