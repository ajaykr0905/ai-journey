"""The installed command of the Day 82 Tensor Core lab.

Only ``compile`` needs triton, and it imports triton when it makes its
backend. Triton has no macOS wheel, so the entry point must load, model and
verify on a host without triton or torch. There ``compile`` must stop with
exit code 3 before it makes the output folder or the dump folder.
"""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from ai_journey import triton_tc_cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "ai-journey-triton-tc"
GUIDE = ROOT / "docs/triton-tensor-core-lab.md"
PINNED = ROOT / "config/day-82-triton-tc.json"
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


def test_console_script_resolves_to_the_triton_tc_cli(capsys):
    entry = scripts()[COMMAND]

    assert entry == "ai_journey.triton_tc_cli:main"
    module, attribute = entry.split(":")
    main = getattr(importlib.import_module(module), attribute)
    assert main is triton_tc_cli.main
    assert triton_tc_cli.PROG == COMMAND
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith(f"usage: {COMMAND} ")
    assert "{model,compile,verify}" in usage


def test_console_script_models_and_verifies_without_triton_or_torch(tmp_path):
    report = tmp_path / "out" / "model.json"

    modeled = without_triton("model", "--config", str(PINNED), "--output", str(report))
    verified = without_triton(
        "verify", "--report", str(report), "--config", str(PINNED)
    )

    assert modeled.returncode == triton_tc_cli.EXIT_OK, modeled.stderr
    lines = modeled.stdout.splitlines()
    # One header, then one device line and five size lines for each device.
    assert len(lines) == 1 + 2 * (1 + 5) + 1
    assert (
        "triton_tc model_row device=h100-sxm5-80gb n=8192 fp32_us=16410.621 "
        "tf32_us=2223.482 bf16_us=1111.179 bound=compute,compute,compute "
        "ratio=14.77:2.00:1 memory_bound=none"
    ) in lines
    assert lines[-1].startswith("triton_tc_model=")
    assert verified.returncode == triton_tc_cli.EXIT_OK, verified.stderr
    assert verified.stdout.startswith("verified_triton_tc_model=")
    assert verified.stdout.rstrip().endswith("texts_recounted=false")


def test_console_script_compile_stops_without_triton_and_writes_nothing(tmp_path):
    output = tmp_path / "new" / "compile.json"
    dump = tmp_path / "new-dump"

    compiled = without_triton(
        "compile",
        "--config",
        str(PINNED),
        "--output",
        str(output),
        "--dump",
        str(dump),
    )

    assert compiled.returncode == triton_tc_cli.EXIT_UNAVAILABLE, compiled.stderr
    assert compiled.stdout == ""
    assert compiled.stderr.startswith("triton-tc compile unavailable: ")
    assert "(triton_not_importable)" in compiled.stderr
    assert not output.parent.exists()
    assert not dump.exists()


def test_the_guide_names_the_installed_command():
    assert (
        f"The parser uses the program name `{COMMAND}`. An installed wheel gives "
        "the same commands under that name."
    ) in flat(GUIDE)
