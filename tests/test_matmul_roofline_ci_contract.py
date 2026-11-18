import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import (
    gpu_ceiling_cli,
    matmul_metal_cli,
    matmul_roofline_cli,
    metal_launch_sweep_cli,
)
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from test_matmul_roofline_recorded import GAP_SHA256

ROOT = Path(__file__).resolve().parents[1]
JOB = "metal-roofline-contracts"
LABS = ("matmul_metal", "gpu_ceiling", "metal_launch_sweep", "matmul_roofline")
SOURCES = " ".join(f"src/ai_journey/{lab}_*.py" for lab in LABS)
TESTS = " ".join(f"tests/test_{lab}_*.py" for lab in LABS)
EVIDENCE = "$RUNNER_TEMP/metal-roofline-evidence/"
REASON = "unsupported_operating_system"
# Each installed Metal command: its CLI module, its message prefix and the
# pinned config that its run must refuse on the Linux runner.
METAL_COMMANDS = {
    "ai-journey-matmul-metal": (
        matmul_metal_cli,
        "matmul-metal",
        "config/day-74-matmul-metal.json",
    ),
    "ai-journey-gpu-ceilings": (
        gpu_ceiling_cli,
        "gpu-ceiling",
        "config/day-74-gpu-ceilings.json",
    ),
    "ai-journey-metal-launch-sweep": (
        metal_launch_sweep_cli,
        "metal-launch-sweep",
        "config/day-74-metal-launch-sweep.json",
    ),
}
# The probe reads platform.system before it imports torch, so replacing it
# before the import makes this host answer as the Linux runner does.
ON_LINUX = """
import importlib
import platform
import sys
platform.system = lambda: "Linux"
code = importlib.import_module(sys.argv[1]).main(sys.argv[2:])
print("torch imported" if "torch" in sys.modules else "torch not imported")
sys.exit(code)
"""


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def refusal_message(prefix):
    return f"{prefix} run unavailable: {UNAVAILABLE_REASONS[REASON]} ({REASON})"


def installed_run(command):
    """One installed run line as its arguments, stderr path and fallback."""

    (words,) = [
        shlex.split(line)
        for line in ci_job(JOB).splitlines()
        if line.strip().startswith(f"{command} ")
    ]
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[1:redirect], words[redirect + 1]


def test_ci_gates_the_four_labs_with_cpu_only_torch_and_numpy():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    installs = [line.strip() for line in job.splitlines() if "pip install" in line]
    assert installs[:3] == [
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        'python -m pip install "numpy>=1.24,<2"',
        "python -m pip install -e . --no-deps",
    ]
    assert not [line for line in installs if "matplotlib" in line]
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f"python -m pytest -q -rs {TESTS}",
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "import torch; assert not torch.backends.mps.is_available()",
        "${{ runner.temp }}/metal-roofline-evidence/",
        "name: metal-roofline-contracts-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


@pytest.mark.parametrize("command", METAL_COMMANDS)
def test_ci_requires_each_installed_metal_run_to_stop_without_a_report(command):
    module, prefix, pinned = METAL_COMMANDS[command]
    job = [line.strip() for line in ci_job(JOB).splitlines()]
    arguments, refusal = installed_run(command)
    run = module.build_parser().parse_args(arguments)

    assert run.command == "run"
    assert run.config == Path(f"$GITHUB_WORKSPACE/{pinned}")
    assert str(run.output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(run.output)
    checks = [
        "status=0",
        next(line for line in job if line.startswith(f"{command} ")),
        f'cat "{refusal}"',
        f'test "$status" -eq {module.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message(prefix)}" "{refusal}"',
        f'test ! -e "{run.output}"',
    ]
    start = job.index(checks[1]) - 1
    assert job[start : start + len(checks)] == checks
    assert module.EXIT_UNAVAILABLE == 3


@pytest.mark.parametrize("command", METAL_COMMANDS)
def test_the_ci_expectations_match_each_command_on_a_linux_host(command, tmp_path):
    module, prefix, pinned = METAL_COMMANDS[command]
    arguments, _ = installed_run(command)
    output = tmp_path / "report.json"
    local = [
        str(ROOT / pinned) if word.endswith(pinned) else word for word in arguments
    ]
    local[local.index("--output") + 1] = str(output)

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, module.__name__, *local],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == module.EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith(refusal_message(prefix))
    assert completed.stdout == "torch not imported\n"
    assert not output.exists()


def test_ci_verifies_the_recorded_gap_with_the_installed_command(tmp_path, capsys):
    lines = [line.strip() for line in ci_job(JOB).splitlines()]
    (writer,) = [shlex.split(line) for line in lines if "RECORDED_GAP_JSON" in line]
    (verify,) = [
        shlex.split(line)
        for line in lines
        if line.startswith("ai-journey-matmul-roofline ")
    ]
    parsed = matmul_roofline_cli.build_parser().parse_args(verify[1:])

    assert writer[:2] == ["python", "-c"]
    assert parsed.command == "verify"
    assert str(parsed.report) == writer[3]
    assert writer[3].startswith(EVIDENCE)
    # The writer runs here as on the runner, and the command checks its file.
    path = tmp_path / "recorded-gap.json"
    completed = subprocess.run(
        [sys.executable, "-c", writer[2], str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(ROOT),
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert matmul_roofline_cli.main(["verify", "--report", str(path)]) == 0
    assert capsys.readouterr().out.startswith(
        f"verified_matmul_roofline_measurement={GAP_SHA256} "
    )
