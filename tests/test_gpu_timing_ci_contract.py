"""The Linux CI job of the gpu-timing and matmul-bottleneck labs.

The job ``gpu-timing-contracts`` installs CPU-only torch, so the tests and
the installed Apple GPU commands meet the path where MPS is not available.
Each installed run must stop with exit code 3 and write nothing, and the
installed verifiers must accept the recorded Apple GPU (Metal) reports.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_timing_cli, matmul_bottleneck_cli, profile_trace_cli
from test_matmul_bottleneck_recorded import REPORT_SHA256 as BOTTLENECK_SHA256
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS
from test_matmul_roofline_ci_contract import ON_LINUX, REASON, ci_job

ROOT = Path(__file__).resolve().parents[1]
JOB = "gpu-timing-contracts"
LABS = ("gpu_timing", "matmul_bottleneck")
SOURCES = " ".join(f"src/ai_journey/{lab}_*.py" for lab in LABS)
TESTS = " ".join(f"tests/test_{lab}_*.py" for lab in LABS)
EVIDENCE = "$RUNNER_TEMP/gpu-timing-evidence/"
# Each installed Apple GPU command: its CLI module, its message prefix, its
# subcommand, its output option and the pinned config that it must refuse.
GPU_COMMANDS = {
    "ai-journey-gpu-timing": (
        gpu_timing_cli,
        "gpu-timing run",
        "output",
        "config/day-75-gpu-timing.json",
    ),
    "ai-journey-profile-trace": (
        profile_trace_cli,
        "profile-trace capture",
        "output_dir",
        "config/day-75-profile-trace.json",
    ),
}
# The verify command of each recorded report and the test module that holds it.
RECORDED = {
    "ai-journey-gpu-timing": (gpu_timing_cli, "test_gpu_timing_recorded"),
    "ai-journey-matmul-bottleneck": (
        matmul_bottleneck_cli,
        "test_matmul_bottleneck_recorded",
    ),
}


def refusal_message(prefix):
    return f"{prefix} unavailable: {UNAVAILABLE_REASONS[REASON]} ({REASON})"


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def installed_run(command):
    (words,) = [
        shlex.split(line)
        for line in lines()
        if line.startswith(f"{command} ") and "2>" in line
    ]
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[1:redirect], words[redirect + 1]


def test_ci_gates_both_labs_with_cpu_only_torch_and_numpy():
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
        "${{ runner.temp }}/gpu-timing-evidence/",
        "name: gpu-timing-contracts-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


@pytest.mark.parametrize("command", GPU_COMMANDS)
def test_ci_requires_each_installed_gpu_run_to_stop_without_a_report(command):
    module, prefix, option, pinned = GPU_COMMANDS[command]
    job = lines()
    arguments, refusal = installed_run(command)
    run = module.build_parser().parse_args(arguments)
    output = getattr(run, option)

    assert run.config == Path(f"$GITHUB_WORKSPACE/{pinned}")
    assert str(output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(output)
    start = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"{command} ") and "2>" in line
    )
    assert job[start - 1] == "status=0"
    assert job[start + 1 : start + 5] == [
        f'cat "{refusal}"',
        f'test "$status" -eq {module.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message(prefix)}" "{refusal}"',
        f'test ! -e "{output}"',
    ]
    assert module.EXIT_UNAVAILABLE == 3


@pytest.mark.parametrize("command", GPU_COMMANDS)
def test_the_ci_expectations_match_each_command_on_a_linux_host(command, tmp_path):
    module, prefix, option, pinned = GPU_COMMANDS[command]
    arguments, _ = installed_run(command)
    output = tmp_path / "out"
    flag = "--" + option.replace("_", "-")
    local = [
        str(ROOT / pinned) if word.endswith(pinned) else word for word in arguments
    ]
    local[local.index(flag) + 1] = str(output)

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


@pytest.mark.parametrize("command", RECORDED)
def test_ci_verifies_each_recorded_report_with_the_installed_command(
    command, tmp_path, capsys
):
    module, source = RECORDED[command]
    job = lines()
    (writer,) = [shlex.split(line) for line in job if f"from {source} import" in line]
    (verify,) = [
        shlex.split(line) for line in job if line.startswith(f"{command} verify ")
    ]
    parsed = module.build_parser().parse_args(verify[1:])

    assert writer[:2] == ["python", "-c"]
    assert str(parsed.report) == writer[3]
    assert writer[3].startswith(EVIDENCE)
    path = tmp_path / "recorded.json"
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
    assert module.main(["verify", "--report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "_measurement=" in out
    if command == "ai-journey-matmul-bottleneck":
        assert out.startswith(
            f"verified_matmul_bottleneck_measurement={BOTTLENECK_SHA256}"
        )
