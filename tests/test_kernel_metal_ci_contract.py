import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey.kernel_metal_cli import EXIT_UNAVAILABLE, build_parser
from ai_journey.kernel_metal_runtime import UNAVAILABLE_REASONS

ROOT = Path(__file__).resolve().parents[1]
JOB = "kernel-metal-contracts"
SOURCES = "src/ai_journey/kernel_metal_*.py"
TESTS = "tests/test_kernel_metal_*.py"
PINNED = "config/day-73-kernel-metal.json"
EVIDENCE = "$RUNNER_TEMP/kernel-metal-evidence/"
REASON = "unsupported_operating_system"
REFUSAL = f"kernel-metal run unavailable: {UNAVAILABLE_REASONS[REASON]} ({REASON})"
# The probe reads platform.system before it imports torch, so replacing it
# before the import makes this host answer as the Linux runner does.
ON_LINUX = """
import platform
import sys
platform.system = lambda: "Linux"
from ai_journey.kernel_metal_cli import main
code = main(sys.argv[1:])
print("torch imported" if "torch" in sys.modules else "torch not imported")
sys.exit(code)
"""


def ci_job(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = workflow.split(f"\n  {name}:\n", 1)[1]
    following = re.search(r"^  \S", job, flags=re.MULTILINE)
    return job if following is None else job[: following.start()]


def installed_run():
    """The installed run line as its arguments, stderr path and fallback."""

    (words,) = [
        shlex.split(line)
        for line in ci_job(JOB).splitlines()
        if line.strip().startswith("ai-journey-kernel-metal ")
    ]
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[1:redirect], words[redirect + 1]


def test_ci_gates_the_lab_with_cpu_only_torch_and_numpy():
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
        "${{ runner.temp }}/kernel-metal-evidence/",
        "name: kernel-metal-contracts-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


def test_ci_requires_the_installed_run_to_stop_without_a_report():
    job = [line.strip() for line in ci_job(JOB).splitlines()]
    arguments, refusal = installed_run()
    run = build_parser().parse_args(arguments)

    assert run.command == "run"
    assert run.config == Path(f"$GITHUB_WORKSPACE/{PINNED}")
    assert str(run.output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(run.output)
    checks = [
        "status=0",
        next(line for line in job if line.startswith("ai-journey-kernel-metal ")),
        f'cat "{refusal}"',
        f'test "$status" -eq {EXIT_UNAVAILABLE}',
        f'grep -F "{REFUSAL}" "{refusal}"',
        f'test ! -e "{run.output}"',
    ]
    start = job.index(checks[0])
    assert job[start : start + len(checks)] == checks


def test_the_ci_expectations_match_the_command_on_a_linux_host(tmp_path):
    arguments, _ = installed_run()
    output = tmp_path / "report.json"
    local = [
        str(ROOT / PINNED) if word.endswith(PINNED) else word for word in arguments
    ]
    local[local.index("--output") + 1] = str(output)

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, *local],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith(REFUSAL)
    assert completed.stdout == "torch not imported\n"
    assert not output.exists()
