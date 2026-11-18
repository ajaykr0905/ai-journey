"""The torch-free CI job of the trace parser, the analyzer and sim-profile.

The job ``profile-sim-cpu`` installs no torch, so its tests prove that the
parser, the analyzer, the verifier and the simulator profiler need none. Its
installed commands then run the pinned sim-profile stage, refuse a capture
and check the recorded Apple GPU (Metal) capture.
"""

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import profile_trace_cli, sim_profile_cli
from test_matmul_roofline_ci_contract import ci_job

ROOT = Path(__file__).resolve().parents[1]
JOB = "profile-sim-cpu"
LABS = ("profile_trace", "sim_profile")
SOURCES = " ".join(f"src/ai_journey/{lab}_*.py" for lab in LABS)
TESTS = " ".join(f"tests/test_{lab}_*.py" for lab in LABS)
EVIDENCE = "$RUNNER_TEMP/profile-sim-evidence/"
SIM_CONFIG = "config/day-75-sim-profile.json"
TRACE_CONFIG = "config/day-75-profile-trace.json"
# The runtime reads the operating system before it imports torch. The Linux
# runner therefore refuses for the operating system, and a Mac with torch
# blocked refuses because torch cannot be imported.
REFUSAL = (
    "profile-trace capture unavailable: Metal kernels run only on macOS "
    "(unsupported_operating_system)"
)
MACOS_REFUSAL = (
    "profile-trace capture unavailable: PyTorch cannot be imported "
    "(torch_not_importable)"
)
WITHOUT_TORCH = """
import importlib
import sys
sys.modules["torch"] = None
sys.exit(importlib.import_module(sys.argv[1]).main(sys.argv[2:]))
"""


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def installed(command):
    return [shlex.split(line) for line in lines() if line.startswith(f"{command} ")]


def test_ci_gates_both_labs_without_torch_or_matplotlib():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    installs = [line.strip() for line in job.splitlines() if "pip install" in line]
    assert installs[:2] == [
        'python -m pip install "numpy>=1.24,<2"',
        "python -m pip install -e . --no-deps",
    ]
    for dependency in ("torch", "matplotlib"):
        assert not [line for line in installs if dependency in line], dependency
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        'sys.exit(importlib.util.find_spec("torch") is not None)',
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f"python -m pytest -q -rs {TESTS}",
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "${{ runner.temp }}/profile-sim-evidence/",
        "name: profile-sim-evidence",
    ):
        assert command in job, command
    for line in job.splitlines():
        if "python -m pytest" in line:
            for pattern in shlex.split(line)[3:]:
                assert pattern.startswith("-") or list(ROOT.glob(pattern)), pattern


def test_ci_runs_and_verifies_the_pinned_stage_with_the_installed_command():
    operations = installed("ai-journey-sim-profile")
    parser = sim_profile_cli.build_parser()

    assert [words[1] for words in operations] == ["run", "verify"]
    run, verify = (parser.parse_args(words[1:]) for words in operations)
    assert run.config == verify.config == Path(f"$GITHUB_WORKSPACE/{SIM_CONFIG}")
    assert run.stage == "round-4"
    assert verify.report == run.output
    assert str(run.output).startswith(EVIDENCE)


def test_ci_requires_the_installed_capture_to_stop_without_a_report(tmp_path):
    (words,) = [
        words for words in installed("ai-journey-profile-trace") if "capture" in words
    ]
    redirect = words.index("2>")
    refusal = words[redirect + 1]
    capture = profile_trace_cli.build_parser().parse_args(words[1:redirect])
    job = lines()
    checks = [
        f'cat "{refusal}"',
        f'test "$status" -eq {profile_trace_cli.EXIT_UNAVAILABLE}',
        f'grep -F "{REFUSAL}" "{refusal}"',
        f'test ! -e "{capture.output_dir}"',
    ]

    assert words[redirect + 2 :] == ["||", "status=$?"]
    assert capture.config == Path(f"$GITHUB_WORKSPACE/{TRACE_CONFIG}")
    assert str(capture.output_dir).startswith(EVIDENCE)
    start = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"{words[0]} capture ")
    )
    assert job[start - 1] == "status=0"
    assert job[start + 1 : start + 1 + len(checks)] == checks
    # The same command, run here with torch blocked, stops the same way.
    output = tmp_path / "capture"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            WITHOUT_TORCH,
            profile_trace_cli.__name__,
            "capture",
            "--config",
            str(ROOT / TRACE_CONFIG),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == profile_trace_cli.EXIT_UNAVAILABLE
    expected = MACOS_REFUSAL if platform.system() == "Darwin" else REFUSAL
    assert completed.stderr.startswith(expected)
    assert not output.exists()


def test_ci_verifies_and_analyzes_the_recorded_capture(tmp_path, capsys):
    (writer,) = [shlex.split(line) for line in lines() if "write_recorded" in line]
    verify, analyze = (
        profile_trace_cli.build_parser().parse_args(words[1:])
        for words in installed("ai-journey-profile-trace")
        if "capture" not in words
    )
    directory = Path(writer[3])

    assert writer[:2] == ["python", "-c"]
    assert str(directory).startswith(EVIDENCE)
    assert (verify.command, analyze.command) == ("verify", "analyze")
    assert verify.report.parent == analyze.trace.parent == directory
    # The writer runs here as on the runner, and both commands read its files.
    local = tmp_path / directory.name
    completed = subprocess.run(
        [sys.executable, "-c", writer[2], str(local)],
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
    report = local / verify.report.name
    trace = local / analyze.trace.name
    assert profile_trace_cli.main(["verify", "--report", str(report)]) == 0
    assert capsys.readouterr().out.startswith("verified_profile_trace_measurement=")
    assert profile_trace_cli.main(["analyze", "--trace", str(trace)]) == 0
