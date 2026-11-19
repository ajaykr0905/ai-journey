"""The Linux CI job of the tiled matmul lab and its gap.

The job ``matmul-tiled-contracts`` installs CPU-only torch, so the tests and
the installed Apple GPU commands meet the path where MPS is not available.
Each installed GPU command must stop with exit code 3 and write nothing. The
installed package must verify the recorded Apple GPU (Metal) reports and
build their gap again. The Day 76 CUDA contract needs no job of its own,
because the globs of ``nvidia-preflight-contracts`` already cover its files.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import matmul_tiled_cli, matmul_tiled_gap_cli
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_timing_ci_contract import refusal_message
from test_matmul_roofline_ci_contract import ON_LINUX, ci_job
from test_matmul_tiled_gap_recorded import RECORDED_GAP_SHA256, RECORDED_TIMING_JSON
from test_matmul_tiled_recorded import RECORDED_REPORT_JSON as TILED_JSON

ROOT = Path(__file__).resolve().parents[1]
JOB = "matmul-tiled-contracts"
SOURCES = "src/ai_journey/matmul_tiled_*.py"
TESTS = "tests/test_matmul_tiled_*.py"
EVIDENCE = "$RUNNER_TEMP/matmul-tiled-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-76-matmul-tiled.json"
RULE = "config/day-75-matmul-bottleneck.json"
COMMAND = "ai-journey-matmul-tiled"
GAP = ["python", "-m", matmul_tiled_gap_cli.__name__]
# Each Apple GPU command: the words that begin its job line, its CLI module
# and the prefix of its refusal message.
GPU_COMMANDS = {
    "run": ([COMMAND, "run"], matmul_tiled_cli, "matmul-tiled run"),
    "timing": ([*GAP, "timing"], matmul_tiled_gap_cli, "matmul-tiled-gap timing"),
}
# Each recorded report that the job writes: the test module that holds it,
# the file name in the evidence directory and the canonical text.
RECORDED = {
    "test_matmul_tiled_recorded": ("recorded-matmul-tiled.json", TILED_JSON),
    "test_matmul_tiled_gap_recorded": (
        "recorded-one-call-timing.json",
        RECORDED_TIMING_JSON,
    ),
    "test_gpu_ceiling_recorded": ("recorded-gpu-ceilings.json", CEILING_JSON),
}


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def command_lines(start):
    """Each job line that begins with the words ``start``, split as a shell does."""

    found = []
    for line in lines():
        if line.startswith(f"{start[0]} "):
            words = shlex.split(line)
            if words[: len(start)] == start:
                found.append(words)
    return found


def installed_run(name):
    """One refused GPU command as its parser arguments and its stderr file."""

    start, _, _ = GPU_COMMANDS[name]
    (words,) = [words for words in command_lines(start) if "2>" in words]
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[len(start) - 1 : redirect], words[redirect + 1]


def local(word, evidence):
    """Map a runner path of the job to this checkout or to a test directory."""

    if word.startswith(WORKSPACE):
        return str(ROOT / word.removeprefix(WORKSPACE))
    if word.startswith(EVIDENCE):
        return str(evidence / word.removeprefix(EVIDENCE))
    return word


def test_the_tiled_job_is_the_last_job_of_the_workflow():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs[-1] == JOB
    assert jobs.count(JOB) == 1
    assert jobs[jobs.index("moe-cpu-lab") + 1] == "nvidia-preflight-contracts"


def test_ci_gates_the_tiled_lab_with_cpu_only_torch_and_numpy():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    installs = [line for line in lines() if "pip install" in line]
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
        "${{ runner.temp }}/matmul-tiled-evidence/",
        "name: matmul-tiled-contracts-evidence",
    ):
        assert command in job, command
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    assert list(ROOT.glob(SOURCES))


@pytest.mark.parametrize("name", GPU_COMMANDS)
def test_ci_requires_each_installed_gpu_command_to_stop_without_a_report(name):
    start, module, prefix = GPU_COMMANDS[name]
    arguments, refusal = installed_run(name)
    parsed = module.build_parser().parse_args(arguments)
    job = lines()
    begin = next(
        index
        for index, line in enumerate(job)
        if line.startswith(" ".join(start) + " ") and "2>" in line
    )

    assert parsed.command == name
    assert parsed.config == Path(WORKSPACE + PINNED)
    assert str(parsed.output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(parsed.output)
    assert job[begin - 1] == "status=0"
    assert job[begin + 1 : begin + 5] == [
        f'cat "{refusal}"',
        f'test "$status" -eq {module.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message(prefix)}" "{refusal}"',
        f'test ! -e "{parsed.output}"',
    ]
    assert module.EXIT_UNAVAILABLE == 3


@pytest.mark.parametrize("name", GPU_COMMANDS)
def test_the_ci_expectations_match_each_command_on_a_linux_host(name, tmp_path):
    _, module, prefix = GPU_COMMANDS[name]
    arguments, _ = installed_run(name)
    output = tmp_path / "out" / "report.json"
    argv = [local(word, tmp_path) for word in arguments]
    argv[argv.index("--output") + 1] = str(output)

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, module.__name__, *argv],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == module.EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith(refusal_message(prefix))
    assert completed.stdout == "torch not imported\n"
    assert not output.parent.exists()


def test_ci_verifies_the_recorded_reports_and_builds_their_gap_again(tmp_path, capsys):
    writers = {}
    for words in command_lines(["python", "-c"]):
        source = re.search(r"from (test_\w+) import ", words[2])
        if source is not None:
            writers[source.group(1)] = (words[2], words[3])
    paths = {source: EVIDENCE + name for source, (name, _) in RECORDED.items()}
    (tiled_verify,) = command_lines([COMMAND, "verify"])
    (timing_verify,) = command_lines([*GAP, "verify-timing"])
    (gap,) = command_lines([*GAP, "gap"])
    (gap_verify,) = command_lines([*GAP, "verify"])
    parser = matmul_tiled_gap_cli.build_parser()
    report = matmul_tiled_cli.build_parser().parse_args(tiled_verify[1:]).report
    timing = parser.parse_args(timing_verify[3:]).report
    built = parser.parse_args(gap[3:])
    checked = parser.parse_args(gap_verify[3:])

    assert {source: target for source, (_, target) in writers.items()} == paths
    assert str(report) == paths["test_matmul_tiled_recorded"]
    assert str(timing) == paths["test_matmul_tiled_gap_recorded"]
    sources = (built.tiled_report, built.ceiling_report, built.timing_report)
    assert sources == (report, Path(paths["test_gpu_ceiling_recorded"]), timing)
    assert (checked.tiled_report, checked.ceiling_report, checked.timing_report) == (
        sources
    )
    assert built.rule_config == checked.rule_config == Path(WORKSPACE + RULE)
    assert checked.report == built.output
    # Each writer runs here as on the runner and writes the canonical text.
    for source, (code, target) in writers.items():
        completed = subprocess.run(
            [sys.executable, "-c", code, local(target, tmp_path)],
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
        name, canonical = RECORDED[source]
        assert (tmp_path / name).read_text(encoding="ascii") == canonical + "\n"
    # Then each command of the job checks those files and prints the digest.
    steps = (
        (matmul_tiled_cli, tiled_verify[1:], "verified_matmul_tiled", TILED_JSON),
        (
            matmul_tiled_gap_cli,
            timing_verify[3:],
            "verified_matmul_tiled_one_call",
            RECORDED_TIMING_JSON,
        ),
        (matmul_tiled_gap_cli, gap[3:], "matmul_tiled_gap", None),
        (matmul_tiled_gap_cli, gap_verify[3:], "verified_matmul_tiled_gap", None),
    )
    for module, argv, prefix, canonical in steps:
        digest = (
            RECORDED_GAP_SHA256
            if canonical is None
            else json.loads(canonical)["report_sha256"]
        )
        assert module.main([local(word, tmp_path) for word in argv]) == 0, argv
        summary = capsys.readouterr().out.splitlines()[-1]
        assert summary.startswith(f"{prefix}_measurement={digest} "), summary
    assert " sources_checked=true " in summary


def test_the_nvidia_preflight_job_covers_the_tiled_cuda_contract():
    job = ci_job("nvidia-preflight-contracts")
    sources = sorted(ROOT.glob("src/ai_journey/gpu_tiled_*.py"))
    tests = sorted(ROOT.glob("tests/test_gpu_tiled_*.py"))

    assert sources and tests
    assert set(sources) <= set(ROOT.glob("src/ai_journey/gpu*.py"))
    assert set(tests) <= set(ROOT.glob("tests/test_gpu*.py"))
    for command in (
        "ruff check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "ruff format --check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m compileall -q src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m pytest -q tests/test_gpu*.py",
    ):
        assert command in job, command
    assert "matmul_tiled" not in job
