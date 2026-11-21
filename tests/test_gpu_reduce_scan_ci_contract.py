"""The Linux CI job of the Day 78 reduction and scan labs.

The job ``gpu-reduce-scan-contracts`` installs CPU-only torch, so the tests
and the installed Apple GPU commands meet the path where MPS is not
available. Each installed run must stop with exit code 3 and write nothing.
The installed verifiers must accept the recorded Apple GPU (Metal) reports,
bound to their recorded configs and, for the reduction, to the recorded Day
74 ceiling. The job ``nvidia-preflight-contracts`` also runs these tests,
because its globs cover every ``gpu*`` file.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_reduce_cli, gpu_scan_cli
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_reduce_recorded import RECORDED_CONFIG as REDUCE_CONFIG
from test_gpu_reduce_recorded import RECORDED_REPORT_JSON as REDUCE_JSON
from test_gpu_scan_recorded import RECORDED_CONFIG as SCAN_CONFIG
from test_gpu_scan_recorded import RECORDED_REPORT_JSON as SCAN_JSON
from test_gpu_timing_ci_contract import refusal_message
from test_matmul_roofline_ci_contract import ON_LINUX, ci_job

ROOT = Path(__file__).resolve().parents[1]
JOB = "gpu-reduce-scan-contracts"
# The Day 77 job was the last job before this one was appended.
PREVIOUS = "matmul-checklist-contracts"
LABS = ("gpu_reduce", "gpu_scan")
SOURCES = " ".join(f"src/ai_journey/{lab}_*.py" for lab in LABS)
TESTS = " ".join(f"tests/test_{lab}_*.py" for lab in LABS)
EVIDENCE = "$RUNNER_TEMP/gpu-reduce-scan-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
CEILINGS = EVIDENCE + "recorded-gpu-ceilings.json"
# Each installed Apple GPU command: its CLI module, the prefix of its refusal
# message and the pinned config that its run must refuse on the Linux runner.
GPU_COMMANDS = {
    "ai-journey-gpu-reduce": (
        gpu_reduce_cli,
        "gpu-reduce run",
        "config/day-78-gpu-reduce.json",
    ),
    "ai-journey-gpu-scan": (
        gpu_scan_cli,
        "gpu-scan run",
        "config/day-78-gpu-scan.json",
    ),
}
# Each file that a job line writes from a test module: the module, the name
# it imports, the file in the evidence directory and the exact text.
WRITTEN = {
    ("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-gpu-ceilings.json",
        CEILING_JSON + "\n",
    ),
    ("test_gpu_reduce_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-gpu-reduce.json",
        REDUCE_JSON + "\n",
    ),
    ("test_gpu_reduce_recorded", "RECORDED_CONFIG"): (
        "recorded-gpu-reduce-config.json",
        json.dumps(REDUCE_CONFIG) + "\n",
    ),
    ("test_gpu_scan_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-gpu-scan.json",
        SCAN_JSON + "\n",
    ),
    ("test_gpu_scan_recorded", "RECORDED_CONFIG"): (
        "recorded-gpu-scan-config.json",
        json.dumps(SCAN_CONFIG) + "\n",
    ),
}
# One fresh interpreter runs each writer of the job with its own argument, as
# on the Linux runner: platform.system answers Linux, so the Metal probe that
# the test modules run at import does not load torch.
WRITE_AS_LINUX = """
import platform
import sys
platform.system = lambda: "Linux"
for code, target in zip(sys.argv[1::2], sys.argv[2::2]):
    sys.argv = ["-c", target]
    exec(compile(code, "-c", "exec"), {"__name__": "__main__"})
"""


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


def writers():
    """Map (module, name) of each ``python -c`` writer to its code and target."""

    found = {}
    for words in command_lines(["python", "-c"]):
        source = re.search(r"from (test_\w+) import (\w+);", words[2])
        if source is not None:
            assert source.groups() not in found, source.groups()
            found[source.groups()] = (words[2], words[3])
    return found


def installed_run(command):
    """One refused run line as its parser arguments and its stderr file."""

    (words,) = [words for words in command_lines([command, "run"]) if "2>" in words]
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[1:redirect], words[redirect + 1]


def local(word, evidence):
    """Map a runner path of the job to this checkout or to a test directory."""

    if word.startswith(WORKSPACE):
        return str(ROOT / word.removeprefix(WORKSPACE))
    if word.startswith(EVIDENCE):
        return str(evidence / word.removeprefix(EVIDENCE))
    return word


def write(keys, evidence):
    """Run the job writers of ``keys`` as on the runner, into ``evidence``."""

    found = writers()
    argv = []
    for key in keys:
        code, target = found[key]
        argv += [code, local(target, evidence)]
    completed = subprocess.run(
        [sys.executable, "-c", WRITE_AS_LINUX, *argv],
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


def test_the_day_78_job_follows_the_checklist_job_in_the_workflow():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(PREVIOUS) + 1] == JOB


def test_ci_gates_both_labs_with_cpu_only_torch_and_numpy():
    job = ci_job(JOB)

    assert "timeout-minutes: 15" in job
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
        "${{ runner.temp }}/gpu-reduce-scan-evidence/",
        "name: gpu-reduce-scan-contracts-evidence",
    ):
        assert command in job, command
    assert Path(__file__).resolve() in set(ROOT.glob("tests/test_gpu_reduce_*.py"))
    for lab in LABS:
        assert list(ROOT.glob(f"src/ai_journey/{lab}_*.py")), lab
        assert list(ROOT.glob(f"tests/test_{lab}_*.py")), lab
    # The reduction globs also hold the CUDA texts of that lab.
    assert ROOT / "src/ai_journey/gpu_reduce_cuda_cli.py" in set(
        ROOT.glob("src/ai_journey/gpu_reduce_*.py")
    )


@pytest.mark.parametrize("command", GPU_COMMANDS)
def test_ci_requires_each_installed_gpu_run_to_stop_without_a_report(command):
    module, prefix, pinned = GPU_COMMANDS[command]
    job = lines()
    arguments, refusal = installed_run(command)
    parsed = module.build_parser().parse_args(arguments)
    start = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"{command} run ") and "2>" in line
    )

    assert parsed.command == "run"
    assert parsed.config == Path(WORKSPACE + pinned)
    assert (ROOT / pinned).is_file()
    assert str(parsed.output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(parsed.output)
    assert job[start - 1] == "status=0"
    assert job[start + 1 : start + 5] == [
        f'cat "{refusal}"',
        f'test "$status" -eq {module.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message(prefix)}" "{refusal}"',
        f'test ! -e "{parsed.output}"',
    ]
    assert module.EXIT_UNAVAILABLE == 3
    if command == "ai-journey-gpu-reduce":
        # The reduction run reads a verified ceiling before it probes MPS, so
        # a line before it writes the recorded Day 74 ceiling.
        assert str(parsed.ceilings) == CEILINGS
        ceiling = writers()[("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON")]
        assert ceiling[1] == CEILINGS
        assert any(ceiling[0] in line for line in job[:start])


@pytest.mark.parametrize("command", GPU_COMMANDS)
def test_the_ci_expectations_match_each_command_on_a_linux_host(command, tmp_path):
    module, prefix, _ = GPU_COMMANDS[command]
    arguments, _ = installed_run(command)
    if command == "ai-journey-gpu-reduce":
        write([("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON")], tmp_path)
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


def test_ci_verifies_each_recorded_report_with_its_config_and_ceiling(tmp_path, capsys):
    found = writers()
    (reduce_verify,) = command_lines(["ai-journey-gpu-reduce", "verify"])
    (scan_verify,) = command_lines(["ai-journey-gpu-scan", "verify"])
    reduce = gpu_reduce_cli.build_parser().parse_args(reduce_verify[1:])
    scan = gpu_scan_cli.build_parser().parse_args(scan_verify[1:])
    paths = {key: EVIDENCE + name for key, (name, _) in WRITTEN.items()}

    assert {key: target for key, (_, target) in found.items()} == paths
    assert (str(reduce.report), str(reduce.config), str(reduce.ceilings)) == (
        paths[("test_gpu_reduce_recorded", "RECORDED_REPORT_JSON")],
        paths[("test_gpu_reduce_recorded", "RECORDED_CONFIG")],
        CEILINGS,
    )
    assert (str(scan.report), str(scan.config)) == (
        paths[("test_gpu_scan_recorded", "RECORDED_REPORT_JSON")],
        paths[("test_gpu_scan_recorded", "RECORDED_CONFIG")],
    )
    # Each writer runs here as on the runner and writes the exact text.
    write(list(WRITTEN), tmp_path)
    for name, text in WRITTEN.values():
        assert (tmp_path / name).read_text(encoding="ascii") == text, name
    # Then each installed verify command accepts its report as a measurement.
    ceiling = json.loads(CEILING_JSON)["report_sha256"]
    for module, argv, prefix, canonical in (
        (gpu_reduce_cli, reduce_verify[1:], "verified_gpu_reduce", REDUCE_JSON),
        (gpu_scan_cli, scan_verify[1:], "verified_gpu_scan", SCAN_JSON),
    ):
        assert module.main([local(word, tmp_path) for word in argv]) == 0, argv
        (summary,) = capsys.readouterr().out.splitlines()
        digest = json.loads(canonical)["report_sha256"]
        assert summary.startswith(f"{prefix}_measurement={digest} "), summary
        if module is gpu_reduce_cli:
            assert f" ceiling={ceiling[:12]} fraction_measured=true " in summary


def test_the_nvidia_preflight_job_still_runs_every_day_78_test():
    job = ci_job("nvidia-preflight-contracts")

    # The Day 78 tests make this job run longer, so its limit is the 15
    # minutes of the Day 74 to 77 Metal lab jobs.
    assert "timeout-minutes: 15" in job
    for lab in LABS:
        sources = set(ROOT.glob(f"src/ai_journey/{lab}_*.py"))
        tests = set(ROOT.glob(f"tests/test_{lab}_*.py"))
        assert sources and tests
        assert sources <= set(ROOT.glob("src/ai_journey/gpu*.py"))
        assert tests <= set(ROOT.glob("tests/test_gpu*.py"))
    for command in (
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "ruff check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "ruff format --check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m compileall -q src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m pytest -q tests/test_gpu*.py",
    ):
        assert command in job, command
