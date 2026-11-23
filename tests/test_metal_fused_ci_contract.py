"""The Linux CI job of the Day 79 fused softmax lab.

The job ``metal-fused-contracts`` installs CPU-only torch, so the tests and
the installed ``run`` meet the path where MPS is not available. The
installed ``run`` must stop with exit code 3 and write nothing. The installed
``verify`` must accept the recorded Apple GPU (Metal) report, bound to its
recorded config and to the recorded Day 74 ceiling. The Triton texts of this
lab need triton and ``TRITON_INTERPRET=1``, so they skip here and run in the
job ``triton-matmul-contracts``. The CI job ``test`` imports every test
module through unittest discovery, so the Triton module must skip there
through ``unittest.SkipTest``.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import metal_fused_cli
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_timing_ci_contract import refusal_message
from test_matmul_roofline_ci_contract import ON_LINUX, ci_job
from test_metal_fused_recorded import RECORDED_CONFIG, RECORDED_REPORT_JSON, recorded

ROOT = Path(__file__).resolve().parents[1]
JOB = "metal-fused-contracts"
# The Triton job of Day 79 comes right before this job.
PREVIOUS = "triton-matmul-contracts"
SOURCES = "src/ai_journey/metal_fused_*.py"
TESTS = "tests/test_metal_fused_*.py"
TRITON_TESTS = "tests/test_metal_fused_triton.py"
EVIDENCE = "$RUNNER_TEMP/metal-fused-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-79-metal-fused.json"
COMMAND = "ai-journey-metal-fused"
CEILINGS = EVIDENCE + "recorded-gpu-ceilings.json"
# Each file that a job line writes from a test module: the module, the name
# it imports, the file in the evidence directory and the exact text.
WRITTEN = {
    ("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-gpu-ceilings.json",
        CEILING_JSON + "\n",
    ),
    ("test_metal_fused_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-metal-fused.json",
        RECORDED_REPORT_JSON + "\n",
    ),
    ("test_metal_fused_recorded", "RECORDED_CONFIG"): (
        "recorded-metal-fused-config.json",
        json.dumps(RECORDED_CONFIG) + "\n",
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
# One fresh interpreter runs the unittest discovery of the CI job ``test`` on
# one module, with each named module hidden.
DISCOVER = """
import sys
import unittest
for name in filter(None, sys.argv[1].split(",")):
    sys.modules[name] = None
argv = ["unittest", "discover", "-s", "tests", "-p", sys.argv[2], "-v"]
unittest.main(module=None, argv=argv)
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


def installed_run():
    """The refused run line as its parser arguments and its stderr file."""

    (words,) = [words for words in command_lines([COMMAND, "run"]) if "2>" in words]
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
    arguments = []
    for key in keys:
        code, target = found[key]
        arguments += [code, local(target, evidence)]
    completed = subprocess.run(
        [sys.executable, "-c", WRITE_AS_LINUX, *arguments],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={
            **os.environ,
            "GITHUB_WORKSPACE": str(ROOT),
            "PYTHONPATH": str(ROOT / "src"),
        },
    )
    assert completed.returncode == 0, completed.stderr


def discover(blocked, environment):
    return subprocess.run(
        [sys.executable, "-c", DISCOVER, blocked, Path(TRITON_TESTS).name],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=ROOT,
        env=environment,
    )


def test_the_fused_job_follows_the_triton_job_in_the_workflow():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(PREVIOUS) + 1] == JOB


def test_ci_gates_the_fused_lab_with_cpu_only_torch_and_numpy():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    installs = [line for line in lines() if "pip install" in line]
    assert installs[:3] == [
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        'python -m pip install "numpy>=1.24,<2"',
        "python -m pip install -e . --no-deps",
    ]
    assert not [line for line in installs if "matplotlib" in line]
    assert not [line for line in installs if "triton" in line]
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
        "${{ runner.temp }}/metal-fused-evidence/",
        "name: metal-fused-contracts-evidence",
    ):
        assert command in job, command
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    assert list(ROOT.glob(SOURCES))
    # Here the Triton module skips. The Triton job runs it in the interpreter.
    assert "TRITON_INTERPRET" not in job
    triton_job = ci_job(PREVIOUS)
    assert '\n    env:\n      TRITON_INTERPRET: "1"\n' in triton_job
    (pytest_line,) = [
        line.strip() for line in triton_job.splitlines() if "-m pytest " in line
    ]
    assert pytest_line.split()[-1] == TRITON_TESTS
    guide = (ROOT / "docs/metal-fused-softmax-lab.md").read_text(encoding="utf-8")
    assert f"The CI job `{PREVIOUS}` runs it with the flag on a Linux runner." in (
        " ".join(guide.split())
    )


def test_ci_requires_the_installed_run_to_stop_without_a_report():
    arguments, refusal = installed_run()
    parsed = metal_fused_cli.build_parser().parse_args(arguments)
    job = lines()
    begin = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"{COMMAND} run ") and "2>" in line
    )
    written = writers()[("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON")][1]

    assert parsed.command == "run"
    assert parsed.config == Path(WORKSPACE + PINNED)
    assert str(parsed.ceilings) == written == CEILINGS
    # The ceiling file is written before the run line that reads it.
    assert job.index(next(line for line in job if CEILINGS in line)) < begin
    assert str(parsed.output).startswith(EVIDENCE)
    assert refusal.startswith(EVIDENCE) and refusal != str(parsed.output)
    assert job[begin - 1] == "status=0"
    assert job[begin + 1 : begin + 5] == [
        f'cat "{refusal}"',
        f'test "$status" -eq {metal_fused_cli.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message("metal-fused run")}" "{refusal}"',
        f'test ! -e "{parsed.output}"',
    ]
    assert metal_fused_cli.EXIT_UNAVAILABLE == 3


def test_the_ci_expectations_match_the_run_on_a_linux_host(tmp_path):
    arguments, _ = installed_run()
    write([("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON")], tmp_path)
    output = tmp_path / "out" / "report.json"
    argv = [local(word, tmp_path) for word in arguments]
    argv[argv.index("--output") + 1] = str(output)

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, metal_fused_cli.__name__, *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == metal_fused_cli.EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith(refusal_message("metal-fused run"))
    assert completed.stdout == "torch not imported\n"
    assert not output.parent.exists()


def test_ci_verifies_the_recorded_report_against_its_config_and_ceiling(
    tmp_path, capsys
):
    found = writers()
    (verify,) = command_lines([COMMAND, "verify"])
    parsed = metal_fused_cli.build_parser().parse_args(verify[1:])

    assert {key: target for key, (_, target) in found.items()} == {
        key: EVIDENCE + name for key, (name, _) in WRITTEN.items()
    }
    assert str(parsed.report) == EVIDENCE + "recorded-metal-fused.json"
    assert str(parsed.config) == EVIDENCE + "recorded-metal-fused-config.json"
    assert str(parsed.ceilings) == CEILINGS
    # Each writer runs here as on the runner and writes the exact text.
    write(list(WRITTEN), tmp_path)
    for name, text in WRITTEN.values():
        assert (tmp_path / name).read_text(encoding="ascii") == text
    # Then the verify line of the job checks the report with both inputs.
    assert metal_fused_cli.main([local(word, tmp_path) for word in verify[1:]]) == 0
    summary = capsys.readouterr().out
    assert summary.startswith(
        f"verified_metal_fused_measurement={recorded()['report_sha256']} "
    )


def test_the_unittest_discovery_of_the_test_job_skips_the_triton_module():
    assert "python -m unittest discover -s tests -v" in ci_job("test")
    environment = {
        key: value for key, value in os.environ.items() if key != "TRITON_INTERPRET"
    }
    environment |= {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}

    # The job test sets no TRITON_INTERPRET, and its torch may bring triton.
    unset = discover("", environment)
    hidden = discover("triton,torch", {**environment, "TRITON_INTERPRET": "1"})

    assert unset.returncode == 0, unset.stderr
    assert "skipped 'the Triton harness runs only with TRITON_INTERPRET=1" in (
        unset.stderr
    )
    assert hidden.returncode == 0, hidden.stderr
    assert "skipped 'the Triton harness needs triton and torch: " in hidden.stderr
    for completed in (unset, hidden):
        assert completed.stderr.rstrip().endswith("OK (skipped=1)")
