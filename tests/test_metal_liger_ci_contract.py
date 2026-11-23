"""The Linux CI job of the Day 80 fused kernel lab.

The job ``metal-liger-contracts`` installs CPU-only torch, so the tests and
the installed ``run`` meet the path where MPS is not available. The
installed ``run`` must stop with exit code 3 and write nothing. The installed
``verify`` must accept the recorded Apple GPU (Metal) report, bound to its
recorded config. The fresh ceiling of that report is not in the repository,
so the job checks the ceiling digest that the report names through
``verify`` alone, as the recorded test does.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import metal_liger_cli
from test_gpu_ceiling_recorded import RECORDED_REPORT_JSON as CEILING_JSON
from test_gpu_timing_ci_contract import refusal_message
from test_matmul_roofline_ci_contract import ON_LINUX, ci_job
from test_metal_fused_ci_contract import WRITE_AS_LINUX
from test_metal_liger_recorded import RECORDED_CONFIG, RECORDED_REPORT_JSON, recorded

ROOT = Path(__file__).resolve().parents[1]
JOB = "metal-liger-contracts"
# The Day 80 Triton IR job comes right before this job.
PREVIOUS = "triton-ir-contracts"
SOURCES = "src/ai_journey/metal_liger_*.py"
TESTS = "tests/test_metal_liger_*.py"
EVIDENCE = "$RUNNER_TEMP/metal-liger-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-80-metal-liger.json"
COMMAND = "ai-journey-metal-liger"
CEILINGS = EVIDENCE + "recorded-gpu-ceilings.json"
# Each file that a job line writes from a test module: the module, the name
# it imports, the file in the evidence directory and the exact text.
WRITTEN = {
    ("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-gpu-ceilings.json",
        CEILING_JSON + "\n",
    ),
    ("test_metal_liger_recorded", "RECORDED_REPORT_JSON"): (
        "recorded-metal-liger.json",
        RECORDED_REPORT_JSON + "\n",
    ),
    ("test_metal_liger_recorded", "RECORDED_CONFIG"): (
        "recorded-metal-liger-config.json",
        json.dumps(RECORDED_CONFIG) + "\n",
    ),
}


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def command_lines(start):
    """Each job line that begins with the words ``start``, split as a shell does."""

    return [
        shlex.split(line)
        for line in lines()
        if line.startswith(f"{start[0]} ") and shlex.split(line)[: len(start)] == start
    ]


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


def test_the_liger_job_is_the_last_job_and_follows_the_triton_ir_job():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[-2:] == [PREVIOUS, JOB]


def test_ci_gates_the_liger_lab_with_cpu_only_torch_and_numpy():
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
        "${{ runner.temp }}/metal-liger-evidence/",
        "name: metal-liger-contracts-evidence",
    ):
        assert command in job, command
    assert "TRITON_INTERPRET" not in job
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    guide = " ".join((ROOT / "docs/metal-liger-lab.md").read_text("utf-8").split())
    assert f"The CI job `{JOB}` runs the tests with CPU-only torch." in guide


def test_ci_requires_the_installed_run_to_stop_without_a_report():
    arguments, refusal = installed_run()
    parsed = metal_liger_cli.build_parser().parse_args(arguments)
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
        f'test "$status" -eq {metal_liger_cli.EXIT_UNAVAILABLE}',
        f'grep -F "{refusal_message("metal-liger run")}" "{refusal}"',
        f'test ! -e "{parsed.output}"',
    ]


def test_the_ci_expectations_match_the_run_on_a_linux_host(tmp_path):
    arguments, _ = installed_run()
    write([("test_gpu_ceiling_recorded", "RECORDED_REPORT_JSON")], tmp_path)
    output = tmp_path / "out" / "report.json"
    argv = [local(word, tmp_path) for word in arguments]
    argv[argv.index("--output") + 1] = str(output)

    completed = subprocess.run(
        [sys.executable, "-c", ON_LINUX, metal_liger_cli.__name__, *argv],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert completed.returncode == metal_liger_cli.EXIT_UNAVAILABLE, completed.stderr
    assert completed.stderr.startswith(refusal_message("metal-liger run"))
    assert completed.stdout == "torch not imported\n"
    assert not output.parent.exists()


def test_ci_verifies_the_recorded_report_against_its_config(tmp_path, capsys):
    found = writers()
    (verify,) = command_lines([COMMAND, "verify"])
    parsed = metal_liger_cli.build_parser().parse_args(verify[1:])

    assert {key: target for key, (_, target) in found.items()} == {
        key: EVIDENCE + name for key, (name, _) in WRITTEN.items()
    }
    assert str(parsed.report) == EVIDENCE + "recorded-metal-liger.json"
    assert str(parsed.config) == EVIDENCE + "recorded-metal-liger-config.json"
    assert parsed.ceilings is None
    write(list(WRITTEN), tmp_path)
    for name, text in WRITTEN.values():
        assert (tmp_path / name).read_text(encoding="ascii") == text
    assert metal_liger_cli.main([local(word, tmp_path) for word in verify[1:]]) == 0
    assert capsys.readouterr().out.startswith(
        f"verified_metal_liger_measurement={recorded()['report_sha256']} "
    )
