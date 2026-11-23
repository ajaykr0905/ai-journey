"""The Linux CI job of the Day 80 Triton IR lab.

The job ``triton-ir-contracts`` runs in the ``python:3.11-slim`` image that
made the recorded dump, so triton 3.9.0 sits at the install path that every
committed text names. The runner is x86_64 and the dump came from aarch64.
The installed ``check`` prints each regenerated digest beside the recorded
one as data. Then the tests with the real triton run with no skip, and they
require every text and metadata record to match, so a difference between
the two architectures fails the job on purpose. Nothing runs on a GPU.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import triton_ir_cli
from ai_journey.triton_ir_manifest import load_manifest
from test_matmul_roofline_ci_contract import ci_job
from test_triton_ir_runtime import FixtureBackend

ROOT = Path(__file__).resolve().parents[1]
JOB = "triton-ir-contracts"
# The Day 79 fused softmax job was the last job on main when this job was
# appended. The Day 80 fused kernel job follows this job.
PREVIOUS = "metal-fused-contracts"
NEXT = "metal-liger-contracts"
IMAGE = "python:3.11-slim"
SOURCES = "src/ai_journey/triton_ir_*.py"
TESTS = "tests/test_triton_ir_*.py"
EVIDENCE = "$RUNNER_TEMP/triton-ir-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
FIXTURES = "tests/fixtures/day80_triton_ir"
COMMAND = "ai-journey-triton-ir"
JUNIT = EVIDENCE + "pytest.xml"
REGENERATED = EVIDENCE + "regenerated-triton-ir.json"
GUIDE = ROOT / "docs/triton-ir-lab.md"
PTX = ROOT / FIXTURES / "sm_80" / "m32_n64_k32_g8_w2_s5" / "ptx.ptx"
# One fresh interpreter runs the unittest discovery of the CI job ``test`` on
# the Triton IR modules, with triton hidden.
DISCOVER = """
import sys
import unittest
sys.modules["triton"] = None
argv = ["unittest", "discover", "-s", "tests", "-p", sys.argv[1], "-v"]
unittest.main(module=None, argv=argv)
"""


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def command_lines(start):
    """Each job line that begins with the words ``start``, split as a shell does."""

    return [
        shlex.split(line)
        for line in lines()
        if line.startswith(f"{start[0]} ") and shlex.split(line)[: len(start)] == start
    ]


def local(word, evidence):
    """Map a runner path of the job to this checkout or to a test directory."""

    if word.startswith(WORKSPACE):
        return str(ROOT / word.removeprefix(WORKSPACE))
    if word.startswith(EVIDENCE):
        return str(evidence / word.removeprefix(EVIDENCE))
    return word


def arguments(words):
    """The parser arguments of one command line, before any redirection."""

    ends = [index for index, word in enumerate(words) if word in (">", "2>")]
    return words[1 : ends[0] if ends else len(words)]


def python_line(marker):
    (words,) = [
        words for words in command_lines(["python", "-c"]) if marker in words[2]
    ]
    return words[2], words[3:]


def test_the_triton_ir_job_follows_the_day_79_jobs():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(JOB) - 1] == PREVIOUS
    assert jobs[jobs.index(JOB) + 1] == NEXT


def test_ci_runs_in_the_image_of_the_dump_with_the_triton_of_the_dump():
    job = ci_job(JOB)
    manifest = json.loads((ROOT / FIXTURES / "index.json").read_text("utf-8"))
    (path,) = set(
        re.findall(
            r"/usr/local/lib/python3\.11/\S+?/standard\.py", PTX.read_text("utf-8")
        )
    )
    package = path.removesuffix("language/standard.py") + "__init__.py"

    assert f"\n    container: {IMAGE}\n" in job
    assert f"`{IMAGE}` container" in " ".join(GUIDE.read_text("utf-8").split())
    assert "setup-python" not in job
    assert "TRITON_INTERPRET" in job and 'TRITON_INTERPRET: "1"' not in job
    installs = [line for line in lines() if "pip install" in line]
    assert installs[:3] == [
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        f'python -m pip install "numpy>=1.24,<2" triton=={manifest["triton"]}',
        "python -m pip install -e . --no-deps",
    ]
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f'triton.__version__ == "{manifest["triton"]}"',
        f'triton.__file__ == "{package}"',
        'platform.machine() == "x86_64"',
        "python -m build --wheel",
        "--no-deps --target",
        "ai_journey.__file__.startswith",
        "name: triton-ir-contracts-evidence",
    ):
        assert command in job, command
    assert manifest["machine"] == "aarch64"
    # A container job maps the runner paths, so the job names none of them.
    assert "${{ runner.temp }}" not in job and "${{ github.workspace }}" not in job
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))


def test_every_installed_command_reads_the_committed_fixtures():
    parser = triton_ir_cli.build_parser()
    commands = command_lines([COMMAND])
    parsed = [parser.parse_args(arguments(words)) for words in commands]

    assert [words[1] for words in commands] == [
        "summarize",
        "verify",
        "stage-map",
        "check",
        "verify",
    ]
    assert {str(item.fixtures) for item in parsed} == {WORKSPACE + FIXTURES}
    assert parsed[1].report == parsed[0].output
    assert str(parsed[3].output) == str(parsed[4].report) == REGENERATED


def test_ci_reports_a_difference_as_data_and_the_tests_fail_on_it():
    job = lines()
    begin = next(
        i for i, line in enumerate(job) if line.startswith(f"{COMMAND} check ")
    )
    (words,) = command_lines([COMMAND, "check"])
    redirect = words.index("2>")
    stderr = words[redirect + 1]

    assert words[redirect + 2 :] == ["||", "status=$?"]
    assert job[begin - 1] == "status=0"
    assert job[begin + 1 : begin + 3] == [
        f'cat "{stderr}"',
        'test "$status" -eq 0 -o "$status" -eq 1',
    ]
    assert (triton_ir_cli.EXIT_OK, triton_ir_cli.EXIT_REJECTED) == (0, 1)
    # The pytest step comes after the comparison and counts no skip.
    pytest_at = next(i for i, line in enumerate(job) if "-m pytest " in line)
    assert pytest_at > begin
    assert job[pytest_at] == (f'python -m pytest -q -rs --junitxml="{JUNIT}" {TESTS}')
    code, arguments = python_line('"skipped"')
    assert arguments == [JUNIT]
    assert 'suite.get("skipped") == "0"' in code
    # These two tests fail on purpose when one text differs from the dump.
    runtime = (ROOT / "tests/test_triton_ir_runtime.py").read_text("utf-8")
    cli = (ROOT / "tests/test_triton_ir_cli.py").read_text("utf-8")
    assert "def test_the_real_triton_regenerates_the_committed_texts(" in runtime
    assert "assert digest == record.stage(stage).sha256, stage" in runtime
    assert "def test_check_regenerates_every_text_with_the_real_triton(" in cli
    assert 'assert " texts=48/48 metadata=12/12 " in completed.stdout' in cli


def test_the_comparison_line_prints_every_stage_of_a_check_report(tmp_path):
    fixtures = ROOT / FIXTURES
    report = tmp_path / "regenerated-triton-ir.json"
    argv = ["check", "--fixtures", str(fixtures), "--output", str(report)]
    clock = {"clock": lambda: 0.0, "load_average": lambda: (1.0, 2.0, 3.0)}
    assert triton_ir_cli.main(argv, backend=FixtureBackend(), **clock) == 1
    code, arguments = python_line("x86_64_vs_aarch64")
    assert arguments == [REGENERATED, "|", "tee", EVIDENCE + "x86_64-vs-aarch64.txt"]

    printed = subprocess.run(
        [sys.executable, "-c", code, str(report)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    ).stdout.splitlines()

    manifest = load_manifest(fixtures)
    assert len(printed) == 4 * len(manifest.records) == 48
    assert sum(" match=true " in line for line in printed) == 3
    assert printed[0].startswith("triton_ir x86_64_vs_aarch64 sm_80 ")
    assert all(
        re.search(r"recorded=[0-9a-f]{64} regenerated=[0-9a-f]{64}$", line)
        for line in printed
    )


def test_the_unittest_discovery_of_the_test_job_imports_every_module():
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", DISCOVER, "test_triton_ir_*.py"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=ROOT,
        env=environment,
    )

    assert "python -m unittest discover -s tests -v" in ci_job("test")
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.rstrip().endswith("OK")
