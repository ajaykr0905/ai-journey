"""The Linux CI job of the Day 79 Triton matmul lab.

The job ``triton-matmul-contracts`` installs CPU-only torch and triton 3.9.0
and sets ``TRITON_INTERPRET=1``. Thus its tests and the installed ``check``
run the pinned text in the Triton CPU interpreter on the Linux runner, and no
test may skip there. The job also runs the Triton texts of the fused softmax
lab in the same interpreter. These are CPU correctness checks: no Triton
kernel runs on a GPU, and nothing is timed. The CI job ``test`` imports every
test module through unittest discovery, so a module that needs triton must
skip there through ``unittest.SkipTest``.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ai_journey import triton_matmul_cli, triton_matmul_runtime
from ai_journey.triton_matmul_kernel import KERNEL_NAME, LAUNCHER_NAME, TRITON_SOURCE
from ai_journey.triton_matmul_runtime import PYPI_VERSION
from test_matmul_roofline_ci_contract import ci_job
from test_triton_matmul_cli import TRITON_LOC, requires_triton
from test_triton_matmul_correctness import StandInBackend
from test_triton_matmul_recorded import RECORDED_REPORT_JSON, RECORDED_SHA256

ROOT = Path(__file__).resolve().parents[1]
JOB = "triton-matmul-contracts"
# The Day 77 job was the last job on main when this job was appended. A Day 78
# job, when the workflow has one, also comes before this job.
DAY_77_JOB = "matmul-checklist-contracts"
DAY_78_JOB = "gpu-reduce-scan-contracts"
SOURCES = "src/ai_journey/triton_matmul_*.py"
TESTS = "tests/test_triton_matmul_*.py"
FUSED_TRITON = "tests/test_metal_fused_triton.py"
EVIDENCE = "$RUNNER_TEMP/triton-matmul-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-79-triton-matmul.json"
COMMAND = "ai-journey-triton-matmul"
JUNIT = EVIDENCE + "pytest.xml"
DISABLED = (
    "triton-matmul check unavailable: TRITON_INTERPRET='0' gives JITFunction, "
    "and this lab runs only the CPU interpreter (interpreter_disabled)"
)
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


def local(word, evidence):
    """Map a runner path of the job to this checkout or to a test directory."""

    if word.startswith(WORKSPACE):
        return str(ROOT / word.removeprefix(WORKSPACE))
    if word.startswith(EVIDENCE):
        return str(evidence / word.removeprefix(EVIDENCE))
    return word


def refused_check():
    """The check with the interpreter turned off: its arguments and stderr file."""

    (words,) = command_lines(["TRITON_INTERPRET=0", COMMAND, "check"])
    redirect = words.index("2>")
    assert words[redirect + 2 :] == ["||", "status=$?"]
    return words[2:redirect], words[redirect + 1]


def no_skip_check():
    """The job line that fails when the JUnit file of pytest counts a skip."""

    (words,) = [
        words for words in command_lines(["python", "-c"]) if "skipped" in words[2]
    ]
    return words[2], words[3:]


def test_the_triton_job_follows_the_day_77_job_and_any_day_78_job():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )
    earlier = DAY_78_JOB if DAY_78_JOB in jobs else DAY_77_JOB

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(JOB) - 1] == earlier
    assert jobs.index(DAY_77_JOB) < jobs.index(JOB)
    assert jobs[jobs.index(JOB) + 1] == "metal-fused-contracts"


def test_ci_runs_the_tests_in_the_interpreter_with_cpu_torch_and_pinned_triton():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    assert '\n    env:\n      TRITON_INTERPRET: "1"\n    steps:\n' in job
    installs = [line for line in lines() if "pip install" in line]
    assert installs[:3] == [
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        f'python -m pip install "numpy>=1.24,<2" triton=={PYPI_VERSION}',
        "python -m pip install -e . --no-deps",
    ]
    assert not [line for line in installs if "matplotlib" in line]
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f'triton.__version__ == "{PYPI_VERSION}"',
        'os.environ["TRITON_INTERPRET"] == "1"',
        "assert not torch.cuda.is_available()",
        f'python -m pytest -q -rs --junitxml="{JUNIT}" {TESTS} {FUSED_TRITON}',
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "${{ runner.temp }}/triton-matmul-evidence/",
        "name: triton-matmul-contracts-evidence",
    ):
        assert command in job, command
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    assert list(ROOT.glob(SOURCES))
    assert (ROOT / FUSED_TRITON).is_file()
    # The rest of the fused softmax lab runs in metal-fused-contracts.
    assert "src/ai_journey/metal_fused" not in job


def test_the_no_skip_line_fails_when_pytest_skipped_a_test(tmp_path):
    code, arguments = no_skip_check()
    (tmp_path / "test_runs.py").write_text("def test_runs():\n    pass\n")
    (tmp_path / "test_skips.py").write_text(
        "import unittest\n\nraise unittest.SkipTest('no triton')\n"
    )
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    assert arguments == [JUNIT]
    for names, status in (
        (["test_runs.py"], 0),
        (["test_runs.py", "test_skips.py"], 1),
    ):
        junit = tmp_path / f"{len(names)}.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
                *(str(tmp_path / name) for name in names),
            ],
            capture_output=True,
            timeout=300,
            check=False,
            cwd=tmp_path,
            env=environment,
        )
        checked = subprocess.run(
            [sys.executable, "-c", code, str(junit)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert checked.returncode == status, (names, checked.stderr)


def test_ci_requires_the_check_to_stop_when_the_interpreter_is_off():
    arguments, refusal = refused_check()
    parsed = triton_matmul_cli.build_parser().parse_args(arguments)
    job = lines()
    begin = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"TRITON_INTERPRET=0 {COMMAND} check ")
    )

    assert parsed.command == "check"
    assert parsed.config == Path(WORKSPACE + PINNED)
    assert str(parsed.output).startswith(EVIDENCE + "refused/")
    assert refusal.startswith(EVIDENCE) and refusal != str(parsed.output)
    assert job[begin - 1] == "status=0"
    assert job[begin + 1 : begin + 5] == [
        f'cat "{refusal}"',
        f'test "$status" -eq {triton_matmul_cli.EXIT_UNAVAILABLE}',
        f'grep -F "{DISABLED}" "{refusal}"',
        f'test ! -e "{EVIDENCE}refused"',
    ]
    assert triton_matmul_cli.EXIT_UNAVAILABLE == 3


def test_the_refusal_is_what_the_check_prints_without_the_interpreter(
    tmp_path, monkeypatch, capsys
):
    class JITFunction:
        """The type that ``triton.jit`` gives when the interpreter is off."""

    namespace = {KERNEL_NAME: JITFunction(), LAUNCHER_NAME: None}
    monkeypatch.setattr(
        triton_matmul_runtime,
        "_import",
        lambda name, reason: SimpleNamespace(__version__=PYPI_VERSION),
    )
    monkeypatch.setattr(
        triton_matmul_runtime, "compile_text", lambda: (TRITON_SOURCE, namespace)
    )
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    arguments, _ = refused_check()

    code = triton_matmul_cli.main([local(word, tmp_path) for word in arguments])

    assert code == triton_matmul_cli.EXIT_UNAVAILABLE
    assert capsys.readouterr() == ("", DISABLED + "\n")
    assert not (tmp_path / "refused").exists()


@requires_triton
def test_the_refusal_is_what_triton_gives_with_the_interpreter_off(tmp_path):
    arguments, _ = refused_check()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            triton_matmul_cli.__name__,
            *(local(word, tmp_path) for word in arguments),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "TRITON_INTERPRET": "0",
        },
    )

    assert completed.returncode == triton_matmul_cli.EXIT_UNAVAILABLE
    assert (completed.stdout, completed.stderr) == ("", DISABLED + "\n")
    assert not (tmp_path / "refused").exists()


def test_ci_checks_the_pinned_plan_and_verifies_both_reports(tmp_path, capsys):
    (loc,) = command_lines([COMMAND, "loc"])
    (grep,) = command_lines(["grep", "-Fx"])
    (check,) = command_lines([COMMAND, "check"])
    verified, recorded = command_lines([COMMAND, "verify"])
    (writer,) = [
        words
        for words in command_lines(["python", "-c"])
        if "from test_triton_matmul_recorded import" in words[2]
    ]
    parser = triton_matmul_cli.build_parser()
    pinned = parser.parse_args(check[1:])
    first = parser.parse_args(verified[1:])
    second = parser.parse_args(recorded[1:])

    assert loc == [COMMAND, "loc", ">", EVIDENCE + "loc.txt"]
    assert grep == ["grep", "-Fx", TRITON_LOC, EVIDENCE + "loc.txt"]
    assert triton_matmul_cli.loc_lines()[0] == TRITON_LOC
    assert pinned.config == first.config == second.config == Path(WORKSPACE + PINNED)
    assert first.report == pinned.output
    assert str(second.report) == writer[3] == EVIDENCE + "recorded-triton-matmul.json"
    # The writer runs here as on the runner and writes the canonical text.
    target = local(writer[3], tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", writer[2], target],
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
    assert Path(target).read_text(encoding="ascii") == RECORDED_REPORT_JSON + "\n"
    assert triton_matmul_cli.main([local(word, tmp_path) for word in recorded[1:]]) == 0
    summary = capsys.readouterr().out
    assert summary.startswith(f"verified_triton_matmul_interpreter={RECORDED_SHA256} ")
    # A host without triton runs the pinned check with the NumPy stand-in, so
    # the report is synthetic, and then the verify line of the job checks it.
    assert (
        triton_matmul_cli.main(
            [local(word, tmp_path) for word in check[1:]],
            backend=StandInBackend(),
            clock=lambda: 1795300000.0,
            load_average=lambda: (1.0, 2.0, 3.0),
        )
        == 0
    )
    synthetic = capsys.readouterr().out.splitlines()[-1]
    assert " cases=9 configs=6 within_bound=54/54 exact_bits=54/54 " in synthetic
    assert triton_matmul_cli.main([local(word, tmp_path) for word in verified[1:]]) == 0
    assert capsys.readouterr().out == f"verified_{synthetic}\n"


def test_the_unittest_discovery_of_the_test_job_skips_the_interpreter_module():
    assert "python -m unittest discover -s tests -v" in ci_job("test")

    completed = subprocess.run(
        [sys.executable, "-c", DISCOVER, "triton", "test_triton_matmul_interpreter.py"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "skipped 'this host has no triton wheel'" in completed.stderr
    assert completed.stderr.rstrip().endswith("OK (skipped=1)"), completed.stderr
