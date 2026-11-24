"""The Linux CI job of the Day 81 Triton flash attention lab.

The job ``triton-flash-contracts`` installs CPU-only torch and triton 3.9.0
and sets ``TRITON_INTERPRET=1``. Thus its tests and the installed ``check``
run the pinned text in the Triton CPU interpreter on the Linux runner, and no
test may skip there. These are CPU correctness checks: no Triton kernel runs
on a GPU, and nothing is timed. No recorded interpreter report exists, so
the job verifies only the report that its own ``check`` writes. The CI job
``test`` imports every test module through unittest discovery, so the
interpreter module must skip there through ``unittest.SkipTest``.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from ai_journey import triton_flash_cli, triton_flash_runtime
from ai_journey.triton_flash_kernel import KERNEL_NAME, LAUNCHER_NAME, TRITON_SOURCE
from ai_journey.triton_matmul_runtime import PYPI_VERSION
from test_matmul_roofline_ci_contract import ci_job
from test_triton_flash_correctness import StandInBackend
from test_triton_matmul_ci_contract import DISCOVER
from test_triton_matmul_cli import requires_triton

ROOT = Path(__file__).resolve().parents[1]
JOB = "triton-flash-contracts"
# The Day 80 fused kernel job comes right before this job, and the Day 81
# Metal job comes right after it.
PREVIOUS = "metal-liger-contracts"
NEXT = "metal-flash-contracts"
SOURCES = "src/ai_journey/triton_flash_*.py"
TESTS = "tests/test_triton_flash_*.py"
EVIDENCE = "$RUNNER_TEMP/triton-flash-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-81-triton-flash.json"
COMMAND = "ai-journey-triton-flash"
JUNIT = EVIDENCE + "pytest.xml"
DISABLED = (
    "triton-flash check unavailable: TRITON_INTERPRET='0' gives JITFunction, "
    "and this lab runs only the CPU interpreter (interpreter_disabled)"
)


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


def test_the_triton_flash_job_follows_the_day_80_jobs():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(JOB) - 1] == PREVIOUS
    assert jobs[jobs.index(JOB) + 1] == NEXT


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
        f'python -m pytest -q -rs --junitxml="{JUNIT}" {TESTS}',
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "${{ runner.temp }}/triton-flash-evidence/",
        "name: triton-flash-contracts-evidence",
    ):
        assert command in job, command
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    assert (ROOT / "tests/test_triton_flash_interpreter.py").is_file()
    # The Metal half of Day 81 runs in metal-flash-contracts.
    assert "metal_flash" not in job


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
    parsed = triton_flash_cli.build_parser().parse_args(arguments)
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
        f'test "$status" -eq {triton_flash_cli.EXIT_UNAVAILABLE}',
        f'grep -F "{DISABLED}" "{refusal}"',
        f'test ! -e "{EVIDENCE}refused"',
    ]
    assert triton_flash_cli.EXIT_UNAVAILABLE == 3


def test_the_refusal_is_what_the_check_prints_without_the_interpreter(
    tmp_path, monkeypatch, capsys
):
    class JITFunction:
        """The type that ``triton.jit`` gives when the interpreter is off."""

    namespace = {KERNEL_NAME: JITFunction(), LAUNCHER_NAME: None}
    monkeypatch.setattr(
        triton_flash_runtime,
        "_import",
        lambda name, reason: SimpleNamespace(__version__=PYPI_VERSION),
    )
    monkeypatch.setattr(
        triton_flash_runtime, "compile_text", lambda: (TRITON_SOURCE, namespace)
    )
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    arguments, _ = refused_check()

    code = triton_flash_cli.main([local(word, tmp_path) for word in arguments])

    assert code == triton_flash_cli.EXIT_UNAVAILABLE
    assert capsys.readouterr() == ("", DISABLED + "\n")
    assert not (tmp_path / "refused").exists()


@requires_triton
def test_the_refusal_is_what_triton_gives_with_the_interpreter_off(tmp_path):
    arguments, _ = refused_check()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            triton_flash_cli.__name__,
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

    assert completed.returncode == triton_flash_cli.EXIT_UNAVAILABLE
    assert (completed.stdout, completed.stderr) == ("", DISABLED + "\n")
    assert not (tmp_path / "refused").exists()


def test_ci_checks_the_model_line_and_the_pinned_plan(tmp_path, capsys):
    (model,) = command_lines([COMMAND, "model"])
    (grep,) = command_lines(["grep", "-Fx"])
    (check,) = command_lines([COMMAND, "check"])
    (verified,) = command_lines([COMMAND, "verify"])
    parser = triton_flash_cli.build_parser()
    pinned = parser.parse_args(check[1:])
    verify = parser.parse_args(verified[1:])

    assert model == [COMMAND, "model", ">", EVIDENCE + "model.txt"]
    assert grep[3:] == [EVIDENCE + "model.txt"]
    assert grep[2] in triton_flash_cli.model_lines()[:-1]
    assert "n=8192 d=64" in grep[2] and "memory_ratio=64.50" in grep[2]
    assert pinned.config == verify.config == Path(WORKSPACE + PINNED)
    assert verify.report == pinned.output
    assert str(pinned.output).startswith(EVIDENCE)
    # A host without triton runs the pinned check with the NumPy stand-in, so
    # the report is synthetic, and then the verify line of the job checks it.
    assert (
        triton_flash_cli.main(
            [local(word, tmp_path) for word in check[1:]],
            backend=StandInBackend(),
            clock=lambda: 1795478400.0,
            load_average=lambda: (1.0, 2.0, 3.0),
        )
        == 0
    )
    synthetic = capsys.readouterr().out.splitlines()[-1]
    assert " cases=6 configs=4 within_bound=48/48 " in synthetic
    assert triton_flash_cli.main([local(word, tmp_path) for word in verified[1:]]) == 0
    assert capsys.readouterr().out == f"verified_{synthetic}\n"


def test_the_unittest_discovery_of_the_test_job_skips_the_interpreter_module():
    assert "python -m unittest discover -s tests -v" in ci_job("test")

    completed = subprocess.run(
        [sys.executable, "-c", DISCOVER, "triton", "test_triton_flash_interpreter.py"],
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


def test_the_guide_names_the_job():
    guide = " ".join(
        (ROOT / "docs/triton-flash-attention-lab.md").read_text("utf-8").split()
    )

    assert f"The CI job `{JOB}` runs the interpreter tests on a Linux runner." in guide
