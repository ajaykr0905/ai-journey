"""The Linux CI job of the Day 82 Tensor Core lab.

The job ``triton-tc-contracts`` installs triton 3.9.0 and CPU-only torch and
sets no ``TRITON_INTERPRET``. Thus the compile contract test and the installed
``compile`` run the real compiler for the declared targets sm_80 and sm_90 on
the Linux runner, and no test may skip there. Triton compiles offline: no GPU
runs, and no PTX or cubin is executed. ``TRITON_TC_DUMP_DIR`` makes the
contract test keep every text, and the job uploads that folder with the
reports, so the instruction classes of Day 82 come only from that artifact.
The CI job ``test`` imports every test module through unittest discovery, so
the contract module must skip there through ``unittest.SkipTest``.
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ai_journey import triton_tc_cli
from ai_journey.triton_matmul_runtime import PYPI_VERSION
from ai_journey.triton_tc_report import COMPILER_LABEL, SYNTHETIC_LABEL
from test_matmul_roofline_ci_contract import ci_job
from test_triton_matmul_ci_contract import DISCOVER
from test_triton_tc_report import SyntheticTcBackend

ROOT = Path(__file__).resolve().parents[1]
JOB = "triton-tc-contracts"
# The Day 81 Metal job comes right before this job, and the Day 82 Metal job
# comes right after it.
PREVIOUS = "metal-flash-contracts"
NEXT = "metal-tc-contracts"
SOURCES = "src/ai_journey/triton_tc_*.py"
TESTS = "tests/test_triton_tc_*.py"
CONTRACT = ROOT / "tests/test_triton_tc_compile_contract.py"
EVIDENCE = "$RUNNER_TEMP/triton-tc-evidence/"
UPLOADED = "${{ runner.temp }}/triton-tc-evidence/"
WORKSPACE = "$GITHUB_WORKSPACE/"
PINNED = "config/day-82-triton-tc.json"
COMMAND = "ai-journey-triton-tc"
JUNIT = EVIDENCE + "pytest.xml"
DUMP_VARIABLE = "TRITON_TC_DUMP_DIR"
TEST_DUMP = EVIDENCE + "test-dump"
CLOCK = {"clock": lambda: 1795478400.0, "load_average": lambda: (1.0, 2.0, 3.0)}


def lines():
    return [line.strip() for line in ci_job(JOB).splitlines()]


def command_lines(start):
    """Each job line that begins with the words ``start``, split as a shell does."""

    return [
        shlex.split(line)
        for line in lines()
        if line.startswith(f"{start[0]} ") and shlex.split(line)[: len(start)] == start
    ]


def arguments(words):
    """The command words before the first shell redirection or operator."""

    for index, word in enumerate(words):
        if word in (">", "2>", "||", "&&"):
            return words[:index]
    return words


def local(word, evidence):
    """Map a runner path of the job to this checkout or to a test directory."""

    if word.startswith(WORKSPACE):
        return str(ROOT / word.removeprefix(WORKSPACE))
    if word.startswith(EVIDENCE):
        return str(evidence / word.removeprefix(EVIDENCE))
    return word


def no_skip_check():
    """The job line that fails when the JUnit file of pytest counts a skip."""

    (words,) = [
        words for words in command_lines(["python", "-c"]) if "skipped" in words[2]
    ]
    return words[2], words[3:]


def test_the_triton_tc_job_follows_the_day_81_jobs():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    jobs = re.findall(
        r"^  ([a-z0-9-]+):$", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE
    )

    assert jobs.count(JOB) == 1
    assert jobs[jobs.index(JOB) - 1] == PREVIOUS
    assert jobs[jobs.index(JOB) + 1] == NEXT


def test_ci_compiles_with_the_pinned_triton_and_no_interpreter():
    job = ci_job(JOB)

    assert 'python-version: "3.11"' in job
    installs = [line for line in lines() if "pip install" in line]
    assert installs[:3] == [
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
        f'python -m pip install "numpy>=1.24,<2" triton=={PYPI_VERSION}',
        "python -m pip install -e . --no-deps",
    ]
    assert PYPI_VERSION == "3.9.0"
    assert not [line for line in installs if "matplotlib" in line]
    assert "ruff==0.6.9 pytest==9.0.2 build==1.2.2.post1" in job
    for command in (
        f"ruff check {SOURCES} {TESTS}",
        f"ruff format --check {SOURCES} {TESTS}",
        f"python -m compileall -q {SOURCES} {TESTS}",
        f'triton.__version__ == "{PYPI_VERSION}"',
        'assert "TRITON_INTERPRET" not in os.environ',
        "assert not torch.cuda.is_available()",
        f'--junitxml="{JUNIT}" {TESTS}',
        f'test -s "{TEST_DUMP}/summary.json"',
        "python -m build --wheel",
        "--no-deps --target",
        "working-directory: ${{ runner.temp }}",
        "ai_journey.__file__.startswith",
        "uses: actions/upload-artifact@v4",
        "name: triton-tc-contracts-evidence",
    ):
        assert command in job, command
    # The compiler must run: the interpreter would give no TTGIR or PTX.
    assert "TRITON_INTERPRET:" not in job
    assert "TRITON_INTERPRET=" not in job
    assert Path(__file__).resolve() in set(ROOT.glob(TESTS))
    assert CONTRACT in set(ROOT.glob(TESTS))
    # The Metal half of Day 82 runs in metal-tc-contracts.
    assert "metal_tc" not in job


def test_the_tests_keep_their_dump_in_the_uploaded_evidence():
    (words,) = [
        shlex.split(line) for line in lines() if line.startswith(f"{DUMP_VARIABLE}=")
    ]
    job = ci_job(JOB)
    upload = job.split("uses: actions/upload-artifact@v4", 1)

    assert words[0] == f"{DUMP_VARIABLE}={TEST_DUMP}"
    assert words[1:6] == ["python", "-m", "pytest", "-q", "-rs"]
    assert words[6:] == [f"--junitxml={JUNIT}", TESTS]
    # The contract module reads the folder from this variable.
    assert f'DUMP_VARIABLE = "{DUMP_VARIABLE}"' in CONTRACT.read_text("utf-8")
    assert TEST_DUMP.startswith(EVIDENCE)
    assert "        if: always()\n" in upload[0].rsplit("- name:", 1)[1]
    assert f"            {UPLOADED}\n" in upload[1]
    assert "if-no-files-found: warn" in upload[1]


def test_the_no_skip_line_fails_when_pytest_skipped_a_test(tmp_path):
    code, junit_path = no_skip_check()
    (tmp_path / "test_runs.py").write_text("def test_runs():\n    pass\n")
    (tmp_path / "test_skips.py").write_text(
        "import unittest\n\nraise unittest.SkipTest('no triton')\n"
    )
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    assert junit_path == [JUNIT]
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


def test_ci_fails_the_step_unless_every_compile_passes_the_rule():
    (words,) = command_lines([COMMAND, "compile"])
    job = lines()
    begin = next(
        index
        for index, line in enumerate(job)
        if line.startswith(f"{COMMAND} compile ")
    )
    output = EVIDENCE + "compile.txt"
    errors = EVIDENCE + "compile-stderr.txt"
    load = {"before": [1.0, 2.0, 3.0], "after": [1.0, 2.0, 3.0]}
    compiled = {"report_sha256": "0" * 64, "host_load": load}

    assert words[len(arguments(words)) :] == [
        ">",
        output,
        "2>",
        errors,
        "||",
        "status=$?",
    ]
    assert job[begin - 1] == "status=0"
    assert job[begin + 1 : begin + 5] == [
        f'cat "{output}" "{errors}"',
        f'echo "compile_status=$status" | tee "{EVIDENCE}compile-status.txt"',
        f'test "$status" -eq {triton_tc_cli.EXIT_OK}',
        f'grep -F " label={COMPILER_LABEL} " "{output}"',
    ]
    # A compile that breaks the rule exits 1 and keeps its report, so the
    # upload step still keeps the evidence.
    assert triton_tc_cli.EXIT_REJECTED == 1
    # The grep finds the summary line of a real compile, not of a stand-in.
    real = triton_tc_cli.summary_line(
        "triton_tc", {**compiled, "label": COMPILER_LABEL}
    )
    stand_in = triton_tc_cli.summary_line(
        "triton_tc", {**compiled, "label": SYNTHETIC_LABEL}
    )
    assert f" label={COMPILER_LABEL} " in real
    assert f" label={COMPILER_LABEL} " not in stand_in


def test_ci_compiles_verifies_and_models_with_the_installed_command(tmp_path, capsys):
    (compile_words,) = command_lines([COMMAND, "compile"])
    verify_dump, verify_model = command_lines([COMMAND, "verify"])
    (model_words,) = command_lines([COMMAND, "model"])
    (grep,) = command_lines(["grep", "-Fx"])
    parser = triton_tc_cli.build_parser()
    compiled = parser.parse_args(arguments(compile_words)[1:])
    checked = parser.parse_args(verify_dump[1:])
    modeled = parser.parse_args(arguments(model_words)[1:])
    verified = parser.parse_args(verify_model[1:])

    assert compiled.config == checked.config == Path(WORKSPACE + PINNED)
    assert modeled.config == verified.config == Path(WORKSPACE + PINNED)
    assert (checked.report, checked.dump) == (compiled.output, compiled.dump)
    assert verified.report == modeled.output and verified.dump is None
    for path in (compiled.output, compiled.dump, modeled.output):
        assert str(path).startswith(EVIDENCE)
    assert model_words[len(arguments(model_words)) :] == [">", EVIDENCE + "model.txt"]
    assert grep[3:] == [EVIDENCE + "model.txt"]
    # The job lines run here with the SYNTHETIC backend of the tests. Its
    # texts are hand-built, not compiler output, so the report is synthetic.
    code = triton_tc_cli.main(
        [local(word, tmp_path) for word in arguments(compile_words)[1:]],
        backend=SyntheticTcBackend(),
        **CLOCK,
    )
    synthetic = capsys.readouterr().out.splitlines()[-1]
    assert code == triton_tc_cli.EXIT_OK
    assert f" label={SYNTHETIC_LABEL} " in synthetic
    assert triton_tc_cli.main([local(word, tmp_path) for word in verify_dump[1:]]) == 0
    assert capsys.readouterr().out == f"verified_{synthetic} texts_recounted=true\n"
    code = triton_tc_cli.main(
        [local(word, tmp_path) for word in arguments(model_words)[1:]], **CLOCK
    )
    printed = capsys.readouterr().out.splitlines()
    assert code == triton_tc_cli.EXIT_OK
    assert grep[2] in printed
    assert "device=h100-sxm5-80gb n=8192 " in grep[2]
    assert triton_tc_cli.main([local(word, tmp_path) for word in verify_model[1:]]) == 0
    assert capsys.readouterr().out.startswith("verified_triton_tc_model=")


def test_the_unittest_discovery_of_the_test_job_skips_the_contract_module():
    assert "python -m unittest discover -s tests -v" in ci_job("test")

    completed = subprocess.run(
        [sys.executable, "-c", DISCOVER, "triton", CONTRACT.name],
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
        (ROOT / "docs/triton-tensor-core-lab.md").read_text("utf-8").split()
    )

    assert (
        f"The CI job `{JOB}` runs the Day 82 tests on a Linux runner with triton "
        "3.9.0 and fails on a skipped test." in guide
    )
    assert "That job did not run yet." in guide
    assert (
        f"Ajay must still read the log of the first run of the CI job `{JOB}`." in guide
    )
