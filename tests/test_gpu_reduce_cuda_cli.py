"""CPU tests of the Day 78 CUDA reduction command. No test compiles or runs CUDA."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_reduce_cuda_cli
from ai_journey.gpu_reduce_cuda import source_files, source_sha256
from ai_journey.gpu_reduce_cuda_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    SUMS_FILE,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ["reduce_r1_shared.cu", "reduce_warp_shuffle.cu"]


def write_sources(directory, capsys):
    assert main(["sources", "--output-dir", str(directory)]) == EXIT_OK
    return capsys.readouterr().out.splitlines()


def test_sources_writes_each_text_and_its_digest(tmp_path, capsys):
    directory = tmp_path / "cuda"

    lines = write_sources(directory, capsys)
    files = dict(source_files())
    assert list(files) == SOURCE_NAMES
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [*SOURCE_NAMES, SUMS_FILE]
    )
    sums = (directory / SUMS_FILE).read_text(encoding="ascii")
    assert sums == "".join(
        f"{source_sha256(text)}  {name}\n" for name, text in files.items()
    )
    assert lines == [
        f"cuda_source={name} sha256={source_sha256(text)} compiled=false"
        for name, text in files.items()
    ]
    for name, text in files.items():
        assert (directory / name).read_text(encoding="ascii") == text


def test_sources_replaces_old_files_in_place(tmp_path, capsys):
    (tmp_path / SOURCE_NAMES[0]).write_text("// an old copy\n")

    write_sources(tmp_path, capsys)
    assert (tmp_path / SOURCE_NAMES[0]).read_text() == dict(source_files())[
        SOURCE_NAMES[0]
    ]


def test_sources_refuses_a_file_or_a_symlink_as_the_directory(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)

    for target in (blocker, link):
        assert main(["sources", "--output-dir", str(target)]) == EXIT_INVALID
        assert "must be a directory, not a symlink" in capsys.readouterr().err
    assert blocker.read_text() == "x"


def test_the_module_runs_as_a_command_with_usage_errors_as_code_2():
    def command(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "ai_journey.gpu_reduce_cuda_cli", *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    shown = command("--help")
    assert shown.returncode == 0
    assert "No GPU runs" in " ".join(shown.stdout.split())
    assert shown.stdout.startswith("usage: ai-journey-gpu-reduce-cuda ")
    assert command().returncode == 2
    assert command("sources").returncode == 2
    assert command("check").returncode == 2


@pytest.mark.parametrize(
    "argv", [["sources", "--output", "x"], ["check", "--source", "x"]]
)
def test_the_commands_refuse_abbreviated_options(argv):
    with pytest.raises(SystemExit) as raised:
        gpu_reduce_cuda_cli.build_parser().parse_args(argv)
    assert raised.value.code == 2


def check_lines(directory, capsys, code=EXIT_OK):
    assert main(["check", "--source-dir", str(directory)]) == code
    captured = capsys.readouterr()
    return captured.out.splitlines(), captured.err


def test_check_reads_back_each_written_text_for_the_14_sizes(tmp_path, capsys):
    write_sources(tmp_path, capsys)

    lines, err = check_lines(tmp_path, capsys)
    totals = [14 * 13, 14 * 14]
    assert lines == [
        f"cuda_check={name} sha256={source_sha256(text)} pinned=true "
        f"checks={total}/{total} sizes=14 compiled=false"
        for (name, text), total in zip(source_files(), totals)
    ]
    assert err == ""


def test_check_rejects_an_edited_text_and_names_its_failed_checks(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    path = tmp_path / "reduce_warp_shuffle.cu"
    path.write_text(
        path.read_text().replace("w, 1u));\n", "w, 1u));\n        __syncthreads();\n")
    )

    lines, err = check_lines(tmp_path, capsys, EXIT_REJECTED)
    assert " pinned=false checks=168/196 sizes=14 " in lines[1]
    assert " pinned=true " in lines[0]
    assert err == (
        "gpu-reduce-cuda check rejected: reduce_warp_shuffle.cu fails "
        "shuffle_barriers, shuffle_warp_stage\n"
    )


def test_check_accepts_new_comments_but_reports_the_new_digest(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    path = tmp_path / "reduce_r1_shared.cu"
    path.write_text("// Copied to the NVIDIA host.\n" + path.read_text())

    lines, _ = check_lines(tmp_path, capsys)
    assert lines[0].endswith(" pinned=false checks=182/182 sizes=14 compiled=false")


def test_check_fails_on_a_missing_or_non_ascii_file(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    (tmp_path / "reduce_warp_shuffle.cu").unlink()
    assert main(["check", "--source-dir", str(tmp_path)]) == EXIT_INVALID
    assert "gpu-reduce-cuda check failed: reduce_warp_shuffle.cu must be" in (
        capsys.readouterr().err
    )

    write_sources(tmp_path, capsys)
    (tmp_path / "reduce_r1_shared.cu").write_bytes(b"// caf\xc3\xa9\n")
    assert main(["check", "--source-dir", str(tmp_path)]) == EXIT_INVALID
    assert "must be ASCII text" in capsys.readouterr().err
