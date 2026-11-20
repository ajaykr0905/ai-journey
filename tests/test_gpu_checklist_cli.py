"""CPU tests of the Day 77 checklist command. No test compiles or runs CUDA."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_checklist_cli
from ai_journey.gpu_checklist_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    SUMS_FILE,
    main,
)
from ai_journey.gpu_checklist_cuda import source_files, source_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = [
    "coarsened_matmul_t16_c2.cu",
    "coarsened_matmul_t16_c4.cu",
    "coarsened_matmul_t32_c2.cu",
    "blocked_matmul_m64_n64_k16_t4x4.cu",
    "blocked_matmul_m128_n128_k8_t8x8.cu",
    "sgemm_row_major.cu",
]


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
    for name, text in files.items():
        assert (directory / name).read_text(encoding="ascii") == text
        line = f"cuda_source={name} sha256={source_sha256(text)} compiled=false"
        assert line in lines
    assert len(lines) == len(SOURCE_NAMES)


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
            [sys.executable, "-m", "ai_journey.gpu_checklist_cli", *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    shown = command("--help")
    assert shown.returncode == 0
    assert "No GPU runs" in " ".join(shown.stdout.split())
    assert command().returncode == 2
    assert command("sources").returncode == 2
    assert command("check").returncode == 2


@pytest.mark.parametrize(
    "argv", [["sources", "--output", "x"], ["check", "--source", "x"]]
)
def test_the_commands_refuse_abbreviated_options(argv):
    with pytest.raises(SystemExit) as raised:
        gpu_checklist_cli.build_parser().parse_args(argv)
    assert raised.value.code == 2


def check_lines(directory, capsys, code=EXIT_OK):
    assert main(["check", "--source-dir", str(directory)]) == code
    captured = capsys.readouterr()
    return captured.out.splitlines(), captured.err


def test_check_reads_back_each_written_text_for_the_18_shapes(tmp_path, capsys):
    write_sources(tmp_path, capsys)

    lines, err = check_lines(tmp_path, capsys)
    totals = [18 * 11] * 3 + [18 * 12] * 2 + [18 * 7]
    assert lines == [
        f"cuda_check={name} sha256={source_sha256(text)} pinned=true "
        f"checks={total}/{total} shapes=18 compiled=false"
        for (name, text), total in zip(source_files(), totals)
    ]
    assert err == ""


def test_check_rejects_an_edited_text_and_names_its_failed_checks(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    path = tmp_path / "coarsened_matmul_t16_c4.cu"
    path.write_text(
        path.read_text().replace(
            "            __syncthreads();\n            for", "            for"
        )
    )

    lines, err = check_lines(tmp_path, capsys, EXIT_REJECTED)
    assert " pinned=false checks=180/198 shapes=18 " in lines[1]
    assert all(" pinned=true " in line for line in lines[:1] + lines[2:])
    assert err == (
        "gpu-checklist check rejected: coarsened_matmul_t16_c4.cu fails "
        "coarsened_barriers\n"
    )


def test_check_accepts_new_comments_but_reports_the_new_digest(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    path = tmp_path / "blocked_matmul_m64_n64_k16_t4x4.cu"
    path.write_text("// Copied to the NVIDIA host.\n" + path.read_text())

    lines, _ = check_lines(tmp_path, capsys)
    assert lines[3].endswith(" pinned=false checks=216/216 shapes=18 compiled=false")


def test_check_fails_on_a_missing_or_non_ascii_file(tmp_path, capsys):
    write_sources(tmp_path, capsys)
    (tmp_path / "sgemm_row_major.cu").unlink()
    assert main(["check", "--source-dir", str(tmp_path)]) == EXIT_INVALID
    assert "gpu-checklist check failed: sgemm_row_major.cu must be" in (
        capsys.readouterr().err
    )

    write_sources(tmp_path, capsys)
    (tmp_path / "coarsened_matmul_t16_c2.cu").write_bytes(b"// caf\xc3\xa9\n")
    assert main(["check", "--source-dir", str(tmp_path)]) == EXIT_INVALID
    assert "must be ASCII text" in capsys.readouterr().err
