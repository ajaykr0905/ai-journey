"""CPU tests of the Day 77 checklist command. No test compiles or runs CUDA."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_checklist_cli
from ai_journey.gpu_checklist_cli import EXIT_INVALID, EXIT_OK, SUMS_FILE, main
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


def test_the_commands_refuse_abbreviated_options():
    with pytest.raises(SystemExit) as raised:
        gpu_checklist_cli.build_parser().parse_args(["sources", "--output", "x"])
    assert raised.value.code == 2
