import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_journey import gpu_tiled_cli
from ai_journey.gpu_tiled_cli import (
    EXIT_INVALID,
    EXIT_OK,
    EXIT_REJECTED,
    SUMS_FILE,
    main,
    source_files,
)
from ai_journey.gpu_tiled_config import GpuTiledPlan
from ai_journey.gpu_tiled_cuda import source_sha256
from ai_journey.gpu_tiled_report import DIGEST_FIELD
from ai_journey.gpu_tiled_verify import load_report
from ai_journey.kernel_launch_io import canonical_json_bytes
from test_gpu_tiled_config import small_mapping, write_plan

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = [
    "tiled_matmul_t8.cu",
    "tiled_matmul_t16.cu",
    "tiled_matmul_t32.cu",
    "sgemm_row_major.cu",
]


def test_run_then_verify_print_one_digest_line(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    output = tmp_path / "out" / "report.json"

    assert main(["run", "--config", str(config), "--output", str(output)]) == EXIT_OK
    built = capsys.readouterr().out.strip()
    report = load_report(output)
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert built == (
        f"gpu_tiled_report={report[DIGEST_FIELD]} plan=small-tiled cases=3 "
        "tiles=2 replay_matches=6/6 traffic_matches=6/6 race_free_runs=6/6 "
        "cuda_checks=66/66 cublas_checks=21/21 occupancy_estimates=4 "
        "simulated_on_cpu=true gpu_executed=false cuda_compiled=false "
        "cublas_called=false"
    )
    for arguments in (
        ["verify", "--report", str(output)],
        ["verify", "--report", str(output), "--config", str(config)],
    ):
        assert main(arguments) == EXIT_OK
        assert capsys.readouterr().out.strip() == built.replace(
            "gpu_tiled_report=", "verified_gpu_tiled_report=", 1
        )


def test_run_uses_the_pinned_plan_without_a_config(tmp_path, capsys, monkeypatch):
    # The pinned plan takes about half a minute; the report test covers it.
    small = GpuTiledPlan.from_mapping(small_mapping(name="pinned-stand-in"))
    monkeypatch.setattr(gpu_tiled_cli, "PINNED_PLAN", small)
    output = tmp_path / "report.json"

    assert main(["run", "--output", str(output)]) == EXIT_OK
    assert " plan=pinned-stand-in " in capsys.readouterr().out
    assert load_report(output, plan=small)["plan"]["name"] == "pinned-stand-in"


def test_run_never_writes_over_its_own_plan(tmp_path, capsys):
    config = write_plan(tmp_path / "plan.json")
    before = config.read_bytes()
    hard = tmp_path / "hard.json"
    os.link(config, hard)
    soft = tmp_path / "soft.json"
    soft.symlink_to(config)

    for target in (config, hard, soft):
        code = main(["run", "--config", str(config), "--output", str(target)])
        assert code == EXIT_INVALID
        assert "must not alias" in capsys.readouterr().err
    assert config.read_bytes() == before


def test_invalid_inputs_fail_and_rejected_reports_are_named(tmp_path, capsys):
    config = tmp_path / "plan.json"
    output = tmp_path / "report.json"
    config.write_bytes(b'{"name":"a","name":"b"}')
    assert main(["run", "--config", str(config), "--output", str(output)]) == 2
    assert "gpu-tiled run failed" in capsys.readouterr().err
    assert main(["verify", "--report", str(tmp_path / "missing.json")]) == 2
    assert "gpu-tiled verify failed" in capsys.readouterr().err

    write_plan(config)
    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    capsys.readouterr()
    output.write_bytes(
        output.read_bytes().replace(b'"race_free":true', b'"race_free":1')
    )
    assert main(["verify", "--report", str(output)]) == EXIT_REJECTED
    assert "gpu-tiled verify rejected" in capsys.readouterr().err
    other = write_plan(tmp_path / "other.json", seed=75)
    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    capsys.readouterr()
    code = main(["verify", "--report", str(output), "--config", str(other)])
    assert code == EXIT_REJECTED
    assert "not built from this plan" in capsys.readouterr().err


def test_sources_writes_each_text_and_its_digest(tmp_path, capsys):
    directory = tmp_path / "cuda"

    assert main(["sources", "--output-dir", str(directory)]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    files = dict(source_files())
    assert list(files) == SOURCE_NAMES
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [*SOURCE_NAMES, SUMS_FILE]
    )
    sums = (directory / SUMS_FILE).read_text(encoding="ascii")
    for name, text in files.items():
        assert (directory / name).read_text(encoding="ascii") == text
        assert f"{source_sha256(text)}  {name}\n" in sums
        line = f"cuda_source={name} sha256={source_sha256(text)} compiled=false"
        assert line in lines
    assert len(lines) == 4

    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert main(["sources", "--output-dir", str(blocker)]) == EXIT_INVALID
    assert "must be a directory" in capsys.readouterr().err


def test_the_module_runs_as_a_command_with_usage_errors_as_code_2():
    def command(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "ai_journey.gpu_tiled_cli", *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )

    shown = command("--help")
    assert shown.returncode == 0
    assert "No GPU runs" in " ".join(shown.stdout.split())
    assert command("run").returncode == 2
    assert command("verify", "--report").returncode == 2


@pytest.mark.parametrize(
    "argv",
    [["run", "--out", "x"], ["verify", "--rep", "x"], ["sources", "--output", "x"]],
)
def test_the_commands_refuse_abbreviated_options(argv):
    with pytest.raises(SystemExit) as raised:
        gpu_tiled_cli.build_parser().parse_args(argv)
    assert raised.value.code == 2
