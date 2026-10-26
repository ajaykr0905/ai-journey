import os
from pathlib import Path
import sys

import pytest

from ai_journey.gpu_preflight_readiness import _run_command, run_readiness


def test_readiness_refuses_checkout_imports_before_creating_evidence(tmp_path):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="installed wheel"):
        run_readiness(
            installed_root=tmp_path / "not-installed",
            resource_config=tmp_path / "source.json",
            output_dir=output,
        )
    assert not output.exists()


def test_failed_installed_command_retains_both_diagnostic_streams(tmp_path):
    with pytest.raises(RuntimeError, match="exited 7"):
        _run_command(
            [
                sys.executable,
                "-c",
                "import sys; print('public output'); print('public error', file=sys.stderr); sys.exit(7)",
            ],
            name="failure",
            directory=tmp_path,
            environment=os.environ.copy(),
        )
    assert (tmp_path / "failure.stdout.txt").read_text().strip() == "public output"
    assert (tmp_path / "failure.stderr.txt").read_text().strip() == "public error"


def test_command_deadline_keeps_logs_and_stops_the_child(tmp_path):
    with pytest.raises(RuntimeError, match="readiness deadline"):
        _run_command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            name="deadline",
            directory=tmp_path,
            environment=os.environ.copy(),
            timeout=0.1,
        )
    assert (tmp_path / "deadline.stdout.txt").is_file()
    assert (tmp_path / "deadline.stderr.txt").is_file()


def test_ci_builds_isolated_gpu_package_gate_and_retains_blocked_evidence():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    ).read_text()
    for contract in (
        "nvidia-preflight-contracts:",
        "ruff check src/ai_journey/gpu*.py tests/test_gpu*.py",
        "python -m pytest -q tests/test_gpu*.py",
        "python -m build --outdir",
        '--no-deps --target "$RUNNER_TEMP/nvidia-installed"',
        "-m ai_journey.gpu_preflight_readiness",
        "--expect-blocked",
        "name: nvidia-preflight-cpu-integrity",
    ):
        assert contract in workflow
