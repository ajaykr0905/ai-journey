from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_package_declares_separate_generation_and_verification_commands():
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert package["project"]["scripts"]["ai-journey-hybrid-parallel"] == (
        "ai_journey.hybrid_parallel_cli:main"
    )
    assert package["project"]["scripts"]["ai-journey-hybrid-parallel-verify"] == (
        "ai_journey.hybrid_parallel_verify:main"
    )


def test_isolated_wheel_generates_and_verifies_without_checkout_imports(tmp_path):
    distribution = tmp_path / "dist"
    installed = tmp_path / "installed"
    evidence = tmp_path / "evidence.json"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(distribution)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(distribution.glob("*.whl"))
    if importlib.util.find_spec("pip") is None:
        uv = shutil.which("uv")
        assert uv is not None, "isolated wheel acceptance requires pip or uv"
        install_command = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ]
    else:
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ]
    subprocess.run(
        install_command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    environment = {
        **os.environ,
        "PATH": f"{installed / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(installed),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        str(installed / "bin" / "ai-journey-hybrid-parallel"),
        "--world-size",
        "2",
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "2",
        "--sequence-parallel-size",
        "1",
        "--tensor-shape",
        "2,2",
        "--tensor-values",
        "0,1,2,3",
        "--token-ids",
        "7,8,9",
        "--layer-count",
        "4",
        "--schedule-policy",
        "gpipe",
        "--microbatch-count",
        "2",
        "--element-size-bytes",
        "4",
        "--pipeline-payload-bytes",
        "16",
        "--source-rank",
        "0",
        "--destination-rank",
        "1",
        "--activation-shape",
        "4",
        "--activation-dtype",
        "float32",
        "--payload-hex",
        bytes(range(16)).hex(),
        "--output",
        str(evidence),
    ]
    generated = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert generated.returncode == 0, generated.stderr
    verification = subprocess.run(
        [
            str(installed / "bin" / "ai-journey-hybrid-parallel-verify"),
            "--report",
            str(evidence),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verification.returncode == 0, verification.stderr
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ai_journey, json; print(json.dumps({'path': ai_journey.__file__}))",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(imported.stdout)["path"].startswith(str(installed))
    assert str(ROOT) not in imported.stdout
