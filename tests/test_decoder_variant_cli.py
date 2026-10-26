from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from ai_journey.gpt2_config import GPT2Config

ROOT = Path(__file__).resolve().parents[1]


def test_installed_script_points_to_comparison_cli() -> None:
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert package["project"]["scripts"]["ai-journey-decoder-compare"] == (
        "ai_journey.decoder_variant_cli:main"
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "ai_journey.decoder_variant_cli", *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, bytes, bytes]:
    train = b"public training bytes for a tiny decoder comparison" * 2
    heldout = b"separate public held out bytes for measured NLL" * 2
    train_path = tmp_path / "train.bin"
    heldout_path = tmp_path / "heldout.bin"
    train_path.write_bytes(train)
    heldout_path.write_bytes(heldout)
    return train_path, heldout_path, train, heldout


def test_cli_writes_cpu_comparison_report_for_explicit_byte_files(
    tmp_path: Path,
) -> None:
    train_path, heldout_path, train, heldout = _inputs(tmp_path)
    report_path = tmp_path / "comparison.json"

    completed = _run_cli(
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 2
    assert report["device"] == "cpu"
    assert set(report["baseline_config"]) == set(GPT2Config.__dataclass_fields__)
    assert report["baseline_config"]["vocab_size"] == 256
    assert report["baseline_config"]["embedding_dim"] == 8
    assert report["baseline_config"]["embedding_dropout"] == 0.0
    assert report["learning_rate"] == 0.001
    assert report["evaluation_windows"] == 4
    assert report["torch_version"] == str(torch.__version__)
    assert report["train_sha256"] == sha256(train).hexdigest()
    assert report["heldout_sha256"] == sha256(heldout).hexdigest()
    assert report["steps"] == 1
    assert [variant["name"] for variant in report["variants"]] == [
        "baseline",
        "rmsnorm",
        "swiglu",
        "rope",
        "combined",
    ]
    assert (
        len({variant["architecture_fingerprint"] for variant in report["variants"]})
        == 5
    )
    assert all(variant["parameter_count"] > 0 for variant in report["variants"])
    assert all(variant["heldout_nll"] > 0 for variant in report["variants"])
    assert train not in report_path.read_bytes()
    assert str(train_path) not in report_path.read_text()


def test_cli_replays_custom_bounded_cpu_settings_byte_for_byte(tmp_path: Path) -> None:
    train_path, heldout_path, _, _ = _inputs(tmp_path)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    flags = (
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--seed",
        "7",
        "--steps",
        "2",
        "--batch-size",
        "3",
        "--learning-rate",
        "0.002",
        "--context-length",
        "6",
        "--embedding-dim",
        "12",
        "--head-count",
        "3",
        "--layer-count",
        "1",
        "--evaluation-windows",
        "2",
    )

    first = _run_cli(*flags, "--output", str(first_path))
    second = _run_cli(*flags, "--output", str(second_path))

    assert first.returncode == second.returncode == 0
    assert first_path.read_bytes() == second_path.read_bytes()
    report = json.loads(first_path.read_text())
    assert (
        report["seed"],
        report["steps"],
        report["batch_size"],
        report["context_length"],
    ) == (
        7,
        2,
        3,
        6,
    )
    assert report["heldout_windows"] == 2
    assert report["learning_rate"] == 0.002
    assert report["evaluation_windows"] == 2


def test_cli_never_overwrites_an_existing_report(tmp_path: Path) -> None:
    train_path, heldout_path, _, _ = _inputs(tmp_path)
    report_path = tmp_path / "comparison.json"
    report_path.write_bytes(b"existing user report")

    completed = _run_cli(
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
    )

    assert completed.returncode != 0
    assert "output already exists" in completed.stderr
    assert report_path.read_bytes() == b"existing user report"


def test_cli_rejects_missing_and_symlinked_inputs(tmp_path: Path) -> None:
    train_path, heldout_path, _, _ = _inputs(tmp_path)
    symlink = tmp_path / "train-link.bin"
    symlink.symlink_to(train_path)

    for invalid in (tmp_path / "missing.bin", symlink):
        report_path = tmp_path / "comparison.json"
        completed = _run_cli(
            "--train",
            str(invalid),
            "--heldout",
            str(heldout_path),
            "--output",
            str(report_path),
        )

        assert completed.returncode != 0
        assert "regular non-symlink file" in completed.stderr
        assert not report_path.exists()


def test_cli_rejects_identical_training_and_heldout_bytes(tmp_path: Path) -> None:
    train_path, heldout_path, train, _ = _inputs(tmp_path)
    heldout_path.write_bytes(train)
    report_path = tmp_path / "comparison.json"

    completed = _run_cli(
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
    )

    assert completed.returncode != 0
    assert "must differ" in completed.stderr
    assert not report_path.exists()


@pytest.mark.parametrize(
    ("flag_values", "message"),
    [
        (("--seed", "-1"), "seed must be"),
        (("--steps", "0"), "steps must be"),
        (("--batch-size", "0"), "batch_size must be"),
        (("--learning-rate", "nan"), "learning_rate must be"),
        (("--context-length", "1"), "context_length must be"),
        (("--embedding-dim", "65"), "embedding_dim must be"),
        (("--head-count", "3"), "divisible by head_count"),
        (("--embedding-dim", "6"), "even head_dim"),
        (
            (
                "--steps",
                "64",
                "--batch-size",
                "32",
                "--context-length",
                "64",
                "--embedding-dim",
                "64",
                "--layer-count",
                "2",
            ),
            "exceeds 200000 work units",
        ),
    ],
)
def test_cli_rejects_invalid_or_unbounded_settings(
    tmp_path: Path, flag_values: tuple[str, ...], message: str
) -> None:
    train_path, heldout_path, _, _ = _inputs(tmp_path)
    report_path = tmp_path / "comparison.json"

    completed = _run_cli(
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
        *flag_values,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not report_path.exists()


def test_cli_rejects_oversized_input_without_creating_report(tmp_path: Path) -> None:
    train_path, heldout_path, _, _ = _inputs(tmp_path)
    train_path.write_bytes(b"a" * (1024 * 1024 + 1))
    report_path = tmp_path / "comparison.json"

    completed = _run_cli(
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
    )

    assert completed.returncode != 0
    assert "input exceeds" in completed.stderr
    assert not report_path.exists()
