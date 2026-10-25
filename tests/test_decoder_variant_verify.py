from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def comparison(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_path = tmp_path / "train.bin"
    heldout_path = tmp_path / "heldout.bin"
    report_path = tmp_path / "comparison.json"
    train_path.write_bytes(b"public training bytes for replay verification" * 2)
    heldout_path.write_bytes(
        b"distinct public heldout bytes for replay verification" * 2
    )
    completed = _run(
        "ai_journey.decoder_variant_cli",
        "--train",
        str(train_path),
        "--heldout",
        str(heldout_path),
        "--output",
        str(report_path),
        "--seed",
        "7",
        "--evaluation-windows",
        "3",
    )
    assert completed.returncode == 0, completed.stderr
    return report_path, train_path, heldout_path


def _verify(
    report: Path, train: Path, heldout: Path
) -> subprocess.CompletedProcess[str]:
    return _run(
        "ai_journey.decoder_variant_verify",
        "--report",
        str(report),
        "--train",
        str(train),
        "--heldout",
        str(heldout),
    )


def test_installed_script_points_to_read_only_verifier() -> None:
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert package["project"]["scripts"]["ai-journey-decoder-verify"] == (
        "ai_journey.decoder_variant_verify:main"
    )


def test_verifier_replays_valid_report_without_modifying_artifacts(
    comparison: tuple[Path, Path, Path],
) -> None:
    report, train, heldout = comparison
    originals = tuple(path.read_bytes() for path in (report, train, heldout))

    completed = _verify(report, train, heldout)

    assert completed.returncode == 0, completed.stderr
    assert "verified=" in completed.stdout
    assert "variants=5" in completed.stdout
    assert tuple(path.read_bytes() for path in (report, train, heldout)) == originals


def test_verifier_rejects_tampered_schema_metadata_and_measurements(
    comparison: tuple[Path, Path, Path],
) -> None:
    report_path, train, heldout = comparison
    report = json.loads(report_path.read_text())
    cases = (
        ("schema_version", 1, "schema_version"),
        ("torch_version", "0.0", "torch_version"),
        ("seed", 2**64, "seed"),
        ("steps", 65, "steps"),
        ("learning_rate", 0.2, "learning_rate"),
        ("evaluation_windows", 257, "evaluation_windows"),
        ("baseline_config.embedding_dim", 128, "embedding_dim"),
        ("baseline_config.vocab_size", 257, "baseline_config"),
        ("variants.0.name", "other", "variants"),
        ("variants.0.architecture_fingerprint", "0" * 64, "fingerprint"),
        ("variants.0.parameter_count", -1, "parameter_count"),
        ("variants.0.heldout_nll", 0.0, "deterministic replay"),
        ("batch_offsets_sha256", "0" * 64, "deterministic replay"),
        ("train_sha256", "0" * 64, "train_sha256"),
    )
    for index, (path, value, message) in enumerate(cases):
        changed = copy.deepcopy(report)
        target = changed
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[int(part) if part.isdigit() else part]
        target[parts[-1]] = value
        tampered_path = report_path.with_name(f"tampered-{index}.json")
        tampered_path.write_text(json.dumps(changed))
        before = tampered_path.read_bytes()

        completed = _verify(tampered_path, train, heldout)

        assert completed.returncode != 0, path
        assert message in completed.stderr, (path, completed.stderr)
        assert tampered_path.read_bytes() == before

    duplicate_path = report_path.with_name("duplicate-key.json")
    duplicate_path.write_text('{"schema_version":2,' + report_path.read_text()[1:])
    duplicate_result = _verify(duplicate_path, train, heldout)
    assert duplicate_result.returncode != 0
    assert "duplicate JSON key" in duplicate_result.stderr


def test_verifier_rejects_changed_input_bytes(
    comparison: tuple[Path, Path, Path],
) -> None:
    report, train, heldout = comparison
    original_report = report.read_bytes()
    for input_path, digest_name in (
        (train, "train_sha256"),
        (heldout, "heldout_sha256"),
    ):
        original_input = input_path.read_bytes()
        input_path.write_bytes(original_input + b"changed")
        completed = _verify(report, train, heldout)
        assert completed.returncode != 0
        assert digest_name in completed.stderr
        input_path.write_bytes(original_input)
    assert report.read_bytes() == original_report


def test_verifier_rejects_symlinked_and_oversized_reports(
    comparison: tuple[Path, Path, Path],
) -> None:
    report, train, heldout = comparison
    symlink = report.with_name("report-link.json")
    symlink.symlink_to(report)
    oversized = report.with_name("oversized.json")
    oversized.write_bytes(b" " * (64 * 1024 + 1))

    linked_result = _verify(symlink, train, heldout)
    oversized_result = _verify(oversized, train, heldout)

    assert linked_result.returncode != 0
    assert "regular non-symlink" in linked_result.stderr
    assert oversized_result.returncode != 0
    assert "report exceeds" in oversized_result.stderr
