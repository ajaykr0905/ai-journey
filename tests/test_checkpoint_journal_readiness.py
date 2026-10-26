"""Readiness cannot damage an existing output or operate through a source import."""

import json
from pathlib import Path

import pytest

import ai_journey.checkpoint_journal_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]


def test_source_import_rejected_before_creating_output(tmp_path):
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="isolated installed package"):
        readiness.run_readiness(
            installed_root=tmp_path, repository=ROOT, output_dir=output
        )
    assert not output.exists()


def test_existing_outputs_are_never_reused(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    (installed / "bin").mkdir(parents=True)
    (installed / "bin/ai-journey-ddp").touch()
    monkeypatch.setattr(readiness, "__file__", str(installed / "readiness.py"))
    output = tmp_path / "retained"
    output.mkdir()
    witness = output / "state.pt"
    witness.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError):
        readiness.run_readiness(
            installed_root=installed, repository=ROOT, output_dir=output
        )
    assert witness.read_bytes() == b"existing user artifact"


def test_ci_runs_installed_fallback_and_shared_checkpoint_safety():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "python -m ai_journey.checkpoint_journal_readiness" in workflow
    for path in (
        "tests/test_checkpoint*.py",
        "tests/test_serialized_optimizer_admission.py",
        "tests/test_training_optimizer_counters.py",
    ):
        assert path in workflow
        assert list(ROOT.glob(path))
    required = json.loads((ROOT / "config/deployment_readiness.json").read_bytes())[
        "required_files"
    ]
    assert "src/ai_journey/checkpoint_journal_readiness.py" in required
    assert "tests/test_checkpoint_journal_readiness.py" in required
