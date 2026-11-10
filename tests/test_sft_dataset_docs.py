from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sft_dataset_guide_documents_source_license_outputs_and_commands():
    guide = (ROOT / "docs/sft-dataset-construction.md").read_text()
    assert "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a" in guide
    assert "Creative Commons Attribution-ShareAlike 3.0 Unported" in guide
    assert "dataset.jsonl" in guide
    assert "chatml.jsonl" in guide
    assert "manifest.json" in guide
    assert "ai-journey-sft-dataset build" in guide
    assert "ai-journey-sft-dataset verify" in guide
    assert "ai-journey-sft-dataset replay" in guide


def test_sft_dataset_guide_preserves_honest_evidence_boundaries():
    guide = (ROOT / "docs/sft-dataset-construction.md").read_text()
    assert "do not prove that Ajay watched" in guide
    assert "personally reviewed the 500 selected examples" in guide
    assert "trained a model" in guide
    assert "Learner-complete evidence still requires Ajay" in guide


def test_readme_exposes_the_day_67_operator_without_claiming_completion():
    readme = (ROOT / "README.md").read_text()
    assert "through Day 67" in readme
    assert "[SFT dataset guide](docs/sft-dataset-construction.md)" in readme
    assert "Workbook entries are planning inputs, not proof" in readme
