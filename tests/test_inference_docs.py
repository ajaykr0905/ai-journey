from pathlib import Path


def test_inference_guide_exposes_reproducible_commands_and_evidence_boundaries():
    guide = Path("docs/inference-capacity-planning.md").read_text()

    assert "ai-journey-inference build" in guide
    assert "ai-journey-inference verify" in guide
    assert "256 GiB" in guide
    assert "64 GiB" in guide
    assert "does not run a model server" in guide
    assert "serving cost require an" in guide
    assert "executed workload with hardware" in guide


def test_readme_links_the_inference_capacity_guide():
    readme = Path("README.md").read_text()

    assert "[inference capacity guide](docs/inference-capacity-planning.md)" in readme
