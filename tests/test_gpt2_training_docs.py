from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_guide_documents_reproducible_commands_and_scope():
    guide = (ROOT / "docs" / "gpt2-training-controls.md").read_text(encoding="utf-8")

    for required in (
        "ai-journey-gpt2-train train",
        "ai-journey-gpt2-train verify",
        "ai-journey-gpt2-train replay",
        "64 target",
        "768 target",
        "weight decay",
        "cosine decay",
        "Gradients are clipped",
        "not a model-quality result",
        "GPU, distributed, throughput, production, lecture-viewing",
    ):
        assert required in guide
