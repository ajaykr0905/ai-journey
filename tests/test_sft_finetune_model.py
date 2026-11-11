from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_model import (
    build_finetune_model,
    finetune_model_evidence,
)


def test_finetune_model_is_seeded_float64_and_reproducible():
    config = SFTFineTuneConfig()
    left = finetune_model_evidence(build_finetune_model(config))
    right = finetune_model_evidence(build_finetune_model(config))
    assert left == right
    assert left["dtype"] == "torch.float64"
    assert left["device"] == "cpu"
    assert left["parameters"] > 0


def test_finetune_model_identity_changes_with_seed():
    left = finetune_model_evidence(build_finetune_model(SFTFineTuneConfig(seed=68)))
    right = finetune_model_evidence(build_finetune_model(SFTFineTuneConfig(seed=69)))
    assert left["architecture_sha256"] == right["architecture_sha256"]
    assert left["model_sha256"] != right["model_sha256"]
