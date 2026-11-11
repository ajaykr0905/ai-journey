from ai_journey.sft_finetune_checkpoint import (
    load_finetune_checkpoint,
    write_finetune_checkpoint,
)
from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_model import build_finetune_model, finetune_model_evidence
from ai_journey.sft_finetune_optimizer import build_finetune_optimizer


def test_checkpoint_round_trips_model_optimizer_rng_and_provenance(tmp_path):
    config = SFTFineTuneConfig()
    source = build_finetune_model(config)
    source_optimizer = build_finetune_optimizer(source, config)
    path = tmp_path / "checkpoint.pt"
    write_finetune_checkpoint(
        path,
        source,
        source_optimizer,
        config,
        completed_steps=4,
        dataset_sha256="a" * 64,
        schedule_sha256="b" * 64,
    )
    restored = build_finetune_model(SFTFineTuneConfig(seed=69))
    restored_optimizer = build_finetune_optimizer(restored, config)
    evidence = load_finetune_checkpoint(path, restored, restored_optimizer, config)
    assert finetune_model_evidence(restored) == finetune_model_evidence(source)
    assert evidence["completed_steps"] == 4
    assert evidence["dataset_sha256"] == "a" * 64
