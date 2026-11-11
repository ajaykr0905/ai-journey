from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_model import build_finetune_model, finetune_model_evidence
from ai_journey.sft_finetune_schedule import build_schedule
from ai_journey.sft_finetune_training import (
    train_finetune_schedule,
    train_finetune_segment,
)


def test_contiguous_training_segments_match_one_shot_optimization():
    config = SFTFineTuneConfig(
        block_size=8, batch_size=2, steps=4, checkpoint_interval=2
    )
    rows = tuple(
        FineTuneExample(f"row-{index}", "train", "prompt answer", ((7, 13),))
        for index in range(3)
    )
    schedule = build_schedule(
        rows, block_size=8, steps=4, batch_size=2, seed=config.seed
    )
    one_shot = build_finetune_model(config)
    train_finetune_schedule(one_shot, schedule, config)
    resumed = build_finetune_model(config)
    optimizer, _ = train_finetune_segment(resumed, schedule.batches[:2], config)
    train_finetune_segment(resumed, schedule.batches[2:], config, optimizer=optimizer)
    assert finetune_model_evidence(resumed) == finetune_model_evidence(one_shot)
