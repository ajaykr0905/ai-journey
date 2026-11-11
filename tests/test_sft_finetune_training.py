from ai_journey.sft_finetune_contract import SFTFineTuneConfig
from ai_journey.sft_finetune_dataset import FineTuneExample
from ai_journey.sft_finetune_model import build_finetune_model, finetune_model_evidence
from ai_journey.sft_finetune_schedule import build_schedule
from ai_journey.sft_finetune_training import train_finetune_schedule


def test_train_finetune_schedule_is_reproducible_and_accounts_for_exposure():
    config = SFTFineTuneConfig(
        block_size=8, batch_size=2, steps=2, checkpoint_interval=1
    )
    rows = tuple(
        FineTuneExample(f"row-{index}", "train", "prompt answer", ((7, 13),))
        for index in range(3)
    )
    schedule = build_schedule(
        rows,
        block_size=config.block_size,
        steps=config.steps,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    left = build_finetune_model(config)
    right = build_finetune_model(config)
    _, left_trace = train_finetune_schedule(left, schedule, config)
    _, right_trace = train_finetune_schedule(right, schedule, config)
    assert left_trace == right_trace
    assert left_trace["optimizer_steps"] == 2
    assert left_trace["input_tokens"] == 32
    assert left_trace["supervised_tokens"] > 0
    assert (
        left_trace["final_model_sha256"]
        == finetune_model_evidence(left)["model_sha256"]
    )
